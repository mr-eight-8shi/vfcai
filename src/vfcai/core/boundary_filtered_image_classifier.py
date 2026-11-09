import os
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm import tqdm
from .boundary_filter_model import UnifiedBoundaryRegressor


class BoundaryFieldDataset(Dataset):
    """Memory-efficient dataset loading pre-computed 5-channel boundary fields [5, H, W]."""

    def __init__(self, cached_fields, labels):
        self.fields = cached_fields
        self.labels = labels
        print(f"--- Boundary Field CNN Dataset Initialized (Total Samples: {len(labels)}) ---")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feat_map = torch.from_numpy(self.fields[idx]).float()
        return feat_map, int(self.labels[idx])


class BoundaryFieldCNN(nn.Module):
    """5-Channel Boundary Field CNN Classifier for spatial functional map recognition."""
    def __init__(self, num_classes=20):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class BoundaryFilteredImageClassifier:
    """General-purpose classifier utilizing Boundary Field CNN with unified pipeline."""
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.model_dir = os.path.join(config.space_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "boundary_field_cnn_classifier.pth")
        self.unified_model_path = os.path.join(self.model_dir, "boundary_unified_regressor.pth")
        self.presence_model_path = os.path.join(self.model_dir, "boundary_presence_classifier.pth")
        
        self.field_cache_path = os.path.join(config.space_dir, "cached_boundary_fields.npz")

    def _load_filter_model(self):
        load_path = self.unified_model_path if os.path.exists(self.unified_model_path) else self.presence_model_path
        if not os.path.exists(load_path):
            raise FileNotFoundError("⚠️ Trained unified model not found. Please run training (Option 1) first.")
        model = UnifiedBoundaryRegressor().to(self.device)
        try:
            model.load_state_dict(torch.load(load_path, map_location=self.device, weights_only=True))
        except TypeError:
            model.load_state_dict(torch.load(load_path, map_location=self.device))
        model.eval()
        return model

    def _image_to_field_tensor(self, img_array, filter_model, upsample_scale=3):
        img_uint8 = (img_array * 255).astype(np.uint8) if img_array.max() <= 1.0 else img_array.astype(np.uint8)
        pil_img = Image.fromarray(img_uint8).convert("RGB")
        orig_w, orig_h = pil_img.size
        if upsample_scale > 1:
            pil_img = pil_img.resize((orig_w * upsample_scale, orig_h * upsample_scale), Image.BICUBIC)
        
        W, H = pil_img.size
        tensor_img = torch.tensor(np.array(pil_img, dtype=np.float32) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        padded_img = F.pad(tensor_img, (4, 4, 4, 4), mode='replicate')
        patches_raw = F.unfold(padded_img, kernel_size=9, stride=3)
        L = patches_raw.shape[2]
        out_h = (H - 1) // 3 + 1
        out_w = (W - 1) // 3 + 1

        patches_tensor = patches_raw.view(1, 3, 9, 9, L).permute(0, 4, 1, 2, 3).squeeze(0)
        
        all_preds = []
        for i in range(0, patches_tensor.shape[0], 4096):
            b_patches = patches_tensor[i:i+4096].to(self.device)
            preds = filter_model(b_patches)
            all_preds.append(preds)

        preds_tensor = torch.cat(all_preds, dim=0)
        presence_scores = torch.sigmoid(preds_tensor[:, 0:1])
        field_tensor = torch.cat([presence_scores, preds_tensor[:, 1:]], dim=1)
        
        field_map_ch = field_tensor.view(1, L, 4).permute(0, 2, 1).view(1, 4, out_h, out_w)
        field_resized = F.interpolate(field_map_ch, size=(H, W), mode='bilinear', align_corners=False)
        
        field_np = field_resized.permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()
        
        field_t = torch.from_numpy(field_np).float().permute(2, 0, 1)
        p, dx, dy, off = field_t[0:1], field_t[1:2], field_t[2:3], field_t[3:4]
        conf = torch.sqrt(dx**2 + dy**2)
        feat_map = torch.cat([p, dx, dy, off, conf], dim=0)
        return feat_map

    def _prepare_cached_fields(self, filter_model, dataset_loaders=None, samples_per_class=None, upsample_scale=3):
        cache_suffix = f"_full" if samples_per_class is None else f"_{samples_per_class}pc"
        dynamic_cache_path = self.field_cache_path.replace(".npz", f"{cache_suffix}.npz")

        if os.path.exists(dynamic_cache_path):
            try:
                print(f"📦 Loading pre-computed boundary fields from cache: [{dynamic_cache_path}]")
                data = np.load(dynamic_cache_path)
                if data["fields"].shape[1] == 5:
                    return data["fields"], data["labels"]
                else:
                    print("⚠️ Old cache format detected. Re-generating...")
            except Exception as e:
                print(f"⚠️ Field cache corrupted ({e}). Re-generating...")
            if os.path.exists(dynamic_cache_path):
                os.remove(dynamic_cache_path)

        print(f"🚀 Pre-computing 5-channel boundary fields for training samples (samples_per_class: {samples_per_class if samples_per_class else 'ALL'})...")
        train_images, train_labels = [], []

        if dataset_loaders is None:
            from ..dao import UbyteDatasetLoader
            dataset_loaders = [
                ("mnist", 0, UbyteDatasetLoader("mnist", config=self.config)),
                ("fashion_mnist", 10, UbyteDatasetLoader("fashion_mnist", config=self.config))
            ]

        for item in dataset_loaders:
            if len(item) == 3:
                dataset_name, offset, loader = item
            else:
                dataset_name, offset = item
                from ..dao import UbyteDatasetLoader
                loader = UbyteDatasetLoader(dataset_name, config=self.config)

            X, y = loader.load_train_data_by_classes(0, 9)
            if len(X) == 0:
                continue
            
            selected_indices = []
            for c in range(10):
                c_indices = np.where(y == c)[0]
                if len(c_indices) > 0:
                    if samples_per_class is None:
                        chosen = c_indices
                    else:
                        chosen = np.random.choice(c_indices, min(samples_per_class, len(c_indices)), replace=False)
                    selected_indices.append(chosen)
            
            if len(selected_indices) > 0:
                selected_indices = np.concatenate(selected_indices)
                train_images.append(X[selected_indices])
                train_labels.append(y[selected_indices] + offset)

        train_images = np.concatenate(train_images, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        cached_fields = []
        with torch.no_grad():
            for img_array in tqdm(train_images, desc="Caching 5ch Fields"):
                feat_map = self._image_to_field_tensor(img_array, filter_model, upsample_scale)
                cached_fields.append(feat_map.cpu().numpy())

        cached_fields_arr = np.array(cached_fields, dtype=np.float16) 
        print(f"💾 Saving float16 cached fields to [{dynamic_cache_path}]...")
        np.savez(dynamic_cache_path, fields=cached_fields_arr, labels=train_labels)
        print(f"✅ 5ch Boundary fields cached successfully.")

        return cached_fields_arr, train_labels

    def train(self, dataset_loaders=None, epochs=300, batch_size=64, lr=0.001, patience=20, samples_per_class=None):
        try:
            filter_model = self._load_filter_model()
        except Exception as e:
            print(e)
            return

        cached_fields, train_labels = self._prepare_cached_fields(filter_model, dataset_loaders=dataset_loaders, samples_per_class=samples_per_class, upsample_scale=3)
        full_dataset = BoundaryFieldDataset(cached_fields, train_labels)
        
        val_size = int(len(full_dataset) * 0.2)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

        model = BoundaryFieldCNN(num_classes=20).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            correct_count = 0
            total_count = 0

            pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train", leave=False)
            for b_fields, b_y in pbar:
                b_fields, b_y = b_fields.to(self.device), b_y.to(self.device)
                optimizer.zero_grad()
                
                outputs = model(b_fields)
                loss = criterion(outputs, b_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * b_fields.size(0)
                _, preds = torch.max(outputs, 1)
                correct_count += (preds == b_y).sum().item()
                total_count += b_y.size(0)

            train_loss = total_loss / len(train_dataset)
            train_acc = (correct_count / total_count) * 100

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Val  ", leave=False)
            with torch.no_grad():
                for b_fields, b_y in val_pbar:
                    b_fields, b_y = b_fields.to(self.device), b_y.to(self.device)
                    outputs = model(b_fields)
                    loss = criterion(outputs, b_y)
                    val_loss += loss.item() * b_fields.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == b_y).sum().item()
                    val_total += b_y.size(0)

            avg_val_loss = val_loss / len(val_dataset)
            val_acc = (val_correct / val_total) * 100

            scheduler.step(avg_val_loss)

            print(f"👉 Epoch [{epoch+1:3d}/{epochs}] Train Loss: {train_loss:.4f} (Acc: {train_acc:5.1f}%) | Val Loss: {avg_val_loss:.4f} (Acc: {val_acc:5.1f}%)")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), self.model_path)
                print(f"   💾 Saved Best Model (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"   ⏳ Early Stopping Counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch+1}!")
                    break

        print(f"\n✅ Training completed. Model saved to: [{self.model_path}]")

    def evaluate(self, dataset_loaders=None, upsample_scale=3):
        try:
            filter_model = self._load_filter_model()
        except Exception as e:
            print(e)
            return

        if not os.path.exists(self.model_path):
            print(f"⚠️ Trained classifier model not found at [{self.model_path}]. Please run training first (Option 2).")
            return

        model = BoundaryFieldCNN(num_classes=20).to(self.device)
        try:
            model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        except TypeError:
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()

        print(f"\n--- 📊 Evaluating Boundary Field CNN Classifier (Unified Pipeline) ---")

        if dataset_loaders is None:
            from ..dao import UbyteDatasetLoader
            dataset_loaders = [
                ("mnist", 0, "MNIST (Classes 0-9)", UbyteDatasetLoader("mnist", config=self.config)),
                ("fashion_mnist", 10, "Fashion-MNIST (Classes 10-19)", UbyteDatasetLoader("fashion_mnist", config=self.config))
            ]

        with torch.no_grad():
            for item in dataset_loaders:
                if len(item) == 4:
                    dataset_name, global_offset, description, loader = item
                else:
                    dataset_name, global_offset, description = item
                    from ..dao import UbyteDatasetLoader
                    loader = UbyteDatasetLoader(dataset_name, config=self.config)

                X_test, y_test = loader.load_test_data()
                
                class_correct = {c: 0 for c in range(10)}
                class_total = {c: 0 for c in range(10)}
                total_correct = 0
                total_samples = 0
                
                for class_id in range(10):
                    indices = np.where(y_test == class_id)[0]
                    if len(indices) == 0:
                        continue
                    sample_indices = np.random.choice(indices, min(50, len(indices)), replace=False)
                    
                    for idx in sample_indices:
                        img_array = X_test[idx]
                        feat_t = self._image_to_field_tensor(img_array, filter_model, upsample_scale).unsqueeze(0).to(self.device)
                        
                        outputs = model(feat_t)
                        _, pred_label = torch.max(outputs, 1)
                        
                        true_global_label = class_id + global_offset
                        if pred_label.item() == true_global_label:
                            class_correct[class_id] += 1
                            total_correct += 1
                        class_total[class_id] += 1
                        total_samples += 1

                print(f"\n📋 [{description}] Class-wise Accuracy Report:")
                for c_id in range(10):
                    c_corr = class_correct[c_id]
                    c_tot = class_total[c_id]
                    c_acc = (c_corr / c_tot) * 100 if c_tot > 0 else 0
                    print(f"  • Class {c_id + global_offset:2d} (Local {c_id}): {c_corr:2d} / {c_tot:2d} correct ({c_acc:5.1f}%)")
                
                overall_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0
                print(f"  👉 Overall [{dataset_name.upper()}] Accuracy: {total_correct}/{total_samples} ({overall_acc:.2f}%)\n")