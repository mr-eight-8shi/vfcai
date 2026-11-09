import os
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from ..dao import UbyteDatasetLoader
from .boundary_filter_model import UnifiedBoundaryRegressor


class BoundaryModelVisualizer:
    """Evaluates trained 9x9 context unified boundary models visually on sample images, generating clean 2-in-1 grids (Original & True Functional Reconstruction Map)."""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.model_dir = os.path.join(config.space_dir, "models")
        self.unified_model_path = os.path.join(self.model_dir, "boundary_unified_regressor.pth")
        self.presence_model_path = os.path.join(self.model_dir, "boundary_presence_classifier.pth")

    def _load_model(self):
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

    def _infer_vector_field(self, pil_img, model):
        W, H = pil_img.size
        tensor_img = torch.tensor(np.array(pil_img, dtype=np.float32) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        padded_img = F.pad(tensor_img, (4, 4, 4, 4), mode='replicate')
        patches_raw = F.unfold(padded_img, kernel_size=9, stride=3)
        L = patches_raw.shape[2]
        
        out_h = (H - 1) // 3 + 1
        out_w = (W - 1) // 3 + 1
        
        patches_tensor = patches_raw.view(1, 3, 9, 9, L).permute(0, 4, 1, 2, 3).squeeze(0)

        batch_size = 4096
        all_preds = []
        
        with torch.no_grad():
            for i in range(0, patches_tensor.shape[0], batch_size):
                b_patches = patches_tensor[i:i+batch_size].to(self.device)
                preds = model(b_patches)
                all_preds.append(preds)
        
        preds_tensor = torch.cat(all_preds, dim=0)
        
        presence_logits = preds_tensor[:, 0:1]
        presence_scores = torch.sigmoid(presence_logits)
        
        field_tensor = torch.cat([presence_scores, preds_tensor[:, 1:]], dim=1)
        field_map_ch = field_tensor.view(1, out_h, out_w, 4).permute(0, 3, 1, 2)
        field_resized = F.interpolate(field_map_ch, size=(H, W), mode='bilinear', align_corners=False)
        
        return field_resized.permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()

    def _generate_true_reconstructed_image(self, field_data, W, H):
        """Redraws the 3x3 region directly from all parameters (Presence, dx, dy, offset) using:
        val = dx * (px - 1) + dy * (py - 1) + offset
        boundary_value = tanh(val) * presence
        """
        out_h = (H - 1) // 3 + 1
        out_w = (W - 1) // 3 + 1
        
        recon_canvas = np.zeros((out_h * 3, out_w * 3), dtype=np.float32)

        for gy in range(out_h):
            for gx in range(out_w):
                y_center = min(gy * 3 + 1, H - 1)
                x_center = min(gx * 3 + 1, W - 1)
                
                pres = field_data[y_center, x_center, 0]      # Presence score
                dx = field_data[y_center, x_center, 1]        # Slope x
                dy = field_data[y_center, x_center, 2]        # Slope y
                offset = field_data[y_center, x_center, 3]    # Offset (SDF)
                
                for py in range(3):
                    for px in range(3):
                        val = dx * (px - 1) + dy * (py - 1) + offset
                        boundary_value = np.tanh(val) * pres
                        recon_canvas[gy * 3 + py, gx * 3 + px] = boundary_value

        recon_normalized = (recon_canvas + 1.0) / 2.0
        recon_pil = Image.fromarray((recon_normalized * 255).clip(0, 255).astype(np.uint8))
        recon_pil = recon_pil.resize((W, H), Image.BILINEAR)
        
        return np.array(recon_pil, dtype=np.float32) / 255.0

    def visualize_sample_images(self):
        sample_dir = os.path.join(self.config.dataset_dir, "sample_image")
        if not os.path.exists(sample_dir):
            print(f"⚠️ sample_image directory not found: [{sample_dir}]")
            return

        image_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print("⚠️ No image files found.")
            return

        try:
            model = self._load_model()
        except Exception as e:
            print(e)
            return

        output_dir = os.path.join(self.config.space_dir, "boundary_filtered_results")
        os.makedirs(output_dir, exist_ok=True)

        print("\n--- 🚀 9x9 Context Boundary & 2-in-1 True Functional Reconstruction Visualization (Samples) ---")
        for filename in image_files:
            t_start = time.time()
            img_path = os.path.join(sample_dir, filename)
            pil_img = Image.open(img_path).convert("RGB")
            W, H = pil_img.size

            field_data = self._infer_vector_field(pil_img, model)
            offset_map = field_data[:, :, 3]
            dx = field_data[:, :, 1]
            dy = field_data[:, :, 2]

            recon_img = self._generate_true_reconstructed_image(field_data, W, H)

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            axes[0].imshow(pil_img)
            axes[0].set_title(f"Original: {filename}")
            axes[0].axis('off')

            im = axes[1].imshow(recon_img, cmap='gray', vmin=0.0, vmax=1.0)
            axes[1].set_title("True Functional Reconstruction Map\n(tanh(dx*x + dy*y + offset) * presence)")
            axes[1].axis('off')
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Reconstructed Intensity')

            plt.tight_layout()
            out_path = os.path.join(output_dir, f"boundary_visualization_2in1_{filename}")
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()

            theta = np.arctan2(dy, dx)
            theta_norm = (theta + np.pi) / (2 * np.pi)
            theta_ids = np.clip(np.floor(theta_norm * 16), 0, 15).astype(int)

            offset_norm = np.clip((offset_map + 1.0) / 2.0, 0.0, 1.0)
            offset_ids = np.clip(np.floor(offset_norm * 8), 0, 7).astype(int)

            token_map = np.stack([theta_ids, offset_ids], axis=-1).tolist()

            json_data = {
                "width": W,
                "height": H,
                "format": "quantized_boundary_tokens",
                "encoding": {"theta_bins": 16, "offset_bins": 8},
                "token_map": token_map
            }

            json_path = os.path.join(output_dir, f"boundary_tokens_{os.path.splitext(filename)[0]}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, separators=(',', ':'))

            print(f"✅ Processed [{filename}] -> Time: {time.time() - t_start:.3f}s")

        print(f"\n📂 Saved to: [{output_dir}]\n")

    def visualize_ubyte_classes(self, upsample_scale=3):
        try:
            model = self._load_model()
        except Exception as e:
            print(e)
            return

        output_dir = os.path.join(self.config.space_dir, "boundary_filtered_ubyte_results")
        os.makedirs(output_dir, exist_ok=True)

        datasets = [
            ("mnist", "mnist"),
            ("fashion_mnist", "fashion")
        ]

        print(f"\n--- 🚀 MNIST & Fashion-MNIST: 3 Samples per Class Reconstruction (Visualizer) ---")
        with torch.no_grad():
            for dataset_name, prefix in datasets:
                loader = UbyteDatasetLoader(dataset_name, config=self.config)
                for class_id in range(10):
                    t_start = time.time()
                    X_cls, _ = loader.load_train_data_by_classes(class_id, class_id)
                    if len(X_cls) == 0:
                        continue
                    
                    sample_indices = np.random.choice(len(X_cls), min(3, len(X_cls)), replace=False)
                    
                    for sample_idx, idx in enumerate(sample_indices):
                        img_array = X_cls[idx]
                        img_uint8 = (img_array * 255).astype(np.uint8) if img_array.max() <= 1.0 else img_array.astype(np.uint8)
                        pil_img = Image.fromarray(img_uint8).convert("RGB")
                        
                        orig_w, orig_h = pil_img.size
                        upsampled_img = pil_img.resize((orig_w * upsample_scale, orig_h * upsample_scale), Image.BICUBIC)
                        W, H = upsampled_img.size

                        field_data = self._infer_vector_field(upsampled_img, model)
                        recon_img = self._generate_true_reconstructed_image(field_data, W, H)

                        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                        
                        axes[0].imshow(upsampled_img)
                        axes[0].set_title(f"Original (3x Upsampled)\n[{dataset_name.upper()} Class {class_id} - Sample {sample_idx+1}]")
                        axes[0].axis('off')

                        im = axes[1].imshow(recon_img, cmap='gray', vmin=0.0, vmax=1.0)
                        axes[1].set_title("True Functional Reconstruction Map\n(tanh(dx*x + dy*y + offset) * presence)")
                        axes[1].axis('off')
                        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Intensity')

                        plt.tight_layout()
                        filename = f"{prefix}_class_{class_id}_sample_{sample_idx+1}.png"
                        out_path = os.path.join(output_dir, filename)
                        plt.savefig(out_path, dpi=150, bbox_inches='tight')
                        plt.close()

                    print(f"✅ [{dataset_name.upper()} Class {class_id}] 3 samples visualized in {time.time() - t_start:.3f}s")

        print(f"\n📂 Saved to: [{output_dir}]\n")