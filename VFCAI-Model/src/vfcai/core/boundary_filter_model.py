import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt


class UnifiedBoundaryRegressor(nn.Module):
    """Predicts boundary presence score and parameters [dx, dy, offset] for the central 3x3 cells from a 9x9 input patch including surrounding context."""
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9 * 9 * 3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.presence_head = nn.Linear(64, 1)
        self.param_head = nn.Linear(64, 3)

    def forward(self, x):
        features = self.net(x)
        
        presence_logit = self.presence_head(features)
        
        out = self.param_head(features)
        v = F.normalize(out[:, :2], p=2, dim=1, eps=1e-8)
        offset = out[:, 2:3]
        
        return torch.cat([presence_logit, v, offset], dim=1)


class BoundaryModelTrainer:
    """Manages training and evaluation for the unified boundary regression model using 9x9 inputs."""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.model_dir = os.path.join(config.space_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "boundary_unified_regressor.pth")
        
        self.presence_model_path = os.path.join(self.model_dir, "boundary_presence_classifier.pth")
        self.regressor_model_path = os.path.join(self.model_dir, "boundary_parameter_regressor.pth")
        
        self.logger = config.logger

    def _generate_training_data(self, num_samples):
        X_data, presence_data, param_data = [], [], []
        
        for _ in range(num_samples):
            grid = np.zeros((9, 9, 3), dtype=np.float32)
            bg_color = np.random.uniform(0.0, 0.3, size=(3,))
            fg_color = np.random.uniform(0.7, 1.0, size=(3,))
            
            if np.random.rand() < 0.15:
                has_line = 0.0
                dx, dy, offset = 0.0, 0.0, 0.0
                for y in range(9):
                    for x in range(9):
                        grid[y, x, :] = bg_color
            else:
                has_line = 1.0
                theta = np.random.uniform(0, 2 * np.pi)
                nx, ny = np.cos(theta), np.sin(theta)
                norm = np.sqrt(nx**2 + ny**2) + 1e-8
                dx, dy = nx / norm, ny / norm
                offset = np.random.uniform(-1.0, 1.0)

                for y in range(9):
                    for x in range(9):
                        val = dx * (x - 4.0) + dy * (y - 4.0) + offset
                        grid[y, x, :] = fg_color if (val >= 0) else bg_color

            noise = np.random.normal(0, 0.02, size=(9, 9, 3))
            grid = np.clip(grid + noise, 0.0, 1.0)

            grid_transposed = np.transpose(grid, (2, 0, 1))

            X_data.append(grid_transposed)
            presence_data.append([has_line])
            param_data.append([dx, dy, offset])

        return (
            np.array(X_data, dtype=np.float32),
            np.array(presence_data, dtype=np.float32),
            np.array(param_data, dtype=np.float32)
        )

    def train_unified_model(self, epochs=30, batch_size=64, num_samples=20000):
        self.logger.info("Training 9x9 context unified boundary regressor...")
        X_data, presence_data, param_data = self._generate_training_data(num_samples)

        targets = np.concatenate([presence_data, param_data], axis=1)

        dataset = TensorDataset(torch.tensor(X_data), torch.tensor(targets))
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = UnifiedBoundaryRegressor().to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        bce_criterion = nn.BCEWithLogitsLoss()

        best_loss = float('inf')
        print(f"\n--- 9x9 Context Boundary Model Training Started ---")
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for b_x, b_y in train_loader:
                b_x, b_y = b_x.to(self.device), b_y.to(self.device)
                optimizer.zero_grad()
                
                preds = model(b_x)
                
                presence_loss = bce_criterion(preds[:, 0:1], b_y[:, 0:1])
                
                has_line_mask = (b_y[:, 0] == 1.0)
                if has_line_mask.any():
                    p_valid = preds[has_line_mask]
                    y_valid = b_y[has_line_mask]
                    
                    pred_dir = p_valid[:, 1:3]
                    true_dir = y_valid[:, 1:3]
                    cos_loss = 1.0 - torch.sum(pred_dir * true_dir, dim=1)
                    
                    offset_loss = F.mse_loss(p_valid[:, 3], y_valid[:, 3], reduction='none')
                    param_loss = (cos_loss + 2.0 * offset_loss).mean()
                else:
                    param_loss = torch.tensor(0.0, device=self.device)

                loss = presence_loss + param_loss
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * b_x.size(0)

            scheduler.step()
            avg_train_loss = train_loss / len(train_dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for b_x, b_y in val_loader:
                    b_x, b_y = b_x.to(self.device), b_y.to(self.device)
                    p = model(b_x)
                    
                    p_loss = bce_criterion(p[:, 0:1], b_y[:, 0:1])
                    mask = (b_y[:, 0] == 1.0)
                    if mask.any():
                        pv = p[mask]
                        yv = b_y[mask]
                        c_loss = 1.0 - torch.sum(pv[:, 1:3] * yv[:, 1:3], dim=1)
                        o_loss = F.mse_loss(pv[:, 3], yv[:, 3], reduction='none')
                        pa_loss = (c_loss + 2.0 * o_loss).mean()
                    else:
                        pa_loss = torch.tensor(0.0, device=self.device)
                        
                    val_loss += (p_loss + pa_loss).item() * b_x.size(0)

            avg_val_loss = val_loss / len(val_dataset)
            print(f"👉 Epoch [{epoch+1}/{epochs}] Train Loss={avg_train_loss:.5f} | Val Loss={avg_val_loss:.5f}")

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), self.model_path)
                torch.save(model.state_dict(), self.presence_model_path)
                torch.save(model.state_dict(), self.regressor_model_path)

        print(f"💾 9x9 Context Unified model saved to: [{self.model_path}]")

    def train_boundary_model(self, epochs=30, batch_size=64, num_samples=20000, patience=5):
        self.train_unified_model(epochs=epochs, batch_size=batch_size, num_samples=num_samples)

    def evaluate_boundary_model(self):
        if not os.path.exists(self.model_path) and not os.path.exists(self.presence_model_path):
            print("⚠️ Trained model not found. Please run training first.")
            return

        print("📊 Evaluating 9x9 context unified model...")
        X_test, presence_test, param_test = self._generate_training_data(200)

        model = UnifiedBoundaryRegressor().to(self.device)
        load_path = self.model_path if os.path.exists(self.model_path) else self.presence_model_path
        model.load_state_dict(torch.load(load_path, map_location=self.device))
        model.eval()

        correct_presence = 0
        correct_params = 0
        line_count = 0

        with torch.no_grad():
            for i in range(len(X_test)):
                x_sample = torch.tensor(X_test[i:i+1], dtype=torch.float32).to(self.device)
                preds = model(x_sample).cpu().numpy().flatten()
                
                presence_logit = preds[0]
                pred_presence = 1.0 if (1.0 / (1.0 + np.exp(-presence_logit))) >= 0.3 else 0.0
                true_presence = presence_test[i][0]

                if pred_presence == true_presence:
                    correct_presence += 1

                if true_presence == 1.0:
                    line_count += 1
                    true_p = param_test[i]
                    pred_p = preds[1:]
                    cos_sim = np.dot(true_p[:2], pred_p[:2]) / (np.linalg.norm(true_p[:2]) * np.linalg.norm(pred_p[:2]) + 1e-8)
                    if cos_sim >= 0.80 and abs(pred_p[2] - true_p[2]) <= 0.4:
                        correct_params += 1

        print(f"  🎯 Presence Accuracy: {(correct_presence/len(X_test))*100:.1f}%")
        if line_count > 0:
            print(f"  🎯 Parameter Accuracy (on lines): {(correct_params/line_count)*100:.1f}%")

    @staticmethod
    def save_boundary_sample_grid_image(space_dir, num_samples=100):
        pass