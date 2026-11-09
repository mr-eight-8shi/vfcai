import os
import sys
import numpy as np


class UbyteDatasetLoader:
    """A generic data loader for IDX/Ubyte binary formats such as MNIST and Fashion-MNIST."""

    def __init__(self, dataset_name: str, config=None):
        self.config = config
        base_dir = getattr(config, "dataset_dir", "dataset") if config else "dataset"
        
        self.data_dir = os.path.join(base_dir, dataset_name)
        self.dataset_name_display = dataset_name.replace("_", " ").title()

        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None

    def _find_file(self, possible_filenames):
        for fname in possible_filenames:
            path = os.path.join(self.data_dir, fname)
            if os.path.exists(path):
                return path
        return None

    def _ensure_train_loaded(self):
        if self._X_train is None or self._y_train is None:
            img_candidates = ["train-images-idx3-ubyte", "train-images.idx3-ubyte"]
            label_candidates = ["train-labels-idx1-ubyte", "train-labels.idx1-ubyte"]

            img_path = self._find_file(img_candidates)
            label_path = self._find_file(label_candidates)

            self._X_train, self._y_train = self._load_ubyte(
                img_path, label_path, f"{self.dataset_name_display} Training"
            )

    def _ensure_test_loaded(self):
        if self._X_test is None or self._y_test is None:
            img_candidates = ["t10k-images-idx3-ubyte", "t10k-images.idx3-ubyte"]
            label_candidates = ["t10k-labels-idx1-ubyte", "t10k-labels.idx1-ubyte"]

            img_path = self._find_file(img_candidates)
            label_path = self._find_file(label_candidates)

            self._X_test, self._y_test = self._load_ubyte(
                img_path, label_path, f"{self.dataset_name_display} Test"
            )

    def load_train_data_by_classes(self, class_min: int, class_max: int):
        self._ensure_train_loaded()
        mask = (self._y_train >= class_min) & (self._y_train <= class_max)
        return self._X_train[mask], self._y_train[mask]

    def load_test_data_by_classes(self, class_min: int, class_max: int):
        self._ensure_test_loaded()
        mask = (self._y_test >= class_min) & (self._y_test <= class_max)
        return self._X_test[mask], self._y_test[mask]

    def load_test_data(self):
        self._ensure_test_loaded()
        return self._X_test, self._y_test

    def _load_ubyte(self, img_path, label_path, dataset_type_name):
        if not img_path or not label_path or not os.path.exists(img_path) or not os.path.exists(label_path):
            print(f"[ERROR] Required binary files not found in directory: {self.data_dir}")
            print(f"  - Searched Image Path: {img_path}")
            print(f"  - Searched Label Path: {label_path}")
            print("Please ensure ubyte files are correctly placed.")
            sys.exit(1)

        print(f"Loading {dataset_type_name} dataset from [{self.data_dir}]...")
        images = self._load_images(img_path)
        labels = self._load_labels(label_path)

        print(f"  ✓ Loaded successfully (Total samples: {len(images)}, Shape: {images.shape[1]}x{images.shape[2]})")
        return images, labels

    def _load_images(self, file_path):
        with open(file_path, "rb") as f:
            _ = f.read(4)
            num_images = int.from_bytes(f.read(4), "big")
            rows = int.from_bytes(f.read(4), "big")
            cols = int.from_bytes(f.read(4), "big")
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(num_images, rows, cols).astype(np.float32) / 255.0

    def _load_labels(self, file_path):
        with open(file_path, "rb") as f:
            _ = f.read(4)
            _ = f.read(4)
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.astype(np.int64)