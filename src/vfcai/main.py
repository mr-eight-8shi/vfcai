import argparse
import sys
import shutil
import os

from .com import AppConfig
from .core import (
    BoundaryModelTrainer,
    BoundaryModelVisualizer,
    BoundaryFilteredImageClassifier
)


def action_runner():
    """Decorator handling configuration loading and exception catching."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            config = AppConfig()
            print("\n" + "=" * 65)
            print("  ⚙️ [AppConfig] 3D Boundary Recognition System Initialized")
            print("=" * 65)
            print(f" • Workspace Directory : {config.space_dir}")
            print(f" • Log Directory        : {config.log_dir}")
            print("=" * 65 + "\n")
            try:
                return func(config, *args, **kwargs)
            except KeyboardInterrupt:
                print("\n\n[ABORT] Interrupted by user.\n")
        return wrapper
    return decorator


@action_runner()
def run_train_and_evaluate(config):
    """[Option 1] Train and evaluate boundary filters (presence and parameter models)."""
    trainer = BoundaryModelTrainer(config)
    print("--- [Step 1/2] Training Presence Classifier & Parameter Regressor ---")
    trainer.train_boundary_model(epochs=30, num_samples=20000, patience=5)
    
    print("\n--- [Step 2/2] Evaluating Boundary Models ---")
    trainer.evaluate_boundary_model()


@action_runner()
def run_classifier_train(config, samples_per_class=None):
    """[Option 2] Train Boundary Field CNN Classifier with configurable sample count per class."""
    classifier = BoundaryFilteredImageClassifier(config)
    
    print("\n--- 🚀 [Auto-Visualization] Generating 3 Samples per Class Reconstruction Maps ---")
    visualizer = BoundaryModelVisualizer(config)
    visualizer.visualize_ubyte_classes(upsample_scale=3)

    classifier.train(epochs=300, batch_size=64, lr=0.001, patience=20, samples_per_class=samples_per_class)


@action_runner()
def run_classifier_evaluate(config):
    """[Option 3] Evaluate saved Boundary Field CNN Classifier (Only)."""
    classifier = BoundaryFilteredImageClassifier(config)
    classifier.evaluate()


@action_runner()
def run_visualizer_pipeline(config):
    """[Option 4] Visualize trained boundary models on sample images (2-in-1 Grids with True Functional Reconstruction)."""
    visualizer = BoundaryModelVisualizer(config)
    visualizer.visualize_sample_images()


@action_runner()
def run_init(config):
    """[Option 9] Initialize and clear the workspace directory."""
    if os.path.exists(config.space_dir):
        shutil.rmtree(config.space_dir)
        print(f"🧹 Workspace '{config.space_dir}' has been initialized.")


def interactive_menu():
    """Interactive CLI menu."""
    while True:
        print("\n" + "═" * 65)
        print("  🚀 3D Boundary & Vector Field Analysis System")
        print("═" * 65)
        print(" [1] Train & Evaluate Boundary Filters (Presence & Regressor)")
        print(" [2] Train Boundary Field CNN Classifier & Visualize Samples (Configurable / Full)")
        print(" [3] Evaluate Boundary Field CNN Classifier (Only)")
        print(" [4] Visualize Boundary Models & True Functional Reconstruction (2-in-1 Grids)")
        print(" [9] Initialize Workspace (Reset)")
        print(" [q] Quit")
        print("═" * 65)

        try:
            choice = input("Enter choice: ").strip().lower()
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if choice in ["q", "quit"]:
            print("Exiting.")
            break
        elif choice == "1":
            run_train_and_evaluate()
        elif choice == "2":
            spc_input = input("Enter samples per class (Leave blank or type 'all' for full dataset): ").strip().lower()
            if spc_input in ["", "all", "none"]:
                samples_per_class = None
                print("➡️ Using FULL dataset for training.")
            else:
                try:
                    samples_per_class = int(spc_input)
                    print(f"➡️ Using {samples_per_class} samples per class.")
                except ValueError:
                    print("⚠️ Invalid input. Defaulting to FULL dataset.")
                    samples_per_class = None
            run_classifier_train(samples_per_class=samples_per_class)
        elif choice == "3":
            run_classifier_evaluate()
        elif choice == "4":
            run_visualizer_pipeline()
        elif choice in ["9", "99"]:
            confirm = input("⚠️ Initialize workspace? (y/n): ").strip().lower()
            if confirm == "y":
                run_init()
        else:
            print("[ERROR] Invalid choice.")


def cli():
    """CLI entry point for vfcai command."""
    parser = argparse.ArgumentParser(description="High-Precision Boundary Recognition CLI")
    parser.add_argument("action", nargs="?", help="Action to execute (train, classify, evaluate, visual, init)")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples per class for training (default: None for full dataset)")
    args = parser.parse_args()

    action = args.action.lower() if args.action else ""
    
    if action in ["train", "1"]:
        run_train_and_evaluate()
    elif action in ["classify", "train_clf", "2"]:
        run_classifier_train(samples_per_class=args.samples)
    elif action in ["evaluate", "eval_clf", "3"]:
        run_classifier_evaluate()
    elif action in ["visual", "4"]:
        run_visualizer_pipeline()
    elif action in ["init", "9", "99"]:
        run_init()
    else:
        interactive_menu()


if __name__ == "__main__":
    cli()