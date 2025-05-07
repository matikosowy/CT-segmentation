from pathlib import Path

import matplotlib.pyplot as plt


def plot_history(history, run_dir):
    """
    Plot training and validation loss and metrics over epochs.

    Args:
        history (dict): Dictionary containing training history with keys:
            - 'train_loss': List of training losses per epoch.
            - 'val_loss': List of validation losses per epoch.
            - 'dice_score': List of Dice scores per epoch.
            - 'jaccard_score': List of Jaccard scores per epoch.
        run_dir (str, Path): Directory to save the plot.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_losses = history["train_loss"]
    val_losses = history["val_loss"]
    dice_scores = history["dice_score"]
    jaccard_scores = history["jaccard_score"]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss Progression")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(dice_scores, label="Dice Score")
    plt.plot(jaccard_scores, label="Jaccard Index")
    plt.title("Validation Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()

    plt.tight_layout()

    fig_path = run_dir / "train_plot.png"
    plt.savefig(fig_path)
    print(f"Training history plot saved at: {run_dir / 'train_plot.png'}")
