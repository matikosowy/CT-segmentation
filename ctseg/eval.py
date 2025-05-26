from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, jaccard_score, precision_score

from ctseg.data import create_2d_segmentation_dataloaders
from ctseg.models import create_unet_2d_model, create_segresnet_2d_model


def evaluate_model(
    model,
    test_loader,
    device,
    output_dir="evaluation_results",
    organ_names=None,
    mode="2d",
    height=None,
):
    """
    Evaluate the model on the test set, print metrics and save visualization results.

    Args:
        model: The trained segmentation model
        test_loader: DataLoader with test data
        device: Device to run inference on
        output_dir: Directory to save results
        organ_names: List of organ names corresponding to mask channels
        mode: "2d" or "3d" - determines how the model processes the data
        height: Height of the input images (for 3D mode, not used in 2D)
    """
    print("=" * 30)
    print("EVALUATION - please wait... (0% for a while is normal)")
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    organ_metrics = {}
    num_organs = None

    # Get organs' names
    if organ_names is None and hasattr(test_loader.dataset, "target_organs"):
        organ_names = test_loader.dataset.target_organs

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            inputs = batch["image"].to(device)
            labels = batch["mask"].to(device)

            if num_organs is None:
                if labels.dim() == 4:  # [B, C, H, W]
                    num_organs = labels.size(1)
                else:  # [B, H, W]
                    num_organs = 1

                for o in range(num_organs):
                    organ_metrics[o] = {
                        "dice_scores": [],
                        "jaccard_scores": [],
                        "precision_scores": [],
                        "recall_scores": [],
                        "f1_scores": [],
                    }

            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            preds_binary = (preds > 0.5).float()

            inputs_np = inputs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            preds_np = preds_binary.cpu().numpy()

            # Evaluate each sample in the batch
            for j in range(inputs.shape[0]):
                # For each organ channel
                for o in range(num_organs):
                    if labels.dim() == 4:  # [B, C, H, W]
                        pred = preds_np[j, o]
                        label = labels_np[j, o]
                    else:  # Single organ case
                        pred = preds_np[j, 0]
                        label = labels_np[j, 0]

                    if label.sum() > 0:
                        dice = 2 * (pred * label).sum() / (pred.sum() + label.sum() + 1e-8)
                        organ_metrics[o]["dice_scores"].append(dice)

                        jac = jaccard_score(label.flatten(), pred.flatten())
                        organ_metrics[o]["jaccard_scores"].append(jac)

                        precision = precision_score(label.flatten(), pred.flatten(), zero_division=1)
                        recall = recall_score(label.flatten(), pred.flatten(), zero_division=1)
                        f1 = f1_score(label.flatten(), pred.flatten(), zero_division=1)

                        organ_metrics[o]["precision_scores"].append(precision)
                        organ_metrics[o]["recall_scores"].append(recall)
                        organ_metrics[o]["f1_scores"].append(f1)

            # Visualize a few samples
            if i % 10 == 0:
                for j in range(min(2, inputs.shape[0])):
                    input_img = inputs_np[j, 0]  # CT image

                    # Ground truth and prediction
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                    for idx, (ax_title, data_source) in enumerate(
                        [
                            ("Ground Truth", labels_np),
                            ("Prediction", preds_np),
                        ]
                    ):
                        axes[idx].imshow(input_img, cmap="gray")
                        colors = plt.cm.tab10(np.linspace(0, 1, num_organs))

                        proxies = []
                        organ_labels = []

                        for o in range(num_organs):
                            if labels.dim() == 4:  # [B, C, H, W]
                                mask = data_source[j, o]
                            else:  # [B, H, W]
                                mask = data_source[j, 0]

                            organ_name = organ_names[o] if organ_names and o < len(organ_names) else f"organ_{o}"

                            colored_mask = np.zeros((*mask.shape, 4))
                            colored_mask[mask > 0.5, :] = colors[o % len(colors)]
                            colored_mask[mask > 0.5, 3] = 0.5

                            axes[idx].imshow(colored_mask, interpolation="none")

                            if mask.sum() > 0:
                                proxy = plt.Rectangle((0, 0), 1, 1, fc=colors[o % len(colors)])
                                proxies.append(proxy)
                                organ_labels.append(organ_name)

                        if proxies:
                            axes[idx].legend(proxies, organ_labels, loc="upper right")

                        axes[idx].set_title(ax_title)
                        axes[idx].axis("off")
                        axes[idx].set_aspect("equal")

                    patient_id = batch["patient_id"][j] if "patient_id" in batch else f"sample_{i}_{j}"
                    try:
                        slice_idx = batch["slice_idx"][j].item() if "slice_idx" in batch else f"slice_{i}_{j}"
                    except (AttributeError, TypeError):
                        slice_idx = f"slice_{i}_{j}"

                    out_file = output_dir / f"patient_{patient_id}_slice_{slice_idx}_combined.png"
                    plt.savefig(out_file)
                    plt.close()

    # Calculate average metrics for each organ and overall
    avg_metrics = {}
    overall_dice = []
    overall_jaccard = []
    overall_precision = []
    overall_recall = []
    overall_f1 = []
    samples_evaluated = 0

    # Print metrics for each organ and collect overall statistics
    print("\n===== Evaluation Metrics =====")
    for o in range(num_organs):
        organ_name = organ_names[o] if organ_names and o < len(organ_names) else f"organ_{o}"

        organ_dice = np.mean(organ_metrics[o]["dice_scores"]) if organ_metrics[o]["dice_scores"] else 0.0
        organ_jaccard = np.mean(organ_metrics[o]["jaccard_scores"]) if organ_metrics[o]["jaccard_scores"] else 0.0

        organ_precision = np.mean(organ_metrics[o]["precision_scores"]) if organ_metrics[o]["precision_scores"] else 0.0
        organ_recall = np.mean(organ_metrics[o]["recall_scores"]) if organ_metrics[o]["recall_scores"] else 0.0
        organ_f1 = np.mean(organ_metrics[o]["f1_scores"]) if organ_metrics[o]["f1_scores"] else 0.0
        organ_samples = len(organ_metrics[o]["dice_scores"])

        print(f"{organ_name}: ")
        print(f"  Dice Score: {organ_dice: .4f}")
        print(f"  Jaccard Index (IoU): {organ_jaccard: .4f}")
        print(f"  Precision: {organ_precision: .4f}")
        print(f"  Recall: {organ_recall: .4f}")
        print(f"  F1 Score: {organ_f1: .4f}")
        print(f"  Samples evaluated: {organ_samples}")

        overall_dice.extend(organ_metrics[o]["dice_scores"])
        overall_jaccard.extend(organ_metrics[o]["jaccard_scores"])
        overall_precision.extend(organ_metrics[o]["precision_scores"])
        overall_recall.extend(organ_metrics[o]["recall_scores"])
        overall_f1.extend(organ_metrics[o]["f1_scores"])
        samples_evaluated += organ_samples

    # Calculate overall metrics
    avg_metrics = {
        "dice_score": np.mean(overall_dice) if overall_dice else 0.0,
        "jaccard_index": np.mean(overall_jaccard) if overall_jaccard else 0.0,
        "precision": np.mean(overall_precision) if overall_precision else 0.0,
        "recall": np.mean(overall_recall) if overall_recall else 0.0,
        "f1_score": np.mean(overall_f1) if overall_f1 else 0.0,
        "samples_evaluated": len(overall_dice),
        "per_organ_metrics": organ_metrics,
    }

    print("\nOverall:")
    print(f"  Dice Score: {avg_metrics['dice_score']: .4f}")
    print(f"  Jaccard Index (IoU): {avg_metrics['jaccard_index']: .4f}")
    print(f"  Precision: {avg_metrics['precision']: .4f}")
    print(f"  Recall: {avg_metrics['recall']: .4f}")
    print(f"  F1 Score: {avg_metrics['f1_score']: .4f}")
    print(f"  Samples evaluated: {avg_metrics['samples_evaluated']}")
    print("=============================\n")

    # Save metrics to file
    metrics_file = output_dir / "evaluation_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write("===== Evaluation Metrics =====\n")
        for o in range(num_organs):
            organ_name = organ_names[o] if organ_names and o < len(organ_names) else f"organ_{o}"

            organ_dice = np.mean(organ_metrics[o]["dice_scores"] or [0.0])
            organ_jaccard = np.mean(organ_metrics[o]["jaccard_scores"] or [0.0])
            organ_precision = np.mean(organ_metrics[o]["precision_scores"] or [0.0])
            organ_recall = np.mean(organ_metrics[o]["recall_scores"] or [0.0])
            organ_f1 = np.mean(organ_metrics[o]["f1_scores"] or [0.0])

            f.write(f"\n{organ_name}: \n")
            f.write(f"  Dice Score: {organ_dice: .4f}\n")
            f.write(f"  Jaccard Index (IoU): {organ_jaccard: .4f}\n")
            f.write(f"  Precision: {organ_precision: .4f}\n")
            f.write(f"  Recall: {organ_recall: .4f}\n")
            f.write(f"  F1 Score: {organ_f1: .4f}\n")

        f.write("\nOverall:\n")
        f.write(f"Dice Score: {avg_metrics['dice_score']: .4f}\n")
        f.write(f"Jaccard Index (IoU): {avg_metrics['jaccard_index']: .4f}\n")
        f.write(f"Precision: {avg_metrics['precision']: .4f}\n")
        f.write(f"Recall: {avg_metrics['recall']: .4f}\n")
        f.write(f"F1 Score: {avg_metrics['f1_score']: .4f}\n")
        f.write(f"Samples evaluated: {avg_metrics['samples_evaluated']}\n")

    return avg_metrics


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


def visualize_samples(dataloader, num_samples=5):
    """Visualize random samples from the dataset with both masks."""
    plt.figure(figsize=(15, num_samples * 4))

    samples_seen = 0
    for batch in dataloader:
        images = batch["image"].numpy()
        masks = batch["mask"].numpy()

        for i in range(len(images)):
            if samples_seen >= num_samples:
                break

            img = images[i, 0]  # Get the first channel
            mask_left = masks[i, 0]  # First channel - left kidney
            mask_right = masks[i, 1]  # Second channel - right kidney

            # Skip if no kidneys in this slice
            if mask_left.sum() < 10 and mask_right.sum() < 10:
                continue

            # Plot the sample
            plt.subplot(num_samples, 3, samples_seen * 3 + 1)
            plt.imshow(img, cmap="gray")
            plt.title("CT Image")
            plt.axis("off")

            plt.subplot(num_samples, 3, samples_seen * 3 + 2)
            plt.imshow(img, cmap="gray")
            plt.imshow(mask_left, alpha=0.5, cmap="Reds")
            plt.title("Left Kidney")
            plt.axis("off")

            plt.subplot(num_samples, 3, samples_seen * 3 + 3)
            plt.imshow(img, cmap="gray")
            plt.imshow(mask_right, alpha=0.5, cmap="Blues")
            plt.title("Right Kidney")
            plt.axis("off")

            samples_seen += 1

        if samples_seen >= num_samples:
            break

    plt.tight_layout()
    plt.savefig("kidney_samples_visualization.png")
    plt.close()
    print("Visualization saved as 'kidney_samples_visualization.png'")


def inference(args, device):
    """Run inference on the test set using a trained model.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        device (torch.device): Device to run the model on.
    """

    print("RUNNING INFERENCE ONLY...")
    assert args.checkpoint is not None, "Checkpoint path must be provided for inference!"

    if args.model == "segresnet":
        model = create_segresnet_2d_model(
            in_channels=1,
            out_channels=len(args.target_organs),
            device=device,
        )
    elif args.model == "unet":
        model = create_unet_2d_model(
            in_channels=1,
            out_channels=len(args.target_organs),
            device=device,
        )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loader = create_2d_segmentation_dataloaders(
        root_dir=args.dataset,
        batch_size=args.batch_size,
        num_patients=args.num_patients,
        min_organ_pixels=args.min_organ_pixels,
        target_organs=args.target_organs,
        split="test",
        reset_cache=args.reset_cache,
    )

    # Save predictions in the same directory as the checkpoint
    checkpoint_path = Path(args.checkpoint)
    output_dir = checkpoint_path.parent / "eval"

    evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        organ_names=args.target_organs,
    )
