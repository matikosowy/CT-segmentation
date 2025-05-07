from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, jaccard_score, precision_score


def evaluate_model(model, test_loader, device, output_dir="evaluation_results"):
    """
    Evaluate the model on the test set and save visualization results.

    Args:
        model: Trained model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        output_dir: Directory to save visualization results

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print("=" * 30)
    print("EVALUATION")
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dice_scores = []
    jaccard_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            inputs = batch["image"].to(device)
            labels = batch["mask"].to(device)

            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            preds_binary = (preds > 0.5).float()

            inputs_np = inputs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            preds_np = preds_binary.cpu().numpy()

            for j in range(inputs.shape[0]):
                pred = preds_np[j].squeeze()
                label = labels_np[j].squeeze()

                if label.sum() > 0:
                    dice = 2 * (pred * label).sum() / (pred.sum() + label.sum() + 1e-8)
                    dice_scores.append(dice)

                    jac = jaccard_score(label.flatten(), pred.flatten())
                    jaccard_scores.append(jac)

                    precision = precision_score(label.flatten(), pred.flatten(), zero_division=1)
                    recall = recall_score(label.flatten(), pred.flatten(), zero_division=1)
                    f1 = f1_score(label.flatten(), pred.flatten(), zero_division=1)

                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1)

            # Save visualization for a few samples
            if i % 10 == 0:
                for b in range(min(2, inputs.shape[0])):
                    input_img = inputs_np[j, 0]  # [C, H, W] -> [H, W]
                    label_img = labels_np[j, 0]
                    pred_img = preds_np[j, 0]

                    plt.figure(figsize=(15, 5))

                    plt.subplot(1, 3, 1)
                    plt.imshow(input_img, cmap="gray")
                    plt.title("Input CT")
                    plt.axis("off")

                    plt.subplot(1, 3, 2)
                    plt.imshow(input_img, cmap="gray")
                    plt.imshow(label_img, cmap="jet", alpha=0.3)
                    plt.title("Ground Truth")
                    plt.axis("off")

                    plt.subplot(1, 3, 3)
                    plt.imshow(input_img, cmap="gray")
                    plt.imshow(pred_img, cmap="jet", alpha=0.3)
                    plt.title("Prediction")
                    plt.axis("off")

                    patient_id = batch["patient_id"][j]
                    slice_idx = batch["slice_idx"][j].item()

                    out_file = f"{output_dir}/patient_{patient_id}_slice_{slice_idx}.png"
                    plt.savefig(out_file)
                    plt.close()

    avg_metrics = {
        "dice_score": np.mean(dice_scores) if dice_scores else 0.0,
        "jaccard_index": np.mean(jaccard_scores) if jaccard_scores else 0.0,
        "precision": np.mean(precision_scores) if precision_scores else 0.0,
        "recall": np.mean(recall_scores) if recall_scores else 0.0,
        "f1_score": np.mean(f1_scores) if f1_scores else 0.0,
        "samples_evaluated": len(dice_scores),
    }

    print("\n===== Evaluation Metrics =====")
    print(f"Dice Score: {avg_metrics['dice_score']: .4f}")
    print(f"Jaccard Index (IoU): {avg_metrics['jaccard_index']: .4f}")
    print(f"Precision: {avg_metrics['precision']: .4f}")
    print(f"Recall: {avg_metrics['recall']: .4f}")
    print(f"F1 Score: {avg_metrics['f1_score']: .4f}")
    print(f"Samples evaluated: {avg_metrics['samples_evaluated']}")
    print("=============================\n")

    return avg_metrics
