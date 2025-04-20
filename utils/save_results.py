import torch
import json
import os

def save_training_results(
    history,
    final_metrics,
    model_path="models/model.pth",
    history_path="reports/training_history.json",
    report_path="reports/final_metrics.txt"
):
    # Create folders if they don't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # 1. Save model weights
    torch.save(final_metrics["model_state_dict"], model_path)

    # 2. Save training history
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    # 3. Save final metrics to text file
    with open(report_path, "w") as f:
        f.write("Final Validation Metrics:\n")
        f.write(f"Accuracy: {final_metrics['val_acc']:.4f}\n")
        f.write(f"F1 Score: {final_metrics['val_f1']:.4f}\n")
