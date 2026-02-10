import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

class TrainingMonitor:
    """
    Handles logging of training metrics to CSV and generating visualization plots.
    """
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_file = os.path.join(self.checkpoint_dir, "training_log.csv")
        
        # Define CSV headers
        self.headers = [
            "epoch", 
            "train_loss", "train_pos", "train_smooth", "train_ent", "tf_ratio",
            "val_loss", "val_pos", "val_smooth", "val_ent"
        ]
        
        # Initialize file if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_step(self, epoch, train_metrics, val_metrics):
        """
        Appends a single epoch's data to the log file.
        """
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics.get("loss_total", 0),
                train_metrics.get("loss_pos", 0),
                train_metrics.get("loss_smooth", 0),
                train_metrics.get("loss_ent", 0),
                train_metrics.get("teacher_forcing_ratio", 0),
                val_metrics.get("loss_total", 0),
                val_metrics.get("loss_pos", 0),
                val_metrics.get("loss_smooth", 0),
                val_metrics.get("loss_ent", 0)
            ])

    def plot_curves(self):
        """
        Reads the CSV log and generates a 2x2 dashboard of training curves.
        Saves to checkpoint_dir/training_log_plot.png
        """
        if not os.path.exists(self.log_file):
            print(f"[Monitor] No log file found at {self.log_file}")
            return

        try:
            df = pd.read_csv(self.log_file)
        except Exception as e:
            print(f"[Monitor] Error reading log file: {e}")
            return
        
        if len(df) < 2:
            print("[Monitor] Not enough data to plot yet.")
            return

        plt.figure(figsize=(14, 10))
        plt.style.use('ggplot')

        # 1. Total Loss Curve
        plt.subplot(2, 2, 1)
        plt.plot(df['epoch'], df['train_loss'], label='Train Total', color='tab:blue')
        plt.plot(df['epoch'], df['val_loss'], label='Val Total', color='tab:red')
        plt.title('Total Loss Convergence')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # 2. Position Loss (Physical Accuracy)
        plt.subplot(2, 2, 2)
        plt.plot(df['epoch'], df['train_pos'], label='Train MSE', color='tab:green')
        plt.plot(df['epoch'], df['val_pos'], label='Val MSE', color='tab:orange')
        plt.title('Position Loss (Reconstruction)')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(True)

        # 3. Regime Disentanglement (The Critical Plot)
        plt.subplot(2, 2, 3)
        plt.plot(df['epoch'], df['train_ent'], label='Train MI', color='purple')
        plt.plot(df['epoch'], df['val_ent'], label='Val MI', color='magenta')
        
        # Add Reference Lines for Interpretation
        plt.axhline(y=-1.0, color='green', linestyle='--', linewidth=1.5, label='Ideal Target (-1.0)')
        plt.axhline(y=0.0, color='black', linestyle='--', linewidth=1.5, label='Collapse (0.0)')
        
        plt.title('Latent Regime Loss (Mutual Info)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value (H_sample - H_batch)')
        plt.legend(loc='upper right')
        plt.grid(True)

        # 4. Teacher Forcing Schedule
        plt.subplot(2, 2, 4)
        plt.plot(df['epoch'], df['tf_ratio'], label='TF Ratio', color='gray', linestyle='-')
        plt.fill_between(df['epoch'], df['tf_ratio'], alpha=0.2, color='gray')
        plt.title('Teacher Forcing Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Ratio')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        save_path = os.path.join(self.checkpoint_dir, "training_log_plot.png")
        plt.savefig(save_path)
        print(f"[Monitor] Loss curves updated: {save_path}")
        plt.close()