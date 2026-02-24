import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

class UnifiedTrainingMonitor:
    """
    Handles logging and dynamic plotting for both the CVAE and Baseline models.
    """
    def __init__(self, checkpoint_dir, model_name="model"):
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_file = os.path.join(self.checkpoint_dir, f"{model_name}_training_log.csv")
        
        # Unified headers covering both models
        self.headers = [
            "epoch", 
            "train_loss", "train_pos", "train_turn", "train_ent", "train_kl", "tf_ratio",
            "val_loss", "val_pos", "val_turn", "val_ent", "val_kl"
        ]
        
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_step(self, epoch, train_metrics, val_metrics, tf_ratio=0.0):
        """
        Appends metrics. Missing metrics (e.g., KL for baseline) default to 0.0.
        """
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics.get("loss_total", 0.0),
                train_metrics.get("loss_pos", 0.0),
                train_metrics.get("loss_turn", 0.0),
                train_metrics.get("loss_ent", 0.0),
                train_metrics.get("loss_kl", 0.0),
                tf_ratio,
                val_metrics.get("loss_total", 0.0),
                val_metrics.get("loss_pos", 0.0),
                val_metrics.get("loss_turn", 0.0),
                val_metrics.get("loss_ent", 0.0),
                val_metrics.get("loss_kl", 0.0)
            ])

    def plot_curves(self):
        """
        Generates a 2x2 dashboard adapting to the model type.
        """
        if not os.path.exists(self.log_file):
            return

        try:
            df = pd.read_csv(self.log_file)
        except Exception as e:
            print(f"[Monitor] Error reading log file: {e}")
            return
        
        if len(df) < 2:
            return

        plt.figure(figsize=(14, 10))
        plt.style.use('ggplot')

        # 1. Total Loss Curve
        plt.subplot(2, 2, 1)
        plt.plot(df['epoch'], df['train_loss'], label='Train Total', color='tab:blue')
        plt.plot(df['epoch'], df['val_loss'], label='Val Total', color='tab:red')
        plt.title(f'Total Loss Convergence ({self.model_name})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # 2. Position Loss (Reconstruction)
        plt.subplot(2, 2, 2)
        plt.plot(df['epoch'], df['train_pos'], label='Train MSE', color='tab:green')
        plt.plot(df['epoch'], df['val_pos'], label='Val MSE', color='tab:orange')
        plt.title('Position Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(True)

        # 3. Dynamic Latent Space Loss (KL vs Entropy)
        plt.subplot(2, 2, 3)
        if df['train_kl'].sum() > 0:
            # It's the CVAE
            plt.plot(df['epoch'], df['train_kl'], label='Train KL', color='purple')
            plt.plot(df['epoch'], df['val_kl'], label='Val KL', color='magenta')
            plt.title('Latent Distribution Match (KL Divergence)')
            plt.ylabel('KL Loss')
        else:
            # It's the Baseline
            plt.plot(df['epoch'], df['train_ent'], label='Train Entropy', color='purple')
            plt.plot(df['epoch'], df['val_ent'], label='Val Entropy', color='magenta')
            plt.title('Regime Diversity (Entropy)')
            plt.ylabel('Negative Entropy')
            
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)

        # 4. Turn Constraint (Aerodynamics)
        plt.subplot(2, 2, 4)
        plt.plot(df['epoch'], df['train_turn'], label='Train Turn Penalty', color='tab:brown')
        plt.plot(df['epoch'], df['val_turn'], label='Val Turn Penalty', color='tab:gray')
        plt.title('20-Degree Turn Constraint Violations')
        plt.xlabel('Epoch')
        plt.ylabel('Penalty Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        save_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_training_plot.png")
        plt.savefig(save_path)
        plt.close()