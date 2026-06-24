"""
experiment_tracking.py
=======================
Unified W&B + MLflow Experiment Tracking Utilities
EA-F-001 Hana Kim LoRA Training

Usage:
  from experiment_tracking import ExperimentTracker
  tracker = ExperimentTracker(project='hana-kim-flux-lora', run_name='rank32_cosine')
  tracker.start()
  tracker.log({'loss': 0.042, 'lr': 4e-4}, step=100)
  tracker.log_image('validation_step250.png', step=250)
  tracker.finish()
"""

import os, json, time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime


class ExperimentTracker:
    """
    Unified experiment tracker supporting Weights & Biases and MLflow.
    Gracefully degrades if either is unavailable.
    """

    def __init__(
        self,
        project: str = 'hana-kim-flux-lora',
        run_name: str = None,
        config: Optional[Dict] = None,
        wandb_api_key: Optional[str] = None,
        mlflow_uri: Optional[str] = None,
        studio_root: Optional[str] = None,
    ):
        self.project = project
        self.run_name = run_name or f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.config = config or {}
        self.wandb_api_key = wandb_api_key or os.environ.get('WANDB_API_KEY', '')
        self.studio_root = studio_root or os.getcwd()
        self.mlflow_uri = mlflow_uri or f'file://{self.studio_root}/mlruns'

        self._wandb_run = None
        self._mlflow_run = None
        self._wandb_enabled = False
        self._mlflow_enabled = False
        self._metrics_buffer: List[Dict] = []
        self._start_time = None

    def start(self):
        """Initialize both tracking backends."""
        self._start_time = time.time()
        print(f'\n🔬 Experiment: {self.run_name}')

        # W&B init
        if self.wandb_api_key and self.wandb_api_key != 'YOUR_WANDB_API_KEY':
            try:
                import wandb
                os.environ['WANDB_API_KEY'] = self.wandb_api_key
                self._wandb_run = wandb.init(
                    project=self.project,
                    name=self.run_name,
                    config=self.config,
                    reinit=True,
                )
                self._wandb_enabled = True
                print(f'  ✅ W&B initialized → {wandb.run.url}')
            except Exception as e:
                print(f'  ⚠️  W&B failed: {e}')
        else:
            print('  ℹ️  W&B skipped (no API key). Set WANDB_API_KEY env var.')

        # MLflow init
        try:
            import mlflow
            mlflow.set_tracking_uri(self.mlflow_uri)
            mlflow.set_experiment(self.project)
            self._mlflow_ctx = mlflow.start_run(run_name=self.run_name)
            self._mlflow_run = self._mlflow_ctx.__enter__()
            self._mlflow_enabled = True

            if self.config:
                mlflow.log_params({
                    k: str(v)[:250]  # MLflow param value limit
                    for k, v in self.config.items()
                })
            print(f'  ✅ MLflow initialized → {self.mlflow_uri}')
            print(f'     Run ID: {self._mlflow_run.info.run_id[:12]}...')
        except Exception as e:
            print(f'  ⚠️  MLflow failed: {e}')

        # Local fallback logger
        self._local_log_path = Path(self.studio_root) / 'reports' / f'{self.run_name}_metrics.jsonl'
        self._local_log_path.parent.mkdir(exist_ok=True)
        print(f'  ✅ Local log → {self._local_log_path}')
        return self

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to all enabled backends."""
        entry = {**metrics, 'step': step, 'timestamp': time.time()}
        self._metrics_buffer.append(entry)

        # W&B
        if self._wandb_enabled and self._wandb_run:
            try:
                import wandb
                self._wandb_run.log(metrics, step=step)
            except Exception as e:
                pass

        # MLflow
        if self._mlflow_enabled:
            try:
                import mlflow
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, float(v), step=step or 0)
            except Exception:
                pass

        # Local
        with open(self._local_log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def log_image(self, image_path: str, caption: str = '', step: Optional[int] = None):
        """Log a validation image."""
        if self._wandb_enabled and self._wandb_run:
            try:
                import wandb
                self._wandb_run.log(
                    {f'validation_{step or ""}': wandb.Image(image_path, caption=caption)},
                    step=step
                )
            except Exception:
                pass

        if self._mlflow_enabled:
            try:
                import mlflow
                mlflow.log_artifact(image_path, artifact_path=f'samples/step_{step or "final"}')
            except Exception:
                pass

    def log_checkpoint(self, checkpoint_path: str, step: int, loss: float):
        """Log model checkpoint."""
        self.log({'checkpoint_step': step, 'checkpoint_loss': loss}, step=step)

        if self._mlflow_enabled:
            try:
                import mlflow
                mlflow.log_artifact(checkpoint_path, artifact_path=f'checkpoints/step_{step}')
            except Exception:
                pass

    def log_final_metrics(self, best_step: int, final_loss: float, training_time_min: float):
        """Log training completion summary."""
        summary = {
            'best_checkpoint_step': best_step,
            'final_loss': final_loss,
            'training_time_minutes': training_time_min,
            'total_steps_logged': len(self._metrics_buffer),
        }
        self.log(summary, step=best_step)

        # W&B summary
        if self._wandb_enabled and self._wandb_run:
            try:
                self._wandb_run.summary.update(summary)
            except Exception:
                pass

    def finish(self):
        """Close all tracking backends and write summary."""
        elapsed = (time.time() - self._start_time) / 60 if self._start_time else 0

        # Final local summary
        summary_path = Path(self.studio_root) / 'reports' / f'{self.run_name}_summary.json'
        summary = {
            'run_name': self.run_name,
            'project': self.project,
            'total_metrics_logged': len(self._metrics_buffer),
            'elapsed_minutes': round(elapsed, 2),
            'wandb_enabled': self._wandb_enabled,
            'mlflow_enabled': self._mlflow_enabled,
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        if self._wandb_enabled and self._wandb_run:
            try:
                self._wandb_run.finish()
            except Exception:
                pass

        if self._mlflow_enabled:
            try:
                import mlflow
                mlflow.log_metric('elapsed_minutes', elapsed)
                self._mlflow_ctx.__exit__(None, None, None)
            except Exception:
                pass

        print(f'\n✅ Experiment finished: {self.run_name}')
        print(f'   Duration: {elapsed:.1f} min')
        print(f'   Metrics logged: {len(self._metrics_buffer)}')
        print(f'   Summary: {summary_path}')

    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot loss and LR curves from buffer."""
        import matplotlib.pyplot as plt
        import numpy as np

        loss_data = [(e['step'], e['train_loss']) for e in self._metrics_buffer
                     if 'train_loss' in e and e.get('step')]
        lr_data   = [(e['step'], e['learning_rate']) for e in self._metrics_buffer
                     if 'learning_rate' in e and e.get('step')]

        if not loss_data:
            print('No loss data to plot.')
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Training Metrics — {self.run_name}', fontweight='bold')

        steps, losses = zip(*loss_data)
        window = max(1, len(losses) // 15)
        smooth = np.convolve(losses, np.ones(window)/window, mode='valid')

        ax1.plot(steps, losses, alpha=0.3, color='steelblue', lw=0.8, label='Raw')
        ax1.plot(steps[window-1:], smooth, color='navy', lw=2, label=f'Smoothed (w={window})')
        ax1.set_xlabel('Step'); ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss'); ax1.legend(); ax1.grid(alpha=0.3)

        if lr_data:
            lr_steps, lrs = zip(*lr_data)
            ax2.plot(lr_steps, lrs, color='tomato', lw=2)
            ax2.fill_between(lr_steps, lrs, alpha=0.15, color='tomato')
            ax2.set_xlabel('Step'); ax2.set_ylabel('LR')
            ax2.set_title('Learning Rate Schedule'); ax2.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'📊 Metrics plot saved: {save_path}')
        plt.show()

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.finish()





if __name__ == '__main__':
    # Quick test / demo
    print('ExperimentTracker loaded. Import and use in your training script.')
    print('Example:')
    print('  from experiment_tracking import ExperimentTracker')
    print('  tracker = ExperimentTracker(project="hana-kim", run_name="run1")')
    print('  tracker.start()')
    print('  tracker.log({"loss": 0.042}, step=100)')
    print('  tracker.finish()')
