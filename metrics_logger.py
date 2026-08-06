import tensorflow as tf
from tensorflow import keras

class MetricsLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = self.model.optimizer.learning_rate
        if hasattr(lr, 'numpy'):
            lr = lr.numpy()
        else:
            lr = float(lr)
        
        print(f"\n📊 Época {epoch+1}: "
              f"loss={logs.get('loss', 0):.4f}, "
              f"val_loss={logs.get('val_loss', 0):.4f}, "
              f"auc={logs.get('auc', 0):.4f}, "
              f"val_auc={logs.get('val_auc', 0):.4f}, "
              f"acc={logs.get('accuracy', 0):.4f}, "
              f"val_acc={logs.get('val_accuracy', 0):.4f}, "
              f"lr={lr:.2e}")
