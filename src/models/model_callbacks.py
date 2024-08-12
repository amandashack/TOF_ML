

def get_latest_checkpoint(checkpoint_dir, model_name="veto_cp"):
    checkpoint_pattern = os.path.join(checkpoint_dir, f"{model_name}-*.index")
    checkpoints = glob.glob(checkpoint_pattern)
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    latest_checkpoint = latest_checkpoint.replace('.index', '')
    return latest_checkpoint


def get_best_checkpoint(checkpoint_dir, model_name="veto_cp"):
    checkpoint_pattern = os.path.join(checkpoint_dir, f"{model_name}-*.index")
    checkpoints = glob.glob(checkpoint_pattern)
    if not checkpoints:
        return None
    best_checkpoint = None
    best_val_loss = float('inf')
    for checkpoint in checkpoints:
        # Extract the epoch and validation loss from the checkpoint file name
        match = re.search(rf"{model_name}-(\d+).index", checkpoint)
        if match:
            epoch = int(match.group(1))
            val_loss_file = checkpoint.replace(".index", ".val_loss")
            if os.path.exists(val_loss_file):
                with open(val_loss_file, 'r') as f:
                    val_loss = float(f.read().strip())
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_checkpoint = checkpoint.replace('.index', '')
    return best_checkpoint

class MemoryUsageCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        memory_info = tf.config.experimental.get_memory_info('GPU:0')
        print(f" - Batch {batch + 1} - GPU Memory Usage: {memory_info['current'] / 1024**2:.2f} "
              f"MB / {memory_info['peak'] / 1024**2:.2f} MB")