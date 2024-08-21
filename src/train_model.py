from sklearn.model_selection import KFold
from build_dataset import *

def train_model(data_filepath, model_outpath, params):
    checkpoint_dir = os.path.join(model_outpath, "checkpoints")
    combined_model_path = os.path.join(model_outpath, "combined_model.h5")
    data = data_gen(data_filepath, model_outpath, params)

    # Calculate steps per epoch
    params['steps_per_epoch'] = np.ceil(len(data.train_data) / batch_size).astype(int)
    params['validation_steps'] = np.ceil(len(val_fold_data) / batch_size).astype(int)

        # Check if veto model exists
        # veto_model = load_veto_model_if_exists(checkpoint_dir, fold)
        veto_model = tf.keras.models.load_model(VETO_MODEL)

        if veto_model is None:
            # If no veto model exists, create generator instances and train a new veto model
            veto_train_gen = DataGenerator(train_fold_data, scalers, batch_size=batch_size)
            veto_val_gen = DataGenerator(val_fold_data, scalers, batch_size=batch_size)

            veto_train_dataset = tf.data.Dataset.from_generator(
                veto_train_gen, output_signature=(
                    tf.TensorSpec(shape=(None, 18), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32)  # mask as the target
                )
            ).take(len(train_fold_data)).cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)

            veto_val_dataset = tf.data.Dataset.from_generator(
                veto_val_gen, output_signature=(
                    tf.TensorSpec(shape=(None, 18), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32)  # mask as the target
                )
            ).take(len(val_fold_data)).cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)

            #print(f"Training veto model on fold {fold}/{n_splits}...")
            veto_model, history = train_veto_model(veto_train_dataset, veto_val_dataset, params, checkpoint_dir)
            veto_model_path = os.path.join(checkpoint_dir, f"veto_model_fold_{fold}.h5")
            veto_model.save(veto_model_path)

        # Create generator instances for training and validation datasets using veto model
        train_gen = DataGeneratorWithVeto(train_fold_data, scalers, veto_model, batch_size=batch_size)
        val_gen = DataGeneratorWithVeto(val_fold_data, scalers, veto_model, batch_size=batch_size)

        # Create tf.data.Dataset from the generator
        train_dataset = tf.data.Dataset.from_generator(
            train_gen, output_signature=(
                tf.TensorSpec(shape=(None, 18), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
            )
        ).take(len(train_fold_data)).cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_generator(
            val_gen, output_signature=(
                tf.TensorSpec(shape=(None, 18), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
            )
        ).take(len(val_fold_data)).cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)

        # Train the main model
        #print(f"Training main model on fold {fold}/{n_splits}...")
        model, history = train_main_model(train_dataset, val_dataset, params, checkpoint_dir)

        model.save(os.path.join(out_path, f"main_model_fold_{fold}.h5"))
        with open(os.path.join(out_path, f"history_fold_{fold}.pkl"), 'wb') as f:
            pickle.dump(history.history, f)

        print(f"Models and history for fold {fold} saved.")
        fold_models.append(model)
        fold += 1

        # Combine models by averaging their weights
    combined_model = create_main_model()  # Use your model creation function
    combined_weights = [model.get_weights() for model in fold_models]
    new_weights = []

    for weights_tuple in zip(*combined_weights):
        new_weights.append([np.mean(np.array(w), axis=0) for w in zip(*weights_tuple)])

    combined_model.set_weights(new_weights)
    combined_model.save(combined_model_path)

    # Final evaluation on test data
    x_test, y_test = preprocess_surrogate_test_data(test_data, scalers, veto_model, combined_model)
    loss_test = evaluate(combined_model, x_test, y_test, plot=False)

    print(f"Final test_loss: {loss_test:.4f}")


if __name__ == '__main__':
    p = ' '.join(sys.argv[2:])
    p = re.findall(r'(\w+)=(\S+)', p)
    params = {}
    for key, value in p:
        try:
            params[key] = float(value)
        except ValueError:
            params[key] = value
    output_file_path = sys.argv[1]
    run_train(output_file_path, params, n_splits=3)
