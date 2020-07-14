import tensorflow as tf
# Import losses:
from tensorflow.keras.losses import MeanSquaredError
# Import optimizers:
from tensorflow.keras.optimizers import Adam
# Import metrics:
from tensorflow.keras.metrics import (
  Mean, MeanAbsolutePercentageError
)
# Import models:
from models.speednet import SpeedNet
# Import processing:
from preprocess.process import Process


class Train(object):
    def __init__(self, params):
        self.epochs = 10000
        # Define loss:
        self.loss_object = MeanSquaredError()
        # Define optimizer:
        self.optimizer = Adam(1e-3)
        # Define metrics for loss:
        self.train_loss = Mean()
        self.train_accuracy = MeanAbsolutePercentageError()
        self.test_loss = Mean()
        self.test_accuracy = MeanAbsolutePercentageError()
        # Define model:
        self.speed_net = SpeedNet()
        # Define pre processor (params):
        preprocessor = Process()
        self.train_ds, self.test_ds = preprocessor.get_datasets()
        # Define Checkpoints:
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer,
                net=self.model)
        # Define Checkpoint manager:
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, f'checkpoints{params.ckpt_dir}',
                max_to_keep=3)

    # Feed forward through and update model on train data:
    @tf.function
    def _update(self, cur_frame, ref_frame, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(cur_frame, ref_frame, True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    # Feed forward through model on test data:
    @tf.function
    def _test(self, cur_frame, ref_frame, labels):
        predictions = self.model(cur_frame, ref_frame)
        loss = self.loss_object(labels, predictions)

        self.test_loss(loss)
        self.test_accuracy(labels, predictions)

    # Log status of each epoch:
    def _log(self, epoch):
        template = 'Epoch {}, Loss: {}, Acc: {}, Test Loss: {}, Test Acc: {}'
        print(template.format(epoch + 1,
            self.train_loss.result(),
            self.train_accuracy.result() * 100,
            self.test_loss.result(),
            self.test_accuracy.result() * 100))

    # Save model to checkpoint:
    def _save(self, verbose=False):
        save_path = self.ckpt_manager.save()]
        if verbose:
            ckptLog = f"Saved checkpoint for step {int(self.ckpt.step)}: {save_path}"
            print(ckptLog)

    # Restore model from checkpoint:
    def _restore(self):
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()
        if self.ckpt_manager.latest_checkpoint:
            print(f"Restored from {self.ckpt_manager.latest_checkpoint}")
        else:
            print("Initializing from scratch.")

    # Reset network metrics:
    def _reset(self):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()

    # Train loop for network:
    def train(self):
        self._restore()
        for epoch in range(self.epochs):
            for ref_frame, cur_frame, labels in self.train_ds:
                self._update(ref_frame, cur_frame, labels)
            for test_ref_frame, cur_ref_frame, test_labels in self.test_ds:
                self._test(test_ref_frame, cur_ref_frame, test_labels)
            self._log(epoch)
            self._save()
            self._reset()
