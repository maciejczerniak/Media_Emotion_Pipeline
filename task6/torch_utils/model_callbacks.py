import os

import torch


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving"""

    def __init__(self, patience=7, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True


class ModelCheckpoint:
    """Save the best model during training"""

    def __init__(
        self,
        filepath,
        monitor="val_loss",
        mode="min",
        verbose=True,
        save_best_only=True,
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.best_score = None

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def __call__(
        self, model, current_score, epoch, optimizer=None, additional_info=None
    ):
        if self.mode == "min":
            score_improved = self.best_score is None or current_score < self.best_score
        else:  # mode == 'max'
            score_improved = self.best_score is None or current_score > self.best_score

        if not self.save_best_only or score_improved:
            if score_improved:
                self.best_score = current_score
                if self.verbose:
                    print(
                        f"Saving model checkpoint at epoch {epoch + 1} with {self.monitor}: {current_score:.4f}"
                    )

            # Save model state
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
                "best_score": self.best_score,
                "current_score": current_score,
            }

            if additional_info:
                checkpoint.update(additional_info)

            torch.save(checkpoint, self.filepath)
