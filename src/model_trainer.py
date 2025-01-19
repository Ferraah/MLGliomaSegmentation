import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import time


class ModelTrainer:
    def __init__(self, model, optimizer, loss_fn, device=None, scheduler=None, metrics=None):
        """
        Initialize the trainer.

        Args:
            model (torch.nn.Module): The PyTorch model to train.
            optimizer (torch.optim.Optimizer): Optimizer for model training.
            loss_fn (callable): Loss function.
            device (str): Device to use ('cpu' or 'cuda'). Default is auto-detect.
            scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
            metrics (dict, optional): Dictionary of metrics with names as keys and callables as values.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scheduler = scheduler
        self.metrics = metrics or {}
        self.model.to(self.device)

    def train(
        self, train_loader, val_loader=None, epochs=10, log_interval=10, save_path=None
    ):
        """
        Train the model.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader, optional): DataLoader for validation data.
            epochs (int): Number of training epochs.
            log_interval (int): Log progress every `log_interval` steps.
            save_path (str, optional): Path to save the best model.

        Returns:
            dict: Training history including loss and metrics for each epoch.
        """
        history = {"train_loss": [], "val_loss": [], "metrics": []}
        best_val_loss = float("inf")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_start = time.time()

            # Training phase
            self.model.train()
            train_loss = 0.0
            for step, batch in enumerate(tqdm(train_loader, desc="Training"), 1):
                inputs, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                if step % log_interval == 0:
                    print(f"Step {step}/{len(train_loader)}, Loss: {loss.item():.4f}")

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)
            print(f"Epoch {epoch + 1} Train Loss: {train_loss:.4f}")

            # Validation phase
            if val_loader:
                val_loss, val_metrics = self.validate(val_loader)
                history["val_loss"].append(val_loss)
                history["metrics"].append(val_metrics)
                print(f"Epoch {epoch + 1} Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")

                # Save best model
                if save_path and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Saved best model with Val Loss: {best_val_loss:.4f}")

            # Step the scheduler
            if self.scheduler:
                self.scheduler.step()

            epoch_end = time.time()
            print(f"Epoch {epoch + 1} completed in {epoch_end - epoch_start:.2f}s")

        return history

    def validate(self, val_loader):
        """
        Validate the model.

        Args:
            val_loader (DataLoader): DataLoader for validation data.

        Returns:
            tuple: Validation loss and metrics.
        """
        self.model.eval()
        val_loss = 0.0
        metrics = {name: 0.0 for name in self.metrics}

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                inputs, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                outputs = self.model(inputs)
                val_loss += self.loss_fn(outputs, labels).item()

                # Calculate metrics
                for name, metric_fn in self.metrics.items():
                    metrics[name] += metric_fn(outputs, labels).item()

        val_loss /= len(val_loader)
        for name in metrics:
            metrics[name] /= len(val_loader)

        return val_loss, metrics

    def evaluate(self, test_loader):
        """
        Evaluate the model on test data.

        Args:
            test_loader (DataLoader): DataLoader for test data.

        Returns:
            dict: Loss and metrics on test data.
        """
        test_loss, test_metrics = self.validate(test_loader)
        print(f"Test Loss: {test_loss:.4f}, Metrics: {test_metrics}")
        return {"loss": test_loss, "metrics": test_metrics}

    def predict(self, loader):
        """
        Make predictions using the model.

        Args:
            loader (DataLoader): DataLoader for input data.

        Returns:
            list: Predictions.
        """
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting"):
                inputs = batch["image"].to(self.device)
                outputs = self.model(inputs)
                predictions.append(outputs.cpu())
        return predictions
