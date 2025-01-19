import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import time
from torch.cuda.amp import GradScaler, autocast


class ModelTrainer:
    def __init__(self, model, optimizer, loss_fn, device=None, scheduler=None, metrics=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scheduler = scheduler
        self.metrics = metrics or {}
        self.model.to(self.device)
        self.scaler = GradScaler()  # For mixed precision training

    def train(
        self, train_loader, val_loader=None, epochs=10, log_interval=10, save_path=None, accumulation_steps=1
    ):
        history = {"train_loss": [], "val_loss": [], "metrics": []}
        best_val_loss = float("inf")

        for epoch in range(epochs):
            print(f"\n[MODEL_TRAINER] Epoch {epoch + 1}/{epochs}")
            epoch_start = time.time()

            # Training phase
            self.model.train()
            train_loss = 0.0
            for step, batch in enumerate(tqdm(train_loader, desc="Training"), 1):
                inputs, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                with autocast():  # Mixed precision
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels) / accumulation_steps

                self.scaler.scale(loss).backward()

                if step % accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                train_loss += loss.item() * accumulation_steps
                if step % log_interval == 0:
                    print(f"[MODEL_TRAINER] Step {step}/{len(train_loader)}, Loss: {loss.item() * accumulation_steps:.4f}")

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)
            print(f"[MODEL_TRAINER] Epoch {epoch + 1} Train Loss: {train_loss:.4f}")

            # Validation phase
            if val_loader:
                val_loss, val_metrics = self.validate(val_loader)
                history["val_loss"].append(val_loss)
                history["metrics"].append(val_metrics)
                print(f"[MODEL_TRAINER] Epoch {epoch + 1} Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")

                # Save best model
                if save_path and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), save_path)
                    print(f"[MODEL_TRAINER] Saved best model with Val Loss: {best_val_loss:.4f}")

            # Step the scheduler
            if self.scheduler:
                self.scheduler.step()

            epoch_end = time.time()
            print(f"[MODEL_TRAINER] Epoch {epoch + 1} completed in {epoch_end - epoch_start:.2f}s")

        return history

    def validate(self, val_loader):
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
                    metric_value = metric_fn(outputs, labels)
                    if isinstance(metric_value, torch.Tensor):
                        if metric_value.numel() > 1:
                            metric_value = metric_value.mean()  # Reduce to a single value
                        metrics[name] += metric_value.item()
                    else:
                        metrics[name] += torch.tensor(metric_value).item()

        val_loss /= len(val_loader)
        for name in metrics:
            metrics[name] /= len(val_loader)

        return val_loss, metrics

    def evaluate(self, test_loader):
        test_loss, test_metrics = self.validate(test_loader)
        return {"loss": test_loss, "metrics": test_metrics}

    def predict(self, loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting"):
                inputs = batch["image"].to(self.device)
                outputs = self.model(inputs)
                predictions.append(outputs.cpu())
        return predictions
