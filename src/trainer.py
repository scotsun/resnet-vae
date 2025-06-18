"""Trainer classes & training infra functionalities integrated with MLflow."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow

from src.loss import vae_loss
from src.model import VAE


class EarlyStopping:
    def __init__(
        self, patience: int, min_delta: int = 0, mode: str = "min", model_signature=None
    ) -> None:
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.model_signature = model_signature

        match mode:
            case "min":
                self.monitor_op, self.delta_op = lambda a, b: a < b, -1 * min_delta
            case "max":
                self.monitor_op, self.delta_op = lambda a, b: a > b, min_delta
            case _:
                raise ValueError("mode must be either `min` or `max`")

    def _log_best_model(self, model):
        """helper function to log model."""
        mlflow.pytorch.log_model(
            model,
            "best_model",
            pip_requirements=["torch==2.2.1+cu121"],
            signature=self.model_signature,
        )

    def step(self, metric_val, model):
        # save the first chkpt
        if self.best_score is None:
            self.best_score = metric_val
            self._log_best_model(model)
            return
        # save the subsequent chkpt
        if self.monitor_op(metric_val, self.best_score + self.delta_op):
            self.best_score = metric_val
            self.counter = 0
            self._log_best_model(model)
            return
        else:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True
            return


class Trainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        early_stopping: EarlyStopping | None,
        verbose_period: int,
        device: torch.device,
        transform=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.verbose_period = verbose_period
        self.device = device
        self.transform = transform
        self.best_score = -float("inf")

    def fit(
        self,
        epochs: int,
        train_loader: DataLoader,
        valid_loader: None | DataLoader = None,
    ):
        for epoch in range(epochs):
            verbose = (epoch % self.verbose_period) == 0
            self._train(train_loader, verbose, epoch)
            if valid_loader is not None:
                self._valid(valid_loader, verbose, epoch)
                if self.early_stopping and self.early_stopping.early_stop:
                    break

    @abstractmethod
    def evaluate(self, **kwarg):
        pass

    @abstractmethod
    def _train(self, **kwarg):
        pass

    @abstractmethod
    def _valid(self, **kwarg):
        pass


class VAETrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        early_stopping: EarlyStopping | None,
        verbose_period: int,
        device: torch.device,
        transform=None,
    ) -> None:
        super().__init__(
            model, optimizer, early_stopping, verbose_period, device, transform
        )

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        vae: VAE = self.model
        vae.train()
        optimizer = self.optimizer
        device = self.device

        with tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch_id}")
            for batch in bar:
                X = batch[0].to(device)
                if self.transform:
                    X = self.transform(X)
                optimizer.zero_grad()
                X_hat, latent_params = vae(X)
                _reconstr_loss, _kl = vae_loss(X_hat, X, **latent_params)
                loss = _reconstr_loss + _kl
                loss.backward()
                optimizer.step()

                bar.set_postfix(
                    recontr_loss=float(_reconstr_loss),
                    kl_c=float(_kl),
                )
        return

    def _valid(self, dataloader, verbose, epoch_id):
        """
        log all the metrics in mlflow;
        return the metric for save-best/early-stop.
        """
        if verbose:
            mse, kl = self.evaluate(dataloader, verbose, epoch_id)
            mlflow.log_metrics({"val_reconstr_loss": mse, "val_kl": kl}, step=epoch_id)
        if self.early_stopping:
            self.early_stopping.step(mse, self.model)
        return mse

    def evaluate(self, dataloader, verbose, epoch_id):
        vae: VAE = self.model
        vae.eval()
        device = self.device

        total_reconstr_loss, total_kl = 0.0, 0.0
        num_batches = len(dataloader)

        with torch.no_grad():
            for batch in tqdm(
                dataloader, disable=not verbose, desc=f"val-epoch {epoch_id}"
            ):
                X = batch[0].to(device)
                if self.transform:
                    X = self.transform(X)
                X_hat, latent_params = vae(X)
                _recontr_loss, _kl = vae_loss(X_hat, X, **latent_params)

                total_reconstr_loss += _recontr_loss
                total_kl += _kl

        mse = float(total_reconstr_loss / num_batches)
        kl = float(total_kl / num_batches)

        if verbose:
            print("val_recontr_loss={:.3f}, val_kl={:.3f}".format(mse, kl))

        return mse, kl


def test_logging(
    trainer: Trainer,
    test_loader: DataLoader,
    metric_names: list[str],
    expr_name: str,
    run_name: str,
):
    mlflow.set_experiment(expr_name)
    run_data = mlflow.search_runs(filter_string=f"attributes.run_name = '{run_name}'")
    if run_data.empty:
        raise ValueError(f"run_name={run_name} does not exist")
    else:
        run_id = run_data.iloc[0].run_id
        try:
            with mlflow.start_run(run_id=run_id):
                trainer.model = mlflow.pytorch.load_model(f"runs:/{run_id}/best_model")
                test_scores = trainer.evaluate(test_loader, True, 0)
                mlflow.log_metrics(metrics=dict(zip(metric_names, test_scores)), step=0)
            print(f"Successfully added metrics to run {run_id}")
        except Exception as e:
            print(f"Error updating run: {e}")
