import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow

from src.loss import vae_loss
from src.model import VAE


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        verbose_period: int,
        device: torch.device,
        transform=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.verbose_period = verbose_period
        self.device = device
        self.transform = transform

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

    def evaluate(self, **kwarg):
        pass

    def _train(self, **kwarg):
        pass

    def _valid(self, **kwarg):
        pass


class VAETrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        verbose_period: int,
        device: torch.device,
        transform=None,
    ) -> None:
        super().__init__(model, optimizer, verbose_period, device, transform)

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
        if verbose:
            mse, kl = self.evaluate(dataloader, verbose, epoch_id)
            mlflow.log_metrics({"val_reconstr_loss": mse, "val_kl": kl}, step=epoch_id)
        return mse

    def evaluate(self, dataloader, verbose, epoch_id):
        vae: VAE = self.model
        vae.eval()
        device = self.device

        total_reconstr_loss, total_kl = 0.0, 0.0

        with torch.no_grad():
            for batch in tqdm(
                dataloader, disable=not verbose, desc=f"val-epoch {epoch_id}"
            ):
                X, label = batch[0], batch[1].reshape(-1).long()
                X, label = X.to(device), label.to(device)
                if self.transform:
                    X = self.transform(X)
                X_hat, latent_params = vae(X)
                _recontr_loss, _kl = vae_loss(X_hat, X, **latent_params)

                total_reconstr_loss += _recontr_loss
                total_kl += _kl

        mse = float(total_reconstr_loss / len(dataloader))
        kl = float(total_kl / len(dataloader))

        if verbose:
            print("val_recontr_loss={:.3f}, val_kl={:.3f}".format(mse, kl))

        return mse, kl
