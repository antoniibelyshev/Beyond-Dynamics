
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb

from msap import MSAP


def train_MSAP(train_dmat, valid_dmat, config, name, N_latent=4, N_inits=3, project_name="MSAP", device_name=None, **early_stopping_kwargs):
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    
    models = []
    valid_losses = []
    cfg = config.copy()
    cfg.update(early_stopping_kwargs)
    for latent_dim in range(1, N_latent + 1):
        print(f"latent_dim: {latent_dim}")
        best_valid_loss = torch.inf
        best_model = None
        for i in range(N_inits):
            print(f"init #{i + 1}")

            model = MSAP(
                device,
                latent_dim,
                train_dmat,
                valid_dmat,
                **config,
            )
            model.to(device)

            cfg["latent_dim"] = latent_dim
            wandb.init(project=project_name, config=cfg, name=name + f"; latent_dim: {latent_dim}, init #{i + 1}")
            wandb_logger = WandbLogger(project=project_name, log_model=True)

            early_stopping_callback = RelativeEarlyStopping(**early_stopping_kwargs)

            checkpoint_callback = ModelCheckpoint(
                monitor='valid_loss',
                mode='min',
                save_top_k=1,
                save_last=False,
            )

            kwargs = {}
            if device_name == "cuda":
                kwargs["gpus"] = 1
                
            trainer = pl.Trainer(
                max_epochs=1000,
                callbacks=[early_stopping_callback, checkpoint_callback],
                logger = wandb_logger,
                **kwargs,
            )
            trainer.fit(model)

            checkpoint_path = checkpoint_callback.best_model_path
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.to(device)

            valid_loss = model.eval_valid().numpy()
            print(f"\n\nbest validation loss: {valid_loss}")
            print("------------------------------------------------------------------------\n\n")
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model = model
            wandb.finish()
        print(f"\n\nbest validation loss over latent dim: {best_valid_loss}")
        print("---------------------------------------------------------------------------\n\n\n\n")
        models.append(best_model)
        valid_losses.append(best_valid_loss)
    return models, valid_losses


class RelativeEarlyStopping(Callback):
    def __init__(self, monitor="valid_loss", min_delta=0.1, patience=100):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best_score = None
        self.mode = "min"

    def on_train_epoch_end(self, trainer, pl_module):
        current_score = trainer.callback_metrics[self.monitor]

        if self.best_score is None:
            self.best_score = current_score
        elif (self.mode == "min" and current_score < self.best_score * (1 - self.min_delta)) or (
            self.mode == "max" and current_score > self.best_score * (1 + self.min_delta)
        ):
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                trainer.should_stop = True
