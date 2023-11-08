import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
import numpy as np


class Decoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, intermediate_dim: int=16, intermediate_layers: int=2, act_fn=nn.Tanh):
        super(Decoder, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.f = nn.Sequential(
                nn.Linear(input_dim, intermediate_dim),
                act_fn(),
                *[nn.Linear(intermediate_dim, intermediate_dim) if i % 2 == 0 else act_fn() for i in range(2 * intermediate_layers - 2)],
                nn.Linear(intermediate_dim, output_dim),
            )

    def forward(self, input):
        output = self.f(input)
        return output


class Periodic1dDecoder(nn.Module):
    def __init__(self, output_dim: int, intermediate_dim: int=16, intermediate_layers: int=2, act_fn=nn.Tanh):
        super(Periodic1dDecoder, self).__init__()
        
        self.decoder = Decoder(2, output_dim, intermediate_dim, intermediate_layers, act_fn)

    def forward(self, x):
        x = x / (x ** 2).sum(axis=1, keepdims=True) ** 0.5
        x = self.decoder(x)
        return x


class MSAP(pl.LightningModule):
    def __init__(
        self,
        device,
        latent_dim,
        train_dmat,
        valid_dmat,
        target_dim,
        lr=1e-2,
        batch_size=64,
        valid_lr=1e-2,
        valid_epochs=2,
        train_lam=1e-4,
        latent_lam=1e-4,
        frac=0.6,
        periodic1d = False,
        **decoder_kwargs,
    ):
        super(MSAP, self).__init__()

        self.d = device

        self.train_dmat = train_dmat.to(device)
        self.valid_dmat = valid_dmat.to(device)
        self.lr = lr
        self.batch_size = batch_size
        self.train_lam = train_lam
        self.latent_lam = latent_lam
        self.valid_epochs = valid_epochs
        self.frac = frac

        latent_space_dim = 2 if periodic1d and latent_dim == 1 else latent_dim
        latent_init = self.get_init(train_dmat, latent_space_dim)
        self.latent = nn.Parameter(latent_init.to(device), True)

        valid_latent_init = self.get_init(valid_dmat, latent_space_dim, True)
        self.valid_latent = nn.Parameter(valid_latent_init.to(device), True)
        self.valid_optimizer = torch.optim.Adam([self.valid_latent], lr=valid_lr)
        self.valid_dataloader = self.get_dataloader("validation")
        self.valid_data = self.valid_dataloader.dataset

        if periodic1d and latent_dim == 1:
            self.decoder = Periodic1dDecoder(target_dim, **decoder_kwargs).to(device)
        else:
            self.decoder = Decoder(self.latent.shape[1], target_dim, **decoder_kwargs).to(device)
        
        self.p = nn.Parameter(0.5 * torch.rand(1).to(device), True)

    def get_init(self, dmat, latent_dim, use_prev=False):
        if use_prev:
            dmat = torch.nan_to_num(dmat, nan=dmat.mean())
            init = dmat[:, self.init_idx]
            if latent_dim == 1:
                other = dmat[:, [self.other_idx]]
                init = init * ((init < other) * 2 - 1)
            return torch.tensor(self.init_scaler.transform(init), dtype=torch.float32)
        self.init_idx = torch.argsort((torch.nan_to_num(dmat, nan=0) ** 2).sum(axis=1).flatten())[-latent_dim:]
        init = torch.nan_to_num(dmat, nan=dmat.mean())[:, self.init_idx]
        if latent_dim == 1:
            self.other_idx = torch.argsort(init.T[0])[len(dmat) // 10]
            other = torch.nan_to_num(dmat, nan=dmat.mean())[:, [self.other_idx]]
            init = init * ((init < other) * 2 - 1)
        
        self.init_scaler = StandardScaler()
        return torch.tensor(self.init_scaler.fit_transform(init), dtype=torch.float32)

    def pow(self):
        return 1.1 + self.p ** 2

    def forward(self):
        outputs = self.decoder(self.latent)
        return outputs

    def compute_loss(self, idx, training=True, eval=False):
        if training:
            fst, snd = self()[idx.T]
            dmat = self.train_dmat
            p = self.pow()
            r1 = self.train_lam * (torch.cat([p.flatten() for p in self.decoder.parameters()]) ** 2).sum()
            r2 = self.latent_lam * (self.latent ** 2).sum()
        else:
            fst = self.decoder(self.valid_latent)[idx.T[0]]
            snd = self().detach()[idx.T[1]]
            dmat = self.valid_dmat
            p = self.pow().detach()
            r1 = 0
            r2 = self.latent_lam * (self.valid_latent ** 2).sum()
        
        dists = (torch.abs(fst - snd) ** p + 1e-3).sum(axis=1) ** (1 / p)
        true_dists = dmat[idx.T[0], idx.T[1]]

        stress = ((dists - true_dists) ** 2).sum() / (dists ** 2).sum()
        if eval:
            return stress
        
        loss = stress + r1 + r2
        if training:
            self.log("stress", stress)
            self.log("r1", r1)
            self.log("r2", r2)
            self.log("loss", loss)
            self.log("p", p)
        return loss

    def training_step(self, idx, batch_num):
        loss = self.compute_loss(idx)

        for _ in range(self.valid_epochs):
            valid_loss = self.compute_loss(self.valid_data, False)
            valid_loss.backward()
            self.valid_optimizer.step()
            self.valid_optimizer.zero_grad()
        valid_loss = self.eval_valid()
        self.log("valid_loss", valid_loss, prog_bar=True)
        return loss

    def eval_valid(self):
        return self.compute_loss(self.valid_data, False, True).cpu().detach()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_dataloader(self, purpose, frac=None):
        if frac is None:
            frac = 0.6

        if purpose == "train":
            dmat = torch.nan_to_num(self.train_dmat, nan=torch.inf)
        if purpose == "validation":
            dmat = torch.nan_to_num(self.valid_dmat, nan=torch.inf)

        N = int(frac * dmat.shape[1])
        idx = torch.argsort(dmat, axis=1)[:, 1:N]
        pairs = torch.cat([torch.tensor([(i, el) for el in idx[i]]) for i in range(len(dmat))])
        pairs.to(self.d)
        if purpose == "train":
            self.n_iterations = np.ceil(len(pairs) / self.batch_size)
        return DataLoader(pairs, batch_size=self.batch_size, shuffle=False if purpose == "validation" else True)

    def train_dataloader(self):
        return self.get_dataloader("train")
