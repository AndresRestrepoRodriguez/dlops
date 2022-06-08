import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


def block_conv(input_channels, output_channels, kernel_size=3, padding=1, stride=1, pool_kernel=2, pool_stride=2):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size, padding=padding, stride=stride),
        nn.ReLU(),
        nn.MaxPool2d(pool_kernel, stride=pool_stride)
    )


class MNISTModule(pl.LightningModule):
    def __init__(self, n_channels=1, n_outputs=1, cfg=None):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.conv1 = block_conv(n_channels, 32)
        #self.conv2 = block_conv(32, 64)
        self.mlp = nn.Sequential(
            nn.Linear(14 * 14 * 32, 100),
            nn.ReLU(),
            nn.Linear(100, n_outputs)
        )
    

    def forward(self, x):
        x = x.float() / 255
        x = self.conv1(x)
        #x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x).squeeze(-1)
        return x


    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self(x))

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
        preds = torch.sigmoid(y_hat) > 0.5
        acc = (preds.long() == y).float().mean()
        #self.log('loss', loss)
        self.log('acc', acc,  prog_bar=True)
        return loss

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
        preds = torch.sigmoid(y_hat) > 0.5
        val_acc = (preds.long() == y).float().mean()
        self.log('val_loss', val_loss,  prog_bar=True)
        self.log('val_acc', val_acc,  prog_bar=True)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
