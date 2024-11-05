import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),  # Consider BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256], max_masks=1):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_masks = max_masks

        # Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        self.final_conv = nn.Conv2d(features[0], out_channels * self.max_masks, kernel_size=1, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear')

            concat = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat)

        output = self.final_conv(x)
        output = self.sigmoid(output)

        # Remove extra mask channels if needed. OPTIONAL, commentable.
        # if output.shape[1] > masks.shape[1]:
        #     output = output[:, :masks.shape[1], :, :]

        return output.view(-1, self.max_masks, output.shape[2], output.shape[3])


def multi_mask_dice_loss(pred, target, smooth=1.):  # Ensure pred has been passed through Sigmoid
    # dim=(1, 2) # potentially useful for both single and multi mask
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)

    # Calculate dice loss for each mask and average
    return 1 - dice.mean()


class CoronarySegmentationModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, max_masks=1):
        super().__init__()
        self.model = UNet(in_channels=1, out_channels=1, max_masks=max_masks)
        self.loss_function = multi_mask_dice_loss
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        preds = self.model(images)
        loss = self.loss_function(preds, masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        preds = self.model(images)
        loss = self.loss_function(preds, masks)
        self.log('val_loss', loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer