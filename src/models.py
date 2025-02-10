import torch
import torch.nn as nn

class DummySegmentationModel(nn.Module):
    def __init__(self, num_classes=55):
        super(DummySegmentationModel, self).__init__()
        # A simple stack of convolutional layers.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Forward pass through the network.
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)  # Output shape: (batch_size, num_classes, H, W)
        return x



class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=55, base_channels=64):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        # Encoder (5 levels)
        self.encoder1 = conv_block(in_channels, base_channels)
        self.encoder2 = conv_block(base_channels, base_channels * 2)
        self.encoder3 = conv_block(base_channels * 2, base_channels * 4)
        self.encoder4 = conv_block(base_channels * 4, base_channels * 8)
        self.encoder5 = conv_block(base_channels * 8, base_channels * 16)  # New level

        # Bottleneck
        self.bottleneck = conv_block(base_channels * 16, base_channels * 32)

        # Decoder
        self.upconv5 = nn.ConvTranspose2d(base_channels * 32, base_channels * 16, kernel_size=2, stride=2)
        self.decoder5 = conv_block(base_channels * 32, base_channels * 16)
        self.upconv4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.decoder4 = conv_block(base_channels * 16, base_channels * 8)
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.decoder3 = conv_block(base_channels * 8, base_channels * 4)
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.decoder2 = conv_block(base_channels * 4, base_channels * 2)
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.decoder1 = conv_block(base_channels * 2, base_channels)

        # Final output
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        enc5 = self.encoder5(self.pool(enc4))  # New level
        bottleneck = self.bottleneck(self.pool(enc5))

        # Decoder
        dec5 = self.upconv5(bottleneck)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)
        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)