import torch.nn.functional as F
from torch import nn
import torch


class UNET(nn.Module):
    def _encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
        )

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
        )
        
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.enc1 = self._encoder_block(3, 32)   
        self.enc2 = self._encoder_block(32, 64)  
        self.enc3 = self._encoder_block(64, 128) 
        
        self.bt1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bt2 = nn.Conv2d(256, 256, 3, padding=1)

        self.dec1 = self._decoder_block(256, 128)   
        self.dec2 = self._decoder_block(192, 64)  
        self.dec3 = self._decoder_block(96, 32) 
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 3, padding=1),
        )
        
    def forward(self, x):
        # x is [3, 32, 32]
        x1 = self.enc1(x) # [32, 16, 16]
        x2 = self.enc2(x1)  # [64, 8, 8]
        x3 = self.enc3(x2) # [128, 4, 4]
        
        x4 = self.bt1(x3) # [256, 4, 4]
        x5 = F.relu(x4)
        x6 = self.bt2(x5) # [256, 4, 4]
        x7 = F.relu(x6)

        x8 = self.dec1(x7) # [128, 8, 8]
        
        x8 = torch.cat([x8, x2], dim=1) # Concat [128, 8, 8] && [64, 8, 8]
        # 192 Channels
        x9 = self.dec2(x8) # [64, 16, 16]
        
        x9 = torch.cat([x9, x1], dim=1) # Concat [64, 16, 16] && [32, 16, 16]
        # 96 Channels
        x10 = self.dec3(x9) # [32, 32, 32]
        
        x11 = self.dec4(x10) # [3, 32, 32]
        
        