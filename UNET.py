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

    def sinusoidal_embedding(self, t, d):
        i = torch.arange(
            d // 2,
            dtype=torch.float32,
            device=t.device
        )
    
        freqs = t[:, None] / (10000 ** (2 * i[None, :] / d))
    
        return torch.cat(
            [torch.sin(freqs), torch.cos(freqs)],
            dim=-1
        ).to(device=t.device)

    def mlp_stack(self, d, d_hidden):
        return nn.Sequential(
            nn.Linear(d, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d)
        )

    def ProjectionLayer(self, temb, Cblock):
        return nn.Linear(temb.shape[1], Cblock)
    
    def _decoder_block(self, in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(),
            )
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.D = 128
        self.cblocks = [32, 64, 128, 256, 256, 128, 64, 32, ]
        self.proj_layers = nn.ModuleList([nn.Linear(self.D, c) for c in self.cblocks])

        self.enc1 = self._encoder_block(16, 32)   
        self.enc2 = self._encoder_block(32, 64)  
        self.enc3 = self._encoder_block(64, 128) 
        
        self.mlp = self.mlp_stack(self.D, self.D*2)
        
        self.bt1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bt2 = nn.Conv2d(256, 256, 3, padding=1)

        self.dec1 = self._decoder_block(256, 128)   
        self.dec2 = self._decoder_block(192, 64)  
        self.dec3 = self._decoder_block(96, 32) 
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, padding=1),
        )
        
    def forward(self, x, t):
        temb = self.mlp(self.sinusoidal_embedding(t, self.D))
        # x is [16, 32, 32], standard CIFAR resolution
        layers = [proj(temb)[:, :, None, None] for proj in self.proj_layers]
            
        x1 = self.enc1(x) # [32, 16, 16]
        x1 = x1 + layers[0]
        
        x2 = self.enc2(x1)  # [64, 8, 8]
        x2 = x2 + layers[1]
        
        x3 = self.enc3(x2) # [128, 4, 4]
        x3 = x3 + layers[2]
        
        x4 = self.bt1(x3) # [256, 4, 4]
        x4 = x4 + layers[3]
        
        x5 = F.relu(x4)
        
        x6 = self.bt2(x5) # [256, 4, 4]
        x6 = x6 + layers[4]
        
        x7 = F.relu(x6)
        
        x8 = self.dec1(x7) # [128, 8, 8]
        x8 = x8 + layers[5]
        x8 = torch.cat([x8, x2], dim=1) # Concat [128, 8, 8] && [64, 8, 8]
        
        # 192 Channels
        x9 = self.dec2(x8) # [64, 16, 16]
        x9 = x9 + layers[6]
        x9 = torch.cat([x9, x1], dim=1) # Concat [64, 16, 16] && [32, 16, 16]
        # 96 Channels
        x10 = self.dec3(x9) # [32, 32, 32]
        x10 = x10 + layers[7]
        
        out = self.dec4(x10) # [16, 32, 32]
        
        return out