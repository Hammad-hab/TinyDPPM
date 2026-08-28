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
        self.cblocks = [
            32,    # enc1
            64,    # enc2
            128,   # enc3
            256,   # enc4
            256,   # enc5
            256,   # enc6
            256,   # bt1
            256,   # bt2
            128,   # dec1
            128,   # dec2
            128,   # dec3
            128,   # dec4
            64,    # dec5
            32,    # dec6
        ]
        self.proj_layers = nn.ModuleList([nn.Linear(self.D, c) for c in self.cblocks])

        self.enc1 = self._encoder_block(3, 32) # Out:128
        self.enc2 = self._encoder_block(32, 64) # Out:64
        self.enc3 = self._encoder_block(64, 128) # Out: 32
        self.enc4 = self._encoder_block(128, 256) # Out: 16
        self.enc5 = self._encoder_block(256, 256) # Out: 8
        self.enc6 = self._encoder_block(256, 256) # Out: 4
        
        self.mlp = self.mlp_stack(self.D, self.D*2)
        
        self.bt1 = nn.Conv2d(256, 256, 3, padding=1)
        self.bt2 = nn.Conv2d(256, 256, 3, padding=1)

        
        self.dec1 = self._decoder_block(256, 128)
        self.dec2 = self._decoder_block(384, 128)
        self.dec3 = self._decoder_block(384, 128)
        self.dec4 = self._decoder_block(256, 128)
        self.dec5 = self._decoder_block(192, 64)
        self.dec6 = self._decoder_block(96, 32)
        
        self.dec7 = nn.Sequential(
            nn.Conv2d(32, 3, 3, padding=1),
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

        x4 = self.enc4(x3) # [128, 4, 4]
        x4 = x4 + layers[3]

        x5 = self.enc5(x4) # [128, 4, 4]
        x5 = x5 + layers[4]

        x6 = self.enc6(x5) # [128, 4, 4]
        x6 = x6 + layers[5]

        # Bottleneck
        x7 = self.bt1(x6) # [256, 4, 4]
        x7 = x7 + layers[6]
        
        x8 = F.relu(x7)
        
        x9 = self.bt2(x8) # [256, 4, 4]
        x9 = x9 + layers[7]
        
        x10 = F.relu(x9)
        # Decoding
        # 
        x11 = self.dec1(x10) # [128, 8, 8]
        x11 = x11 + layers[8]
        x11 = torch.cat([x11, x5], dim=1) # Concat [128, 8, 8] && [64, 8, 8]
        
        x12 = self.dec2(x11) # [64, 16, 16]
        x12 = x12 + layers[9]
        x12 = torch.cat([x12, x4], dim=1) # Concat [64, 16, 16] && [32, 16, 16]

        x13 = self.dec3(x12) # [64, 16, 16]
        x13 = x13 + layers[10]
        x13 = torch.cat([x13, x3], dim=1) # Concat [64, 16, 16] && [32, 16, 16]

        x14 = self.dec4(x13) # [64, 16, 16]
        x14 = x14 + layers[11]
        x14 = torch.cat([x14, x2], dim=1) # Concat [64, 16, 16] && [32, 16, 16]  

        x15 = self.dec5(x14) # [64, 16, 16]
        x15 = x15 + layers[12]
        x15 = torch.cat([x15, x1], dim=1) # Concat [64, 16, 16] && [32, 16, 16]

        x16 = self.dec6(x15) # [64, 16, 16]
        x16 = x16 + layers[13]
        
        out = self.dec7(x16)
        return out

if __name__ == "__main__":
    model = UNET()
    
    x = torch.randn(2, 3, 256, 256)
    t = torch.randint(0, 1000, (2,))
    
    y = model(x, t)
    
    print(x.shape) # [2, 3, 256]
    print(y.shape) # [2, 3, 256]