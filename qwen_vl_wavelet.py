import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward

class WaveletVisionBranch(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        # J=1 means 1 level of decomposition: LL, (LH, HL, HH)
        self.dwt = DWTForward(J=1, mode='zero', wave='db1')
        
        # We want to extract the High-Frequency edges (HH band)
        # Assuming the DWT outputs feature maps that are smaller, 
        # we project them into the LLM embedding space.
        self.projection = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((16, 16)), # Flatten to sequence length 256
            nn.Flatten(2) 
        )

    def forward(self, high_res_image_tensor):
        """
        high_res_image_tensor: [B, C, 1024, 1024]
        Returns Sequence of Edge Tokens: [B, 256, embed_dim]
        """
        # yl is Low Frequency, yh is High Frequency tuples
        yl, yh = self.dwt(high_res_image_tensor)
        
        high_freq_bands = yh[0]
        
        # Extract the HH band (index 2) which captures the finest diagonal edges
        hh_band = high_freq_bands[:, :, 2, :, :]
        
        # Project into Embed Dimension
        edge_tokens = self.projection(hh_band) # [B, embed_dim, N]
        return edge_tokens.transpose(1, 2) # [B, N, embed_dim]

class QwenVLWithWavelets(nn.Module):
    def __init__(self, qwen_base_model):
        super().__init__()
        self.qwen = qwen_base_model
        self.wavelet_branch = WaveletVisionBranch(embed_dim=self.qwen.config.hidden_size)
        
    def forward(self, input_ids, images, labels=None):
        
        base_hidden_states = self.qwen(input_ids=input_ids, images=images, output_hidden_states=True).hidden_states[-1]

        edge_tokens = self.wavelet_branch(images)

        fused_states = torch.cat([base_hidden_states, edge_tokens], dim=1)
        
        return fused_states
