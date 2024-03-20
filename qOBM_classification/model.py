import torch
import torch.nn as nn

class DinoV2net(nn.Module):
    def __init__(self, num_classes, dinov2_cache_dir="/storage/home/hhive1/rchen438/data/Dental_CBCT_dinov2/cache"):
        super(DinoV2net, self).__init__()
        
        torch.hub.set_dir(dinov2_cache_dir)
        dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        #for param in dinov2_vitb14.parameters():
        #    param.requires_grad = False
            
        self.dinov2 = dinov2_vitb14
        self.linear = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2 * dinov2_vitb14.embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes),
        )
        
    def forward(self, x):
        x = self.dinov2.forward_features(x)
        cls_token = x["x_norm_clstoken"]
        patch_tokens = x["x_norm_patchtokens"]
        linear_input = torch.cat([
            cls_token,
            patch_tokens.mean(dim=1),
        ], dim=1)
        
        out = self.linear(linear_input)
        return out