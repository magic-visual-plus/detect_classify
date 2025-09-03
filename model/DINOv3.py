import os
import torch
import torch.nn as nn
import dinov3

class DinoV3Classifier(nn.Module):
    def __init__(self, backbone_name, backbone_weights, num_classes, head_embed_dim=100, freeze_backbone=True):
        super().__init__()
        self.backbone = self.load_backbone(backbone_name, backbone_weights)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.embed_dim * 2, head_embed_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(head_embed_dim, num_classes)
        )
        if freeze_backbone:
            self.freeze_backbone()
        
    def load_backbone(self, backbone_name, weights):
        REPO_DIR = os.path.dirname(os.path.dirname(dinov3.__file__))
        backbone = torch.hub.load(REPO_DIR, backbone_name, source="local", weights=weights)
        return backbone
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.backbone.forward_features(x)
        cls_token = x["x_norm_clstoken"]
        patch_tokens = x["x_norm_patchtokens"]
        linear_input = torch.cat(
            [
                cls_token,
                patch_tokens.mean(dim=1),
            ],
            dim=1,
        )
        logits = self.head(linear_input)
        return logits


        