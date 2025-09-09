import os
import torch
import torch.nn as nn
import dinov3
import utils
import utils.dataset
from typing import List

class DinoV3Classifier(nn.Module):
    def __init__(self, backbone_name, num_classes, backbone_weights=None, check_point_path=None, REPO_DIR=None, freeze_backbone=True):
        super().__init__()
        self.num_classes = num_classes
        if backbone_weights:
            self.backbone = self.load_backbone(backbone_name, backbone_weights, REPO_DIR=REPO_DIR)
        else:
            self.backbone = self.load_backbone(backbone_name, REPO_DIR=REPO_DIR)
        if num_classes == 2:
            self.head = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(self.backbone.embed_dim * 2, 1)
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(self.backbone.embed_dim * 2, num_classes)
            )
        if freeze_backbone:
            self.freeze_backbone()

        if check_point_path:
            checkpoint = torch.load(check_point_path, map_location="cpu")
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("model."):
                        new_key = k[len("model."):]
                    else:
                        new_key = k
                    new_state_dict[new_key] = v
                state_dict = new_state_dict
            else:
                state_dict = checkpoint

            self.load_state_dict(state_dict)
            print(f"Successfully loaded model weights from: {check_point_path}")

        
    def load_backbone(self, backbone_name, weights=None, REPO_DIR=None):
        # REPO_DIR = os.path.dirname(os.path.dirname(dinov3.__file__))
        if REPO_DIR is None:
            REPO_DIR = "/root/dinov3"
        if weights is None:
            backbone = torch.hub.load(REPO_DIR, backbone_name, source="local", pretrained=False)
        else:
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

    def predict(self, img, bboxs, scale_factor=1.0, pad_mode="constant", pad_color=(114, 114, 114), transformer=None) -> List[dict]:
        results = []
        cropped_images = []
        for bbox in bboxs:
            x, y, x2, y2 = bbox
            w = x2 - x
            h = y2 - y
            cropped_image = utils.dataset.crop_adaptive_square(img, x, y, w, h, scale_factor, pad_mode, pad_color)
            if transformer is not None:
                cropped_image = transformer(cropped_image)
            cropped_images.append(cropped_image)
        if isinstance(cropped_images[0], torch.Tensor):
            batch = torch.stack(cropped_images, dim=0)
        else:
            batch = torch.stack([torch.from_numpy(np.array(img)).permute(2,0,1) if not isinstance(img, torch.Tensor) else img for img in cropped_images], dim=0)
        
        device = next(self.parameters()).device
        batch = batch.to(device)
        
        logits = self.forward(batch)

        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            pred_classes = (probs > 0.5).long().squeeze(1)
            scores = torch.where(pred_classes == 1, probs.squeeze(1), 1 - probs.squeeze(1))
        else:
            probs = torch.softmax(logits, dim=1)
            pred_classes = torch.argmax(probs, dim=1)
            scores = probs[range(len(pred_classes)), pred_classes]
        
        for i, bbox in enumerate(bboxs):
            results.append({
                'bbox': bbox,
                'pred_class': pred_classes[i].item(),
                'score': scores[i].item()
            })
        return results


        