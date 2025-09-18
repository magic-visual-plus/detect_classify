import os
import numpy as np
import torch
import cv2
from hq_det.models.dino import hq_dino

class DinoFeatureExtractor:
    def __init__(self, model, max_size=1536):
        """
        目标检测模型backbone特征提取
        """
        self.model = model
        self.max_size = max_size

    def create_batch_data(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_scale = 1.0
        max_hw = max(img.shape[0], img.shape[1])
        if max_hw > self.max_size:
            rate = self.max_size / max_hw
            img = cv2.resize(img, (int(img.shape[1] * rate), int(img.shape[0] * rate)))
            img_scale = rate
        batch_data = self.model.imgs_to_batch([img])
        return batch_data, img_scale

    def get_feature_map(self, img_path):
        batch_data, img_scale = self.create_batch_data(img_path)
        batch_data.update(self.model.model.data_preprocessor(batch_data, self.model.model.training))
        feats = self.model.model.extract_feat(batch_data['inputs'])
        return feats
    
    def __call__(self, img_path):
        feats = self.get_feature_map(img_path)
        return feats

    @staticmethod
    def save_feats_to_npz(img_feats, save_path):
        feat_arrays = [
            (feat.detach().cpu().numpy().astype(np.float16) if hasattr(feat, 'detach') else
             (feat.cpu().numpy().astype(np.float16) if hasattr(feat, 'cpu') else np.array(feat, dtype=np.float16)))
            for feat in img_feats
        ]
        np.savez(save_path, *feat_arrays)
    
    @staticmethod
    def load_feats_from_npz(save_path):
        feats = np.load(save_path)
        feats = [feats[k][0] for k in feats.keys()]
        return feats


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1]
    input_path = sys.argv[2]
    model = hq_dino.HQDINO(model=model_path)
    model.eval()
    model.to("cuda:0")
    extractor = DinoFeatureExtractor(model)
    feats = extractor(input_path)
    extractor.save_feats_to_npz(feats, f"{input_path}.npz")
    feats = extractor.load_feats_from_npz(f"{input_path}.npz")
    for feat in feats:
        print(feat.shape)