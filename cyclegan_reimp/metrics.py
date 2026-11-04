
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from scipy import linalg
from PIL import Image
import os


class InceptionFeatures(nn.Module):
    

    def __init__(self, device="cpu"):
        super().__init__()
        net = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        net.fc = nn.Identity()
        net.eval()
        self.net = net.to(device)
        self.device = device
        self.tf = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    @torch.no_grad()
    def features_for_folder(self, folder):
        feats = []
        files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for f in files:
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            x = self.tf(img).unsqueeze(0).to(self.device)
            feats.append(self.net(x).cpu().numpy().reshape(-1))
        return np.stack(feats, axis=0)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    try:
        # Older scipy (<1.12): disp=False returns (sqrtm_result, error_estimate)
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    except TypeError:
        # Newer scipy: 'disp' kwarg was removed; sqrtm returns just the matrix
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def compute_fid(real_dir, fake_dir, device="cpu"):
    extractor = InceptionFeatures(device)
    feats_real = extractor.features_for_folder(real_dir)
    feats_fake = extractor.features_for_folder(fake_dir)

    mu_r, sigma_r = feats_real.mean(0), np.cov(feats_real, rowvar=False)
    mu_f, sigma_f = feats_fake.mean(0), np.cov(feats_fake, rowvar=False)
    return frechet_distance(mu_r, sigma_r, mu_f, sigma_f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir", required=True)
    p.add_argument("--fake_dir", required=True)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()
    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
    fid = compute_fid(args.real_dir, args.fake_dir, device)
    print(f"FID({args.real_dir} , {args.fake_dir}) = {fid:.3f}")
