from scipy.linalg import sqrtm
import torch
import numpy as np

def eval_score(real_images, fake_images, inception_model):
    inception_model.eval()

    with torch.no_grad():
        real_features = inception_model(real_images).detach().cpu().numpy()
        mu1, cov1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)

        fake_features = inception_model(fake_images).detach().cpu().numpy()
        mu2, cov2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)

    # Calculate FID score
    diff = mu1 - mu2
    cov_sqrt, _ = sqrtm(cov1.dot(cov2), disp=False)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    fid_score = diff.dot(diff) + np.trace(cov1 + cov2 - 2 * cov_sqrt)
    return fid_score
    

