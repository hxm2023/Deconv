import sys, numpy as np
sys.path.insert(0, "src")
from data import load_images, degrade
from baselines import rl, tv, wiener_deconv
from skimage.metrics import peak_signal_noise_ratio as psnr
imgs = load_images()
rng = np.random.default_rng(0)
for name, x in imgs.items():
    y = degrade(x, sigma=2.0, photons=300, rng=rng)
    print(name, "noisy", round(psnr(x, y, data_range=1), 2),
          "RL", round(psnr(x, rl(y), data_range=1), 2),
          "TV", round(psnr(x, tv(y), data_range=1), 2),
          "Wiener", round(psnr(x, wiener_deconv(y), data_range=1), 2))
