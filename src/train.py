import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "src")
from data import load_images, degrade
from model import UNet

imgs = load_images()
train_names = ["coins", "brick", "camera", "ihc"]
test_names = ["cell", "retina"]
rng = np.random.default_rng(0)
def crop(x, n=64):
    i = rng.integers(0, x.shape[0]-n); j = rng.integers(0, x.shape[1]-n)
    return x[i:i+n, j:j+n]
X, Y = [], []
for k in range(1200):
    name = train_names[k % len(train_names)]
    y = crop(imgs[name]); x = degrade(y, 2.0, 300, rng)
    X.append(x[None]); Y.append(y[None])
X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
Y = torch.as_tensor(np.asarray(Y), dtype=torch.float32)
net = UNet(16)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
for ep in range(12):
    idx = torch.randperm(len(X))
    for q in range(0, len(X), 64):
        mb = idx[q:q+64]
        pred = net(X[mb])
        loss = ((pred - Y[mb])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    print("epoch", ep, "loss", round(float(loss), 5), flush=True)
torch.save(net.state_dict(), "checkpoints/deconv.pt")
# eval full images
from baselines import rl, tv, wiener_deconv
from skimage.metrics import peak_signal_noise_ratio as psnr
for name in test_names:
    y = degrade(imgs[name], 2.0, 300, rng)
    with torch.no_grad():
        p = net(torch.as_tensor(y, dtype=torch.float32)[None, None]).numpy()[0,0]
    print(name, "noisy", round(psnr(imgs[name], y, data_range=1),2),
          "TV", round(psnr(imgs[name], tv(y), data_range=1),2),
          "Wiener", round(psnr(imgs[name], wiener_deconv(y), data_range=1),2),
          "U-Net", round(psnr(imgs[name], np.clip(p,0,1), data_range=1),2), flush=True)
