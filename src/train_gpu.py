import sys, numpy as np, torch
sys.path.insert(0, "src")
from data import load_images, degrade
from model import UNet
from baselines import rl, tv, wiener_deconv
from skimage.metrics import peak_signal_noise_ratio as psnr

dev = "cuda" if torch.cuda.is_available() else "cpu"
imgs = load_images(128)
rng = np.random.default_rng(0)
train_names = list(imgs)
X, Y = [], []
for k in range(6000):
    name = train_names[k % len(train_names)]
    img = imgs[name]
    i = rng.integers(0, 64); j = rng.integers(0, 64)
    y = img[i:i+64, j:j+64]
    x = degrade(y, 2.0, 300, rng)
    X.append(x[None]); Y.append(y[None])
X = torch.as_tensor(np.asarray(X), dtype=torch.float32).to(dev)
Y = torch.as_tensor(np.asarray(Y), dtype=torch.float32).to(dev)
net = UNet(32).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
for ep in range(40):
    idx = torch.randperm(len(X), device=dev)
    for q in range(0, len(X), 256):
        mb = idx[q:q+256]
        pred = net(X[mb])
        loss = ((pred - Y[mb])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    print("epoch", ep, "loss", round(float(loss), 5), flush=True)
torch.save(net.state_dict(), "checkpoints/deconv_gpu.pt")
for name in ["cell", "retina", "ihc", "camera"]:
    img = imgs[name]
    y = degrade(img[64:128], 2.0, 300, rng)
    with torch.no_grad():
        p = net(torch.as_tensor(y, dtype=torch.float32)[None, None].to(dev)).cpu().numpy()[0,0]
    print(name, "noisy", round(psnr(img[64:128], y, data_range=1),2),
          "TV", round(psnr(img[64:128], tv(y), data_range=1),2),
          "Wiener", round(psnr(img[64:128], wiener_deconv(y), data_range=1),2),
          "U-Net", round(psnr(img[64:128], np.clip(p,0,1), data_range=1),2), flush=True)
