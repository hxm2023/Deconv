import sys, json, numpy as np, torch
sys.path.insert(0, "src")
from data import load_images, degrade
from baselines import rl, tv, wiener_deconv
from model import UNet
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from scipy.stats import wilcoxon

dev = "cuda" if torch.cuda.is_available() else "cpu"
imgs = load_images(128)
net = UNet(32).to(dev); net.load_state_dict(torch.load("checkpoints/deconv_gpu.pt")); net.eval()
rng = np.random.default_rng(123)

# tune TV / Wiener on validation bottom halves (coins, brick)
val = ["coins", "brick"]
def val_score(fn, weights):
    best = (-1, None)
    for w in weights:
        vals = []
        for name in val:
            for k in range(4):
                y = degrade(imgs[name][64:128], 2.0, 300, rng)
                p = fn(y, w)
                vals.append(psnr(imgs[name][64:128], np.clip(p,0,1), data_range=1))
        m = float(np.mean(vals))
        if m > best[0]: best = (m, w)
    return best[1]
w_tv = val_score(lambda y,w: tv(y,w), [0.01,0.02,0.04,0.08,0.12,0.16,0.2])
def wiener_w(y,w): return wiener_deconv(y, balance=w)
w_wi = val_score(wiener_w, [0.002,0.005,0.01,0.02,0.05,0.1])
print("tuned tv", w_tv, "wiener", w_wi, flush=True)

test = ["cell", "retina", "ihc", "camera"]
raw = {}
for name in test:
    gt_full = imgs[name][64:128]
    rows = {k: [] for k in ["noisy","tv","wiener","rl","unet"]}
    # 4 64x64 test windows (stride 32 over 64x128 bottom? use 64x64 quadrants)
    wins = [(0,0),(0,64),(32,0),(32,64)] if gt_full.shape[1] >= 128 else [(0,0)]
    img = imgs[name]
    for (i,j) in [(64,0),(64,64)]:
        gt = img[i:i+64, j:j+64]
        y = degrade(gt, 2.0, 300, rng)
        rows["noisy"].append(psnr(gt, y, data_range=1))
        rows["tv"].append(psnr(gt, np.clip(tv(y,w_tv),0,1), data_range=1))
        rows["wiener"].append(psnr(gt, np.clip(wiener_w(y,w_wi),0,1), data_range=1))
        rows["rl"].append(psnr(gt, np.clip(rl(y),0,1), data_range=1))
        with torch.no_grad():
            p = net(torch.as_tensor(y, dtype=torch.float32)[None,None].to(dev)).cpu().numpy()[0,0]
        rows["unet"].append(psnr(gt, np.clip(p,0,1), data_range=1))
    raw[name] = rows
    print(name, {k: round(float(np.mean(v)),2) for k,v in rows.items()}, flush=True)

# aggregate patch-level paired stats
u = np.array([x for n in test for x in raw[n]["unet"]])
t = np.array([x for n in test for x in raw[n]["tv"]])
w = np.array([x for n in test for x in raw[n]["wiener"]])
print("unet-tv", round(float(np.mean(u-t)),2), "wilcoxon", wilcoxon(u,t).pvalue)
print("unet-wiener", round(float(np.mean(u-w)),2), "wilcoxon", wilcoxon(u,w).pvalue)
json.dump(dict(rows=raw, tv_weight=w_tv, wiener_balance=w_wi,
               unet_tv=float(np.mean(u-t)), unet_wiener=float(np.mean(u-w)),
               p_tv=float(wilcoxon(u,t).pvalue), p_wi=float(wilcoxon(u,w).pvalue)),
          open("results/benchmark.json","w"), indent=2)
