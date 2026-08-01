import json, time, numpy as np, BranchCutCathedral as bcc
from pathlib import Path

cfg = bcc.Config(n_data=100_000, batch_size=4096, epochs=40, hidden=192,
                 blocks=3, mixed_features=64, out_dir="./run")
out = Path(cfg.out_dir); out.mkdir(exist_ok=True)
np.random.seed(cfg.seed); bcc.xp.random.seed(cfg.seed)

X, Y, W, labels = bcc.generate_dataset(cfg)
w_audit = bcc.w_distribution_summary(bcc.to_numpy(W), bcc.to_numpy(labels), cfg)
net = bcc.BranchCutNet(cfg); opt = bcc.Adam(net.params(), lr=cfg.lr)
lab = bcc.to_numpy(labels); pos = float((lab >= .999).mean())
pw = float(np.clip((1-pos)/max(pos,1e-6), 1, 20))
print(f"params {net.parameter_count():,} | interior weight {pw:.2f}", flush=True)

n = len(lab); t0 = time.time(); losses=[]
for ep in range(1, cfg.epochs+1):
    opt.lr = bcc.cosine_lr(ep, cfg.epochs, cfg)
    p = bcc.xp.asarray(np.random.permutation(n)); X,Y,W,labels = X[p],Y[p],W[p],labels[p]
    tot=0.0; nb=0
    for s in range(0, n, cfg.batch_size):
        e=min(s+cfg.batch_size,n); bs=e-s
        x,y,w,tg = X[s:e],Y[s:e],W[s:e],labels[s:e]
        pt,pi = net.fwd(x,y,w)
        it = (tg >= .999).astype(np.float32)
        be = 4.0*np.exp(-((tg-.24)/.22)**2)*(1-it)
        tw = 1.0 + 5.0*np.sqrt(tg) + be + 3.0*it
        d = pt-tg
        tl = (tw*d*d).mean()
        pr = pi.clip(1e-5,1-1e-5)
        cw = 1.0 + it*(pw-1.0)
        cl = -(cw*(it*np.log(pr)+(1-it)*np.log(1-pr))).mean()
        tot += float(tl + cfg.bce_weight*cl); nb+=1
        net.bwd(2*tw*d/bs, cfg.bce_weight*cw*(pr-it)/(pr*(1-pr)+1e-6)/bs)  # all device-side
        opt.step()
    losses.append(tot/nb)
    if ep % 5 == 0 or ep == 1:
        print(f"[{ep:3d}/{cfg.epochs}] loss={tot/nb:.6f} lr={opt.lr:.2e} {time.time()-t0:.0f}s", flush=True)

bcc.save_checkpoint(net, opt, cfg.epochs, losses, cfg, out/"model.npz",
                    w_audit=w_audit)
print(f"DONE {time.time()-t0:.0f}s -> {out}/model.npz", flush=True)
