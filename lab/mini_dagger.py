#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mini_dagger.py — одна мини-итерация DAgger поверх mini_bc (семантика dagger_train.py:
драйвер эпизода — оракул или политика с шансом oracle-share, метка ВСЕГДА оракул).
Цель: показать на локальном стенде, что DAgger поднимает замкнутый цикл."""
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_env import SimOsuEnv
from oracle import oracle_policy, aux_target
from net import BCNet, bcnet_to_ppo

MAPS_DIR = sys.argv[1] if len(sys.argv) > 1 else "sim_maps"
RESUME = sys.argv[2] if len(sys.argv) > 2 else "mini_bc_bcnet.pt"
OUT = sys.argv[3] if len(sys.argv) > 3 else "mini_dagger"
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
N_STACK = 8
BATCH, LR, TRAIN_EVERY, WARMUP = 64, 3e-4, 2, 1024
BTN_W, VALUE_W, AUX_W, GAMMA = 1.5, 0.5, 1.0, 0.99
ORACLE_SHARE = 0.5          # как dagger_train iter1
ACT_NOISE_ORACLE = 0.03
LAG_P = (0.4, 0.35, 0.25)   # как dagger_train: закрывает дыру lag=2

maps = sorted(glob.glob(os.path.join(MAPS_DIR, "*.osu")))
assert maps
dev = "cpu"
net = BCNet(n_stack=N_STACK).to(dev)
net.load_state_dict(torch.load(RESUME, map_location=dev))
print("продолжаю с", RESUME, flush=True)
opt = torch.optim.Adam(net.parameters(), lr=LR)

buf_obs = np.zeros((STEPS, 120, 160, N_STACK), dtype=np.uint8)
buf_move = np.zeros(STEPS, dtype=np.int64)
buf_btn = np.zeros(STEPS, dtype=np.int64)
buf_aux = np.zeros((STEPS, 2), dtype=np.float32)
buf_r = np.zeros(STEPS, dtype=np.float32)
buf_done = np.zeros(STEPS, dtype=np.float32)

env = SimOsuEnv(maps, domain_rand=True)
obs, _ = env.reset(seed=31337)
env._lag = int(env.np_random.choice(3, p=LAG_P))
env._press_off = 10.0
driver = "oracle" if env.np_random.random() < ORACLE_SHARE else "policy"
stack = np.repeat(obs, N_STACK, axis=2)
ep_acc, ep_drv = [], []

btn_w, t0, fill = None, time.time(), 0
stats = {"loss": 0.0, "m": 0, "b": 0, "n": 0}
for step in range(STEPS):
    midx, btn = oracle_policy(env)                     # метка — всегда оракул
    ax_, ay_ = aux_target(env)
    buf_obs[step] = stack
    buf_move[step] = midx
    buf_btn[step] = btn
    buf_aux[step] = (ax_, ay_)

    if driver == "oracle":
        exec_midx, exec_btn = midx, btn
        if env.np_random.random() < ACT_NOISE_ORACLE:
            exec_midx = int(env.np_random.integers(0, 27))
    else:
        with torch.no_grad():
            x = torch.from_numpy(stack).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
            lm, lb, _, _ = net(x)
        exec_midx = int(torch.multinomial(F.softmax(lm.squeeze(0), 0), 1))
        exec_btn = int(torch.multinomial(F.softmax(lb.squeeze(0), 0), 1))

    obs, r, term, trunc, info = env.step(np.array([exec_midx, exec_btn], dtype=np.int64))
    stack = np.concatenate([stack[:, :, 1:], obs], axis=2)
    buf_r[step] = r
    buf_done[step] = float(term or trunc)
    fill = step + 1
    if info:
        ep_acc.append(info.get("acc")); ep_drv.append(driver)
    if term or trunc:
        obs, _ = env.reset()
        env._lag = int(env.np_random.choice(3, p=LAG_P))
        env._press_off = 10.0
        driver = "oracle" if env.np_random.random() < ORACLE_SHARE else "policy"
        stack = np.repeat(obs, N_STACK, axis=2)

    if btn_w is None and fill >= WARMUP:
        counts = np.bincount(buf_btn[:fill], minlength=5)
        w = 1.0 / np.maximum(counts, 1).astype(np.float32)
        w /= w.mean(); w[0] = min(w[0], 0.3)
        btn_w = torch.tensor(np.clip(w, 0.05, 10.0))

    if btn_w is not None and (step + 1) % TRAIN_EVERY == 0:
        idx = np.random.randint(0, fill, size=BATCH)
        newest = fill - 1
        done_np = buf_done[idx].copy(); done_np[idx == newest] = 1.0
        nxt = (idx + 1) % fill
        x = torch.from_numpy(buf_obs[idx]).permute(0, 3, 1, 2).float() / 255.0
        x2 = torch.from_numpy(buf_obs[nxt]).permute(0, 3, 1, 2).float() / 255.0
        mv = torch.from_numpy(buf_move[idx]); bt = torch.from_numpy(buf_btn[idx])
        at = torch.from_numpy(buf_aux[idx])
        with torch.no_grad():
            _, _, v_next, _ = net(x2)
            target = torch.from_numpy(buf_r[idx]) + GAMMA * (1.0 - torch.from_numpy(done_np)) * v_next
        lm, lb, lv, la = net(x)
        loss = (F.cross_entropy(lm, mv)
                + BTN_W * F.cross_entropy(lb, bt, weight=btn_w)
                + VALUE_W * F.smooth_l1_loss(lv, target)
                + AUX_W * F.mse_loss(la, at))
        opt.zero_grad(); loss.backward(); opt.step()
        stats["loss"] += loss.item()
        stats["m"] += (lm.argmax(1) == mv).sum().item()
        stats["b"] += (lb.argmax(1) == bt).sum().item()
        stats["n"] += BATCH

    if (step + 1) % 1000 == 0:
        k = max(1, stats["n"]); nb = max(1, stats["n"] // BATCH)
        pa = ([a for a, d in zip(ep_acc, ep_drv) if d == "policy"] or [0])
        oa = ([a for a, d in zip(ep_acc, ep_drv) if d == "oracle"] or [0])
        print(f"{step+1:>6,}/{STEPS:,} | loss={stats['loss']/nb:.3f} | agree(буфер)={stats['m']/k:.3f} "
              f"| ep_acc: oracle={np.mean(oa):.3f} policy={np.mean(pa):.3f} "
              f"| {(step+1)/(time.time()-t0):.0f} ш/с", flush=True)
        stats = {"loss": 0.0, "m": 0, "b": 0, "n": 0}

torch.save(net.state_dict(), OUT + "_bcnet.pt")
bcnet_to_ppo(net, maps, N_STACK, dev, OUT)
print("сохранено:", OUT + ".zip", flush=True)
