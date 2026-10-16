import glob
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from sim_env import SimOsuEnv, DT_MS
from env import DIRS, MOVE_DISTANCES, ACTIONS_PER_SECOND
from vision import OSU_W, OSU_H

try:                      # монитор необязателен: без него обучение не меняется
    import ai_telemetry
except Exception:         # noqa: BLE001
    ai_telemetry = None

# ============ НАСТРОЙКИ ============
TOTAL_STEPS = 500_000      # шагов среды = размер датасета
TRAIN_EVERY = 4            # градиентный шаг каждые N шагов среды
BATCH       = 256
LR          = 3e-4
WARMUP      = 2048         # шагов до старта обучения
BUF_SIZE    = 30_000       # кольцевой буфер
ACT_NOISE   = 0.03         # шум ТОЛЬКО на исполнение (метка чистая)
BTN_W       = 1.5          # вес лосса кнопки
VALUE_W     = 0.5          # вес критика
AUX_W       = 1.0          # AUX: вес головы «вектор к цели»
GAMMA       = 0.99
OUT = "bc_policy_fresh"
N_STACK = 4


# ============ УЧИТЕЛЬ v3 ============
# (не менялся со stage3a — офсетов НЕТ, DAgger придёт позже своим этапом)

def _predict_pending(env):
    """Позиция и кнопки на момент ИСПОЛНЕНИЯ новой команды + флаг летящего клика."""
    px, py, bl, br = env.cursor_x, env.cursor_y, env.btn_l, env.btn_r
    has_press = False
    pending = list(env._action_q)[-env._lag:] if env._lag else []
    for a in pending:
        if a is None:
            continue
        aa = np.asarray(a).flatten()
        d = DIRS[int(aa[0]) % len(DIRS)]
        sp = MOVE_DISTANCES[min(len(MOVE_DISTANCES) - 1, int(aa[0]) // len(DIRS))]
        px = min(OSU_W, max(0.0, px + d[0] * sp * env._scale))
        py = min(OSU_H, max(0.0, py + d[1] * sp * env._scale))
        b = int(aa[1])
        if b == 1:
            has_press = has_press or not bl
            bl = True
        elif b == 2:
            bl = False
        elif b == 3:
            has_press = has_press or not br
            br = True
        elif b == 4:
            br = False
    return px, py, bl, br, has_press


def _best_move(px, py, tx, ty, scale):
    best, bd = 4, math.hypot(tx - px, ty - py)
    for midx in range(27):
        d = DIRS[midx % len(DIRS)]
        sp = MOVE_DISTANCES[min(len(MOVE_DISTANCES) - 1, midx // len(DIRS))]
        nx = min(OSU_W, max(0.0, px + d[0] * sp * scale))
        ny = min(OSU_H, max(0.0, py + d[1] * sp * scale))
        dist = math.hypot(tx - nx, ty - ny)
        if dist < bd:
            bd, best = dist, midx
    return best, bd


def oracle_policy(env):
    """Учитель v3."""
    t = env.sim_time
    L, m = env._lag, env.cur
    px, py, bl, br, has_press = _predict_pending(env)

    t_obj_i, hold_i = None, None
    for i in range(env._expire_ptr, min(len(env.hitobjects), env._expire_ptr + 8)):
        o, st = env.hitobjects[i], env.obj_state[i]
        if o["type"] == 0 and not st["judged"]:
            if t_obj_i is None: t_obj_i = i
        elif o["type"] == 1:
            if st["head"] is None:
                if t_obj_i is None: t_obj_i = i
            elif (st["head"] in ("300", "100", "50")
                  and not st["final"]
                  and env.sim_time < o["time"]):
                if hold_i is None: hold_i = i
    t_obj = env.hitobjects[t_obj_i] if t_obj_i is not None else None

    if env._open_sliders:
        o = env.hitobjects[env._open_sliders[-1]]
        tx, ty = env._slider_pos(o, t + (L + 1) * DT_MS)
    elif hold_i is not None:
        o = env.hitobjects[hold_i]
        tx, ty = float(o["x"]), float(o["y"])
    elif env._open_spinners:
        a = math.atan2(py - 192.0, px - 256.0) + 1.5
        tx, ty = 256.0 + 45.0 * math.cos(a), 192.0 + 45.0 * math.sin(a)
    elif t_obj is not None:
        tx, ty = float(t_obj["x"]), float(t_obj["y"])
    else:
        g = env._goal()
        tx, ty = g if g is not None else (px, py)

    midx, _ = _best_move(px, py, tx, ty, 1.0)
    d = DIRS[midx % len(DIRS)]
    sp = MOVE_DISTANCES[min(len(MOVE_DISTANCES) - 1, midx // len(DIRS))]
    ax = min(OSU_W, max(0.0, px + d[0] * sp * 1.0))
    ay = min(OSU_H, max(0.0, py + d[1] * sp * 1.0))

    on_obj = (t_obj is not None and
              math.hypot(t_obj["x"] - ax, t_obj["y"] - ay) <= m["r_osu"])
    in_window = (t_obj is not None and -30 < t - t_obj["time"] < 22)
    want_click = on_obj and in_window and not has_press
    pending_slider_hold = (has_press and t_obj is not None and t_obj["type"] == 1)
    want_hold = bool(env._open_sliders or env._open_spinners
                     or hold_i is not None or pending_slider_hold)
    if want_click:
        if not bl:   btn = 1
        elif not br: btn = 3
        else:        btn = 2
    elif want_hold:
        btn = 1
    else:
        btn = 2

    return midx, btn


# AUX: цель оракула переиспользуем как aux-таргет, вычислим отдельно
def aux_target(env):
    """Нормированный вектор (курсор -> текущая цель), координаты /512 и /384.
    Сигнал из env = учитель для aux-головы, НЕ вход сети. В инференс не идёт."""
    g = env._goal()
    if g is None:
        return 0.0, 0.0
    dx = (g[0] - env.cursor_x) / 512.0
    dy = (g[1] - env.cursor_y) / 384.0
    return dx, dy


# ============ СЕТЬ (60HZ: вход 8 каналов; AUX: доп-голова) ============
class BCNet(nn.Module):
    def __init__(self, n_stack=8):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_stack, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, n_stack, 120, 160)).shape[1]
        self.linear = nn.Linear(n, 512)
        self.move_head = nn.Linear(512, 27)
        self.btn_head = nn.Linear(512, 5)
        self.value_head = nn.Linear(512, 1)
        self.aux_head = nn.Linear(512, 2)      # AUX: [dx, dy] к цели

    def forward(self, x):
        f = F.relu(self.linear(self.cnn(x)))
        return (self.move_head(f), self.btn_head(f),
                self.value_head(f).squeeze(-1), self.aux_head(f))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    maps = sorted(glob.glob("sim_maps/*.osu")) + sorted(glob.glob("sim_maps_t1/*.osu"))
    assert maps, "нет карт в sim_maps"

    env = SimOsuEnv(maps, domain_rand=True)
    net = BCNet(n_stack=N_STACK).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    buf_obs = np.zeros((BUF_SIZE, 120, 160, N_STACK), dtype=np.uint8)
    buf_move = np.zeros(BUF_SIZE, dtype=np.int64)
    buf_btn = np.zeros(BUF_SIZE, dtype=np.int64)
    buf_aux = np.zeros((BUF_SIZE, 2), dtype=np.float32)
    buf_r = np.zeros(BUF_SIZE, dtype=np.float32)
    buf_done = np.zeros(BUF_SIZE, dtype=np.float32)
    fill, ptr = 0, 0

    obs, _ = env.reset(seed=2024)
    env._lag = int(env.np_random.choice(2, p=[0.65, 0.35]))   # 60HZ: 0/1 шаг = 0/16.7мс
    env._press_off = 10.0
    stack = np.repeat(obs, N_STACK, axis=2)

    btn_w, last_info = None, {}
    stats = {"loss": 0.0, "m": 0, "b": 0, "v": 0.0, "a": 0.0, "n": 0}
    t0 = time.time()

    for step in range(TOTAL_STEPS):
        midx, btn = oracle_policy(env)
        ax_, ay_ = aux_target(env)                      # AUX: таргет до step

        buf_obs[ptr] = stack
        buf_move[ptr] = midx
        buf_btn[ptr] = btn
        buf_aux[ptr] = (ax_, ay_)

        exec_midx = midx
        if env.np_random.random() < ACT_NOISE:
            exec_midx = int(env.np_random.integers(0, 27))
        obs, r, term, trunc, info = env.step(
            np.array([exec_midx, btn], dtype=np.int64))
        stack = np.concatenate([stack[:, :, 1:], obs], axis=2)

        buf_r[ptr] = r
        buf_done[ptr] = float(term or trunc)
        ptr = (ptr + 1) % BUF_SIZE
        fill = min(fill + 1, BUF_SIZE)

        if info:
            last_info = info
        if term or trunc:
            obs, _ = env.reset()
            env._lag = int(env.np_random.choice(2, p=[0.65, 0.35]))  # 60HZ
            env._press_off = 10.0
            stack = np.repeat(obs, N_STACK, axis=2)

        # --- веса кнопок ---
        if (btn_w is None and fill >= WARMUP) or \
           (btn_w is not None and (step + 1) % 25_000 == 0):
            counts = np.bincount(buf_btn[:fill], minlength=5)
            counts_smooth = np.maximum(counts, 1).astype(np.float32)
            w = 1.0 / counts_smooth
            w /= w.mean()
            w[0] = min(w[0], 0.3)
            btn_w = torch.tensor(np.clip(w, 0.05, 10.0), device=dev)
            if step < WARMUP + 100:
                print("баланс кнопок (0..4):", counts)
                print("веса кнопок:", btn_w.cpu().numpy().round(2))

        # --- обучение: BC + TD-критик + AUX ---
        if btn_w is not None and (step + 1) % TRAIN_EVERY == 0:
            idx = np.random.randint(0, fill, size=BATCH)

            newest = (ptr - 1) % BUF_SIZE if fill == BUF_SIZE else fill - 1
            done_np = buf_done[idx].copy()
            done_np[idx == newest] = 1.0
            nxt = (idx + 1) % fill

            x = torch.from_numpy(buf_obs[idx]).to(dev).permute(0, 3, 1, 2).float() / 255.0
            mv = torch.from_numpy(buf_move[idx]).to(dev)
            bt = torch.from_numpy(buf_btn[idx]).to(dev)
            at = torch.from_numpy(buf_aux[idx]).to(dev)   # AUX

            with torch.no_grad():
                x2 = torch.from_numpy(buf_obs[nxt]).to(dev).permute(0, 3, 1, 2).float() / 255.0
                _, _, v_next, _ = net(x2)
                r_t = torch.from_numpy(buf_r[idx]).to(dev)
                d_t = torch.from_numpy(done_np).to(dev)
                target = r_t + GAMMA * (1.0 - d_t) * v_next

            lm, lb, lv, la = net(x)
            v_loss = F.smooth_l1_loss(lv, target)
            aux_loss = F.mse_loss(la, at)                  # AUX
            loss = (F.cross_entropy(lm, mv)
                    + BTN_W * F.cross_entropy(lb, bt, weight=btn_w)
                    + VALUE_W * v_loss
                    + AUX_W * aux_loss)
            opt.zero_grad(); loss.backward(); opt.step()

            stats["loss"] += loss.item()
            stats["m"] += (lm.argmax(1) == mv).sum().item()
            stats["b"] += (lb.argmax(1) == bt).sum().item()
            stats["v"] += v_loss.item()
            stats["a"] += aux_loss.item()
            stats["n"] += BATCH

            # heartbeat для монитора: раз в 500 шагов (дешёвая запись файла)
            if ai_telemetry is not None and (step + 1) % 500 == 0:
                nb = max(1, stats["n"] // BATCH)
                ai_telemetry.publish(
                    "bc_train", ai_status="RUNNING", training=True,
                    model=OUT, model_loaded=True, train_steps=step + 1,
                    loss=stats["loss"] / nb,
                )

        if (step + 1) % 10_000 == 0:
            k = max(1, stats["n"])
            nb = max(1, stats["n"] // BATCH)
            rate = (step + 1) / (time.time() - t0)
            print(f"{step+1:>8,}/{TOTAL_STEPS:,} | loss={stats['loss']/nb:.3f} "
                  f"| move acc={stats['m']/k:.3f} | btn acc={stats['b']/k:.3f} "
                  f"| value={stats['v']/nb:.2f} | AUX-MSE={stats['a']/nb:.4f} "
                  f"| учитель: acc={last_info.get('acc')} miss={last_info.get('miss')} "
                  f"| {rate:.0f} шаг/с | ост. ~{(TOTAL_STEPS-step-1)/rate/60:.0f} мин")
            stats = {"loss": 0.0, "m": 0, "b": 0, "v": 0.0, "a": 0.0, "n": 0}

    # ============ ПРОГРЕВ КРИТИКА (без aux — голова не участвует) ============
    print("\n=== прогрев критика (MC, actor заморожен) ===")
    WARMUP_STEPS = 150_000
    v_opt = torch.optim.Adam(net.value_head.parameters(), lr=3e-4)

    obs, _ = env.reset()
    stack = np.repeat(obs, N_STACK, axis=2)
    ep_frames, ep_rewards = [], []
    vloss_acc, vloss_n = 0.0, 0

    for wstep in range(WARMUP_STEPS):
        with torch.no_grad():
            x = torch.from_numpy(stack).to(dev).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            lm, lb, _, _ = net(x)
            midx, btn = int(lm.argmax(1).item()), int(lb.argmax(1).item())
        exec_midx = midx
        if env.np_random.random() < ACT_NOISE:
            exec_midx = int(env.np_random.integers(0, 27))
        obs, r, term, trunc, info = env.step(np.array([exec_midx, btn], dtype=np.int64))

        ep_frames.append(stack.copy())
        ep_rewards.append(float(r))
        stack = np.concatenate([stack[:, :, 1:], obs], axis=2)

        if term or trunc:
            G = 0.0
            rets = [0.0] * len(ep_rewards)
            for t in range(len(ep_rewards) - 1, -1, -1):
                G = ep_rewards[t] + GAMMA * G
                rets[t] = G
            for i in range(0, len(rets), 256):
                chunk = ep_frames[i:i + 256]
                xw = torch.from_numpy(np.stack(chunk)).to(dev).permute(0, 3, 1, 2).float() / 255.0
                tgt = torch.tensor(rets[i:i + 256], dtype=torch.float32, device=dev)
                vloss = F.smooth_l1_loss(net(xw)[2], tgt)
                v_opt.zero_grad(); vloss.backward(); v_opt.step()
                vloss_acc += vloss.item(); vloss_n += 1
            obs, _ = env.reset()
            stack = np.repeat(obs, N_STACK, axis=2)
            ep_frames, ep_rewards = [], []

        if (wstep + 1) % 50_000 == 0:
            k = max(1, vloss_n)
            print(f"  прогрев {wstep+1:,}/{WARMUP_STEPS:,} | v_loss={vloss_acc/k:.3f}")
            vloss_acc, vloss_n = 0.0, 0

    print("прогрев завершён")
    torch.save(net.state_dict(), OUT + "_bcnet.pt")   # AUX: не терять aux-голову (P1 в E1)
    # ============ перенос весов в PPO (aux_head НЕ переносится) ============
    venv = VecTransposeImage(VecFrameStack(
        DummyVecEnv([lambda: Monitor(SimOsuEnv(maps))]),
        n_stack=N_STACK, channels_order="last"))
    model = PPO("CnnPolicy", venv, n_steps=512, batch_size=512,
                learning_rate=5e-5, ent_coef=0.0, verbose=0, device=dev)
    pol = model.policy
    sd = net.state_dict()

    assert len(pol.mlp_extractor.policy_net) == 0, "net_arch не пуст"
    assert pol.action_net.weight.shape == (32, 512)
    assert pol.value_net.weight.shape == (1, 512)

    pol.features_extractor.cnn.load_state_dict(
        {k[len("cnn."):]: v for k, v in sd.items() if k.startswith("cnn.")})
    pol.features_extractor.linear.load_state_dict(
        {"0.weight": sd["linear.weight"], "0.bias": sd["linear.bias"]})
    with torch.no_grad():
        pol.action_net.weight[:27].copy_(sd["move_head.weight"].to(dev))
        pol.action_net.bias[:27].copy_(sd["move_head.bias"].to(dev))
        pol.action_net.weight[27:].copy_(sd["btn_head.weight"].to(dev))
        pol.action_net.bias[27:].copy_(sd["btn_head.bias"].to(dev))
        pol.value_net.weight.copy_(sd["value_head.weight"].to(dev))
        pol.value_net.bias.copy_(sd["value_head.bias"].to(dev))

    model.save(OUT)
    print(f"\nсохранено: {OUT}.zip")
    if ai_telemetry is not None:
        ai_telemetry.clear("bc_train")

    # ============ самопроверка ============
    net.eval()
    m2 = PPO.load(OUT, device=dev)
    idx = np.random.randint(0, fill, size=512)
    batch_hwc = buf_obs[idx]
    with torch.no_grad():
        x = torch.from_numpy(batch_hwc).to(dev).permute(0, 3, 1, 2).float() / 255.0
        lm, lb, lv, _ = net(x)
        x_raw = torch.from_numpy(batch_hwc).to(dev).permute(0, 3, 1, 2).float()
        v_ppo = m2.policy.predict_values(x_raw).squeeze(-1)

    a2, _ = m2.predict(batch_hwc, deterministic=True)
    match_m = (a2[:, 0] == lm.argmax(1).cpu().numpy()).mean()
    match_b = (a2[:, 1] == lb.argmax(1).cpu().numpy()).mean()
    v_corr = (float(np.corrcoef(v_ppo.cpu().numpy(), lv.cpu().numpy())[0, 1])
              if lv.std() > 1e-8 and v_ppo.std() > 1e-8 else 0.0)

    print(f"самопроверка переноса: move {match_m:.3f} | btn {match_b:.3f} "
          f"| value corr {v_corr:.3f} (должно быть 1.000)")
    if match_m < 0.99 or match_b < 0.99 or v_corr < 0.99:
        print("!! перенос сломан — PPO НЕ запускать, кинь мне этот вывод")
    else:
        print("перенос чист. дальше: train_stage.py (стек 8!) → PPO → servo → DAgger")


if __name__ == "__main__":
    main()