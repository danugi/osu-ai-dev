# bc_train3.py — BC с нуля на входе v3: координатные каналы + кнопки + v4-учитель.
#
# ПОЧЕМУ (сводка улик):
#   - 5 прогонов (BC, А, Б, В) упираются в cl nav 66-68px при живом tf 3-7px;
#   - far_probe (v1.1): направление «курсор -> цель» из пикселей НЕ извлекается
#     (bin ~случая в полосе 25-100px, aux-голова мертва под прямой супервизией);
#   - аудит master: состояние кнопок (btn_l/btn_r) в кадре нет, а метки кнопок
#     от него зависят -> неоднозначные метки; учитель v3 никогда не выдаёт
#     btn=4 (правая не отпускается) и btn=0.
# Лечение: дать сети явные входы (coords.py: курсор/цель/кнопки, режим full =
# стек 12) + учитель v4 (oracle_v4.py) + телепорты в данных + единый лаг.
#
# ЧТО ИЗМЕНЕНО против bc_train.py (остальное — семантика в семантику):
#   1) стек: 8 игровых кадров + каналы [cursor, goal, btn_l, btn_r] (--coords);
#   2) учитель: oracle_v4 (кнопочная машина состояний; MOVE-метки те же);
#   3) телепорты В ДАННЫХ: --teleport-p (толчок 30-90px, как stage2c) +
#      --goal-tp-p (прыжок в кольцо 0-35px у цели, как в попытке В);
#   4) лаг 0/1/2 = 0.40/0.35/0.25 (--lag-p): единый стандарт новой линии
#      (попытка В показала: lag=2 в данных не портит teacher-forced);
#   5) самопроверка переноса в PPO — на РЕАЛЬНЫХ состояниях буфера
#      (dagger_train2 этого не делал — поправка аудита);
#   6) --maps (у bc_train зашито sim_maps,sim_maps_t1).
#
# Запуск (боевой, ~1.5-2ч на RTX-классе GPU):
#   python bc_train3.py --out bc3_policy
# A/B-контроль (старый вход, новый учитель):
#   python bc_train3.py --coords none --out bc3_ctrl
# Тест конвейера: python bc_train3.py --steps 50000 --critic-warmup 0 --out bc3_test
#
# После обучения (все инструменты кита понимают _bcnet.pt напрямую):
#   python far_probe.py bc3_policy_bcnet.pt --lag-p 0.4,0.35,0.25
#        ожидание: far bin >= 0.7, угол(move~метка) <= 30г, aux <= 25г
#   python dagger_gap.py bc3_policy_bcnet.pt --lag-p 0.4,0.35,0.25
#        ожидание: cl nav med <= 25px (было 66-68 у всей старой линии)
#   python diagnose_v3.py bc3_policy_bcnet.pt sim_maps 0 30 det rand --lag-p 0.4,0.35,0.25
import argparse
import glob
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from sim_env import SimOsuEnv
from net import BCNet
from coords import MODES, n_stack_for, push as cpush, stack0 as cstack0, bcnet_to_ppo_wrap
from oracle_v4 import oracle_policy, aux_target
from vision import OSU_W, OSU_H

# ============ НАСТРОЙКИ (как bc_train.py) ============
TOTAL_STEPS = 500_000
TRAIN_EVERY = 4
BATCH       = 256
LR          = 3e-4
BTN_INIT    = 2048         # шагов до расчёта весов кнопок (потом обучение)
BUF_SIZE    = 30_000
ACT_NOISE   = 0.03
BTN_W       = 1.5
VALUE_W     = 0.5
AUX_W       = 1.0
GAMMA       = 0.99
CRITIC_WARMUP = 150_000    # прогрев критика в конце (0 = выключить)


def main():
    ap = argparse.ArgumentParser(
        description="BC v3: coords-вход (cursor/goal/btn) + oracle_v4 + телепорты")
    ap.add_argument("--steps", type=int, default=TOTAL_STEPS,
                    help="шагов фазы 1; дефолт %(default)s, тест конвейера: 50000")
    ap.add_argument("--critic-warmup", type=int, default=CRITIC_WARMUP,
                    help="шагов прогрева критика в конце; тест: 0")
    ap.add_argument("--out", default="bc3_policy",
                    help="имя весов без расширения (НЕ затирай bc_policy_fresh)")
    ap.add_argument("--coords", choices=list(MODES), default="full",
                    help="none=8 каналов (контроль A/B), full=12 (+cursor/goal/btn), "
                         "fullline=13 (+линия); dots/dotsline — без кнопок")
    ap.add_argument("--maps", default="sim_maps,sim_maps_t1",
                    help="папки с картами через запятую")
    ap.add_argument("--teleport-p", type=float, default=0.05,
                    help="вероятность телепорта-толчка курсора на шаг (30-90px)")
    ap.add_argument("--goal-tp-p", type=float, default=0.10,
                    help="вероятность прыжка курсора в кольцо 0-35px у цели")
    ap.add_argument("--rand-tp-p", type=float, default=0.03,
                    help="вероятность прыжка курсора в случайную точку поля "
                        "(фабрика far/xfar-состояний; ровно то, что меряет far_probe)")
    ap.add_argument("--lag-p", default="0.4,0.35,0.25",
                    help="распределение лага 0/1/2 (единый стандарт новой линии)")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--train-every", type=int, default=TRAIN_EVERY)
    ap.add_argument("--no-buf-gpu", action="store_true",
                    help="не держать буфер кадров в VRAM")
    ap.add_argument("--no-amp", action="store_true",
                    help="без bf16/channels_last/cudnn.benchmark")
    a = ap.parse_args()

    lag_p = tuple(float(x) for x in a.lag_p.split(","))
    assert len(lag_p) == 3 and abs(sum(lag_p) - 1.0) < 1e-6, "--lag-p: три доли, сумма 1"
    if a.steps < 20_000:
        print(f"[i] тестовый масштаб: {a.steps} шагов — проверка конвейера, не качества")

    N_STACK = n_stack_for(a.coords)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    if dev != "cuda":
        print("!! GPU не найден — обучение пойдёт на CPU (в разы дольше).")
    maps = []
    for d in (x.strip() for x in a.maps.split(",")):
        maps += sorted(glob.glob(d + "/*.osu"))
    assert maps, "нет карт в " + a.maps
    print(f"[stack] N_STACK={N_STACK} (coords={a.coords}) | карт: {len(maps)} | "
          f"tp={a.teleport_p} goal-tp={a.goal_tp_p} | lag {lag_p} | "
          f"учитель: oracle_v4 (кнопки 0-4)")

    env = SimOsuEnv(maps, domain_rand=True, teleport_prob=a.teleport_p)
    net = BCNet(n_stack=N_STACK).to(dev)

    # --- план B (PLAN-SPEEDUP): bf16 + channels_last + cudnn.benchmark ---
    if a.no_amp:
        use_amp = False
    elif dev == "cuda":
        try:
            use_amp = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            use_amp = torch.cuda.get_device_capability(0)[0] >= 8
        if not use_amp:
            print("[amp] карта без аппаратного bf16 — fp32 (выключить: --no-amp).")
    else:
        use_amp = os.environ.get("OSU_AMP_CPU") == "1"
    if use_amp:
        if dev == "cuda":
            torch.backends.cudnn.benchmark = True
        net = net.to(memory_format=torch.channels_last)
        print("[amp] bf16 + channels_last"
              + (" + cudnn.benchmark" if dev == "cuda" else "")
              + " (выключить: --no-amp)")
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    buf_size = BUF_SIZE if a.steps >= BUF_SIZE else a.steps
    use_buf_gpu = False
    buf_obs_t = None
    if dev == "cuda" and not a.no_buf_gpu:
        free_b, _ = torch.cuda.mem_get_info()
        need_b = buf_size * 120 * 160 * N_STACK
        if free_b - need_b > 2 * 1024 ** 3:
            try:
                buf_obs_t = torch.empty((buf_size, 120, 160, N_STACK),
                                        dtype=torch.uint8, device=dev)
                use_buf_gpu = True
            except RuntimeError:
                buf_obs_t = None
        if use_buf_gpu:
            print(f"[buf-gpu] кадры в VRAM: {need_b / 2**30:.1f} ГБ "
                  f"(свободно было {free_b / 2**30:.1f} ГБ)")
        else:
            print(f"[buf-gpu] мало VRAM: свободно {free_b / 2**30:.1f} ГБ, "
                  f"нужно {need_b / 2**30:.1f} ГБ + 2 ГБ запас — буфер в ОЗУ")
    elif dev == "cpu" and os.environ.get("OSU_BUF_GPU_CPU") == "1":
        buf_obs_t = torch.empty((buf_size, 120, 160, N_STACK), dtype=torch.uint8)
        use_buf_gpu = True
        print("[buf-gpu] ОТЛАДКА: тензорный буфер на CPU (OSU_BUF_GPU_CPU=1)")
    buf_obs = None if use_buf_gpu else np.zeros(
        (buf_size, 120, 160, N_STACK), dtype=np.uint8)
    if not use_buf_gpu:
        print(f"[ram] кольцевой буфер ~{buf_size * 120 * 160 * N_STACK / 1e9:.1f} ГБ ОЗУ")
    buf_move = np.zeros(buf_size, dtype=np.int64)
    buf_btn = np.zeros(buf_size, dtype=np.int64)
    buf_aux = np.zeros((buf_size, 2), dtype=np.float32)
    buf_r = np.zeros(buf_size, dtype=np.float32)
    buf_done = np.zeros(buf_size, dtype=np.float32)
    fill, ptr = 0, 0

    obs, _ = env.reset(seed=2024)
    env._lag = int(env.np_random.choice(3, p=lag_p))
    env._press_off = 10.0
    stack = cstack0(obs, env, a.coords)

    btn_w, last_info = None, {}
    stats = {"loss": 0.0, "m": 0, "b": 0, "v": 0.0, "a": 0.0, "n": 0}
    btn_counts = np.zeros(5, dtype=np.int64)
    t0 = time.time()

    log_every = 10_000 if a.steps >= 50_000 else max(500, a.steps // 5)
    for step in range(a.steps):
        midx, btn = oracle_policy(env)          # v4: кнопочная машина состояний
        ax_, ay_ = aux_target(env)

        if use_buf_gpu:
            buf_obs_t[ptr] = torch.from_numpy(stack)
        else:
            buf_obs[ptr] = stack
        buf_move[ptr] = midx
        buf_btn[ptr] = btn
        buf_aux[ptr] = (ax_, ay_)
        btn_counts[btn] += 1

        exec_midx = midx
        if env.np_random.random() < ACT_NOISE:
            exec_midx = int(env.np_random.integers(0, 27))

        # целевой телепорт ПОСЛЕ записи пары (состояние, метка): прыжок виден
        # в следующем кадре/каналах, метка следующего шага = коррекция серво
        if a.rand_tp_p > 0 and env.np_random.random() < a.rand_tp_p:
            env.cursor_x = float(env.np_random.uniform(40, OSU_W - 40))
            env.cursor_y = float(env.np_random.uniform(40, OSU_H - 40))
        elif a.goal_tp_p > 0 and env.np_random.random() < a.goal_tp_p:
            g = env._goal()
            if g is not None:
                ang = env.np_random.uniform(0.0, 2.0 * math.pi)
                dist = env.np_random.uniform(0.0, 35.0)
                env.cursor_x = min(OSU_W - 20.0, max(20.0, g[0] + math.cos(ang) * dist))
                env.cursor_y = min(OSU_H - 20.0, max(20.0, g[1] + math.sin(ang) * dist))

        obs, r, term, trunc, info = env.step(
            np.array([exec_midx, btn], dtype=np.int64))
        stack = cpush(stack, obs, env, a.coords)

        buf_r[ptr] = r
        buf_done[ptr] = float(term or trunc)
        ptr = (ptr + 1) % buf_size
        fill = min(fill + 1, buf_size)

        if info:
            last_info = info
        if term or trunc:
            obs, _ = env.reset()
            env._lag = int(env.np_random.choice(3, p=lag_p))
            env._press_off = 10.0
            stack = cstack0(obs, env, a.coords)

        # --- веса кнопок (v4: классы 0 и 4 теперь непустые) ---
        if (btn_w is None and fill >= BTN_INIT) or \
           (btn_w is not None and (step + 1) % 25_000 == 0):
            counts = np.bincount(buf_btn[:fill], minlength=5)
            counts_smooth = np.maximum(counts, 1).astype(np.float32)
            w = 1.0 / counts_smooth
            w /= w.mean()
            w[0] = min(w[0], 0.3)
            btn_w = torch.tensor(np.clip(w, 0.05, 10.0), device=dev)
            if btn_w is not None and step < BTN_INIT + 100:
                print("баланс кнопок v4 (0..4):", counts)
                print("веса кнопок:", btn_w.cpu().numpy().round(2))

        # --- обучение: BC + TD-критик + AUX ---
        if btn_w is not None and (step + 1) % a.train_every == 0:
            idx = np.random.randint(0, fill, size=a.batch)

            newest = (ptr - 1) % buf_size if fill == buf_size else fill - 1
            done_np = buf_done[idx].copy()
            done_np[idx == newest] = 1.0
            nxt = (idx + 1) % fill

            if use_buf_gpu:
                ix = torch.from_numpy(idx.astype(np.int64)).to(dev)
                x = buf_obs_t[ix].permute(0, 3, 1, 2).float() / 255.0
                x2 = buf_obs_t[(ix + 1) % fill].permute(0, 3, 1, 2).float() / 255.0
            else:
                x = torch.from_numpy(buf_obs[idx]).to(dev).permute(0, 3, 1, 2).float() / 255.0
                x2 = torch.from_numpy(buf_obs[nxt]).to(dev).permute(0, 3, 1, 2).float() / 255.0
            mv = torch.from_numpy(buf_move[idx]).to(dev)
            bt = torch.from_numpy(buf_btn[idx]).to(dev)
            at = torch.from_numpy(buf_aux[idx]).to(dev)

            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=use_amp):
                _, _, v_next, _ = net(x2)
                r_t = torch.from_numpy(buf_r[idx]).to(dev)
                d_t = torch.from_numpy(done_np).to(dev)
                target = r_t + GAMMA * (1.0 - d_t) * v_next

            with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_amp):
                lm, lb, lv, la = net(x)
                v_loss = F.smooth_l1_loss(lv, target)
                aux_loss = F.mse_loss(la, at)
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
            stats["n"] += a.batch

        if (step + 1) % log_every == 0:
            k = max(1, stats["n"])
            nb = max(1, stats["n"] // a.batch)
            rate = (step + 1) / (time.time() - t0)
            print(f"{step+1:>8,}/{a.steps:,} | loss={stats['loss']/nb:.3f} "
                  f"| move acc={stats['m']/k:.3f} | btn acc={stats['b']/k:.3f} "
                  f"| value={stats['v']/nb:.2f} | AUX-MSE={stats['a']/nb:.4f} "
                  f"| учитель: acc={last_info.get('acc')} miss={last_info.get('miss')} "
                  f"| {rate:.0f} шаг/с | ост. ~{(a.steps-step-1)/max(rate,1)/60:.0f} мин")
            stats = {"loss": 0.0, "m": 0, "b": 0, "v": 0.0, "a": 0.0, "n": 0}

    print("\nбаланс кнопок v4 за прогон (0..4):", btn_counts,
          "| пустые классы:", [i for i in range(5) if btn_counts[i] == 0] or "нет")

    # ============ ПРОГРЕВ КРИТИКА (MC, actor заморожен; можно выключить) ============
    if a.critic_warmup > 0:
        print(f"\n=== прогрев критика {a.critic_warmup:,} шагов (MC, actor заморожен) ===")
        v_opt = torch.optim.Adam(net.value_head.parameters(), lr=3e-4)
        obs, _ = env.reset()
        env._lag = int(env.np_random.choice(3, p=lag_p))
        env._press_off = 10.0
        stack = cstack0(obs, env, a.coords)
        ep_frames, ep_rewards = [], []
        vloss_acc, vloss_n, wlog = 0.0, 0, max(1, a.critic_warmup // 3)
        for wstep in range(a.critic_warmup):
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
            stack = cpush(stack, obs, env, a.coords)
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
                env._lag = int(env.np_random.choice(3, p=lag_p))
                env._press_off = 10.0
                stack = cstack0(obs, env, a.coords)
                ep_frames, ep_rewards = [], []
            if (wstep + 1) % wlog == 0:
                print(f"  прогрев {wstep+1:,}/{a.critic_warmup:,} | v_loss={vloss_acc/max(1,vloss_n):.3f}")
                vloss_acc, vloss_n = 0.0, 0

    torch.save(net.state_dict(), a.out + "_bcnet.pt")   # aux-голова сохраняется
    # ============ перенос в PPO через coords-обёртку + самопроверка ============
    from stable_baselines3 import PPO
    bcnet_to_ppo_wrap(net, maps, N_STACK, dev, a.out)
    print(f"сохранено: {a.out}.zip + {a.out}_bcnet.pt")

    # самопроверка на РЕАЛЬНЫХ состояниях буфера (не только на шуме):
    net.eval()
    m2 = PPO.load(a.out, device=dev)
    idx = np.random.randint(0, fill, size=512)
    if use_buf_gpu:
        batch_hwc = buf_obs_t[torch.from_numpy(idx.astype(np.int64)).to(dev)].cpu().numpy()
    else:
        batch_hwc = buf_obs[idx]
    with torch.no_grad():
        x = torch.from_numpy(batch_hwc).to(dev).permute(0, 3, 1, 2).float() / 255.0
        lm, lb, lv, _ = net(x)
    a2, _ = m2.predict(batch_hwc, deterministic=True)
    match_m = (a2[:, 0] == lm.argmax(1).cpu().numpy()).mean()
    match_b = (a2[:, 1] == lb.argmax(1).cpu().numpy()).mean()

    # Δlogits-гейт на случайных стеках (исправленный, как verify_transfer)
    rng = np.random.default_rng(0)
    maxd = 0.0
    for _ in range(64):
        s = rng.integers(0, 256, size=(120, 160, N_STACK), dtype=np.uint8)
        xx = torch.from_numpy(s).to(dev).permute(2, 0, 1).unsqueeze(0).float()
        with torch.no_grad():
            d = m2.policy.get_distribution(xx)
            bl, bb, _, _ = net(xx / 255.0)
        maxd = max(maxd,
                   (d.distribution[0].logits - F.log_softmax(bl, -1)).abs().max().item(),
                   (d.distribution[1].logits - F.log_softmax(bb, -1)).abs().max().item())

    print(f"самопроверка переноса (реальные состояния): move {match_m:.3f} | btn {match_b:.3f} "
          f"| max |Δlogits| {maxd:.5f} (должно быть 1.000 / <1e-2)")
    if match_m < 0.99 or match_b < 0.99 or maxd >= 1e-2:
        print("!! перенос сломан — PPO НЕ запускать, кинь мне этот вывод")
    else:
        print("перенос чист. дальше: far_probe / dagger_gap / diagnose_v3 (понимают .pt)")


if __name__ == "__main__":
    main()
