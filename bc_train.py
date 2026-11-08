import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from stable_baselines3 import PPO

from sim_env import SimOsuEnv
from oracle import oracle_policy, aux_target
from net import BCNet, bcnet_to_ppo

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
N_STACK = 8     # 60Гц/стек-8 — единая линия; потребители читают стек из весов (CHANGES-STAGE3.md)


def main():
    ap = argparse.ArgumentParser(
        description="BC: сбор в симе + обучение + перенос в PPO")
    ap.add_argument("--steps", type=int, default=TOTAL_STEPS,
                    help="шагов фазы 1; дефолт %(default)s, тест конвейера: 50000")
    ap.add_argument("--warmup", type=int, default=150_000,
                    help="шагов прогрева критика; дефолт %(default)s, тест: 20000")
    ap.add_argument("--out", default=OUT,
                    help="имя весов без расширения; дефолт %(default)s. Тесту — ДРУГОЕ имя, чтобы не затереть боевые веса")
    ap.add_argument("--no-buf-gpu", action="store_true",
                    help="не держать буфер кадров в VRAM (план A выключен, буфер в ОЗУ)")
    ap.add_argument("--no-amp", action="store_true",
                    help="без bf16/channels_last/cudnn.benchmark (план B выключен)")
    ap.add_argument("--timing", action="store_true",
                    help="печатать раскладку мс/шаг: env / сбор+H2D / GPU (план E)")
    a = ap.parse_args()
    if a.steps <= WARMUP:
        raise SystemExit(f"--steps={a.steps} мало: обучение стартует только после WARMUP={WARMUP} шагов")
    if a.steps < 20_000:
        print(f"[i] тестовый масштаб: {a.steps} шагов — проверка конвейера, не качества; веса НЕ для игры")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    if dev != "cuda":
        print("!! GPU не найден — обучение пойдёт на CPU (в разы дольше).")
        print("   Лечение (NVIDIA):")
        print("   pip uninstall -y torch")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu124")
    maps = sorted(glob.glob("sim_maps/*.osu")) + sorted(glob.glob("sim_maps_t1/*.osu"))
    assert maps, "нет карт в sim_maps"

    env = SimOsuEnv(maps, domain_rand=True)
    net = BCNet(n_stack=N_STACK).to(dev)
    # --- план B (PLAN-SPEEDUP): bf16 + channels_last + cudnn.benchmark ---
    # bf16 быстрый только на Ampere+ (RTX 30xx и новее). На более старых
    # картах torch молча эмулирует его кастами — скорость падает ~x2
    # (наблюдалось: 40 -> 20 шаг/с). Поэтому автодетект поддержки.
    if a.no_amp:
        use_amp = False
    elif dev == "cuda":
        # 3.2.2: в новых torch is_bf16_supported() по умолчанию считает
        # ЭМУЛЯЦИЮ за поддержку (including_emulation=True): T4 и прочий
        # pre-Ampere умеют хранить bf16-тензоры, автокаст идёт медленными
        # кастами — скорость падает в разы (T4: 26 против ~60 шаг/с).
        # Спрашиваем только про железо (Ampere+/ROCm); на старых torch,
        # где kwarg ещё нет, сам вызов и так был только про железо.
        try:
            use_amp = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:  # старый torch без including_emulation
            use_amp = torch.cuda.get_device_capability(0)[0] >= 8
        if not use_amp:
            print("[amp] карта без аппаратного bf16 (нужно Ampere+: RTX 30xx или новее).")
            print("       На T4/Turing новые torch отвечают «bf16 есть», но это эмуляция")
            print("       без ускорения — план B выключен, обучение в fp32 как раньше.")
    else:
        use_amp = os.environ.get("OSU_AMP_CPU") == "1"  # отладка bf16 без CUDA
    if use_amp:
        if dev == "cuda":
            torch.backends.cudnn.benchmark = True
        net = net.to(memory_format=torch.channels_last)
        print("[amp] bf16 + channels_last" +
              (" + cudnn.benchmark" if dev == "cuda" else "") +
              " (выключить: --no-amp)")
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    # буфер не длиннее прогона: короткому тесту 30000 слотов ни к чему
    buf_size = BUF_SIZE if a.steps >= BUF_SIZE else a.steps
    # --- план A (PLAN-SPEEDUP): кадры в VRAM, если помещаются с запасом ---
    use_buf_gpu = False
    buf_obs_t = None
    if dev == "cuda" and not a.no_buf_gpu:
        free_b, _ = torch.cuda.mem_get_info()
        need_b = buf_size * 120 * 160 * N_STACK
        if free_b - need_b > 2 * 1024 ** 3:   # 2 ГБ запас: активации, cudnn, аллокатор
            try:
                buf_obs_t = torch.empty((buf_size, 120, 160, N_STACK),
                                        dtype=torch.uint8, device=dev)
                use_buf_gpu = True
            except RuntimeError:              # OOM на выделении — тихий откат в ОЗУ
                buf_obs_t = None
        if use_buf_gpu:
            print(f"[buf-gpu] кадры в VRAM: {need_b / 2**30:.1f} ГБ "
                  f"(свободно было {free_b / 2**30:.1f} ГБ)")
        else:
            print(f"[buf-gpu] мало VRAM: свободно {free_b / 2**30:.1f} ГБ, "
                  f"нужно {need_b / 2**30:.1f} ГБ + 2 ГБ запас — буфер в ОЗУ")
    elif dev == "cpu" and os.environ.get("OSU_BUF_GPU_CPU") == "1":
        # отладка тензорного пути без CUDA (проверка семантики; только CPU)
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
    env._lag = int(env.np_random.choice(2, p=[0.65, 0.35]))   # 60HZ: 0/1 шаг = 0/16.7мс
    env._press_off = 10.0
    stack = np.repeat(obs, N_STACK, axis=2)

    btn_w, last_info = None, {}
    tim = {"env": 0.0, "prep": 0.0, "gpu": 0.0, "n": 0} if a.timing else None   # план E
    stats = {"loss": 0.0, "m": 0, "b": 0, "v": 0.0, "a": 0.0, "n": 0}
    t0 = time.time()

    log_every = 10_000 if a.steps >= 50_000 else max(500, a.steps // 5)
    for step in range(a.steps):
        t_a = time.perf_counter() if tim is not None else 0.0
        midx, btn = oracle_policy(env)
        ax_, ay_ = aux_target(env)                      # AUX: таргет до step

        if use_buf_gpu:
            buf_obs_t[ptr] = torch.from_numpy(stack)     # H2D: один кадр-стек, 153 КБ
        else:
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
        ptr = (ptr + 1) % buf_size
        fill = min(fill + 1, buf_size)

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
        if tim is not None:
            t_b = time.perf_counter()
            tim["env"] += t_b - t_a
            tim["n"] += 1
        if btn_w is not None and (step + 1) % TRAIN_EVERY == 0:
            idx = np.random.randint(0, fill, size=BATCH)

            newest = (ptr - 1) % buf_size if fill == buf_size else fill - 1
            done_np = buf_done[idx].copy()
            done_np[idx == newest] = 1.0
            nxt = (idx + 1) % fill

            if use_buf_gpu:                              # план A: gather на GPU
                ix = torch.from_numpy(idx.astype(np.int64)).to(dev)
                x = buf_obs_t[ix].permute(0, 3, 1, 2).float() / 255.0
                x2 = buf_obs_t[(ix + 1) % fill].permute(0, 3, 1, 2).float() / 255.0
            else:
                x = torch.from_numpy(buf_obs[idx]).to(dev).permute(0, 3, 1, 2).float() / 255.0
                x2 = torch.from_numpy(buf_obs[nxt]).to(dev).permute(0, 3, 1, 2).float() / 255.0
            mv = torch.from_numpy(buf_move[idx]).to(dev)
            bt = torch.from_numpy(buf_btn[idx]).to(dev)
            at = torch.from_numpy(buf_aux[idx]).to(dev)   # AUX

            if tim is not None:
                if dev == "cuda":
                    torch.cuda.synchronize()             # план E: честная граница сбор/H2D
                t_c = time.perf_counter()
                tim["prep"] += t_c - t_b

            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=use_amp):
                _, _, v_next, _ = net(x2)
                r_t = torch.from_numpy(buf_r[idx]).to(dev)
                d_t = torch.from_numpy(done_np).to(dev)
                target = r_t + GAMMA * (1.0 - d_t) * v_next

            with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_amp):
                lm, lb, lv, la = net(x)
                v_loss = F.smooth_l1_loss(lv, target)
                aux_loss = F.mse_loss(la, at)              # AUX
                loss = (F.cross_entropy(lm, mv)
                        + BTN_W * F.cross_entropy(lb, bt, weight=btn_w)
                        + VALUE_W * v_loss
                        + AUX_W * aux_loss)
            opt.zero_grad(); loss.backward(); opt.step()   # backward вне autocast
            if tim is not None:
                if dev == "cuda":
                    torch.cuda.synchronize()               # план E
                tim["gpu"] += time.perf_counter() - t_c

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
                    model=a.out, model_loaded=True, train_steps=step + 1,
                    loss=stats["loss"] / nb,
                )

        if (step + 1) % log_every == 0:
            k = max(1, stats["n"])
            nb = max(1, stats["n"] // BATCH)
            rate = (step + 1) / (time.time() - t0)
            print(f"{step+1:>8,}/{a.steps:,} | loss={stats['loss']/nb:.3f} "
                  f"| move acc={stats['m']/k:.3f} | btn acc={stats['b']/k:.3f} "
                  f"| value={stats['v']/nb:.2f} | AUX-MSE={stats['a']/nb:.4f} "
                  f"| учитель: acc={last_info.get('acc')} miss={last_info.get('miss')} "
                  f"| {rate:.0f} шаг/с | ост. ~{(a.steps-step-1)/rate/60:.0f} мин")
            if tim is not None:
                n_t = max(1, tim["n"])
                print(f"            тайминг мс/шаг: env {tim['env']/n_t*1000:.2f} "
                      f"| сбор+H2D {tim['prep']/n_t*1000:.2f} | GPU {tim['gpu']/n_t*1000:.2f}")
                tim = {"env": 0.0, "prep": 0.0, "gpu": 0.0, "n": 0}
            stats = {"loss": 0.0, "m": 0, "b": 0, "v": 0.0, "a": 0.0, "n": 0}

    # ============ ПРОГРЕВ КРИТИКА (без aux — голова не участвует) ============
    print("\n=== прогрев критика (MC, actor заморожен) ===")
    WARMUP_STEPS = a.warmup
    wlog_every = 50_000 if WARMUP_STEPS >= 150_000 else max(1, WARMUP_STEPS // 3)
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

        if (wstep + 1) % wlog_every == 0:
            k = max(1, vloss_n)
            print(f"  прогрев {wstep+1:,}/{WARMUP_STEPS:,} | v_loss={vloss_acc/k:.3f}")
            vloss_acc, vloss_n = 0.0, 0

    print("прогрев завершён")
    torch.save(net.state_dict(), a.out + "_bcnet.pt")   # AUX: не терять aux-голову (P1 в E1)
    # ============ перенос весов в PPO (aux_head НЕ переносится) ============
    bcnet_to_ppo(net, maps, N_STACK, dev, a.out)
    print(f"\nсохранено: {a.out}.zip")
    if ai_telemetry is not None:
        ai_telemetry.clear("bc_train")

    # ============ самопроверка ============
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
        x_raw = torch.from_numpy(batch_hwc).to(dev).permute(0, 3, 1, 2).float()
        v_ppo = m2.policy.predict_values(x_raw).squeeze(-1)

    a2, _ = m2.predict(batch_hwc, deterministic=True)
    match_m = (a2[:, 0] == lm.argmax(1).cpu().numpy()).mean()
    match_b = (a2[:, 1] == lb.argmax(1).cpu().numpy()).mean()
    v_corr = (float(np.corrcoef(v_ppo.cpu().numpy(), lv.cpu().numpy())[0, 1])
              if lv.std() > 1e-8 and v_ppo.std() > 1e-8 else 0.0)

    # Δlogits-гейт (спасён из reverse_check.py): распределения PPO и BCNet
    # обязаны совпасть; кривая нормализация даст расхождение на raw-входе
    rng = np.random.default_rng(0)
    maxd = 0.0
    for _ in range(64):
        s = rng.integers(0, 256, size=(120, 160, N_STACK), dtype=np.uint8)
        x = torch.from_numpy(s).to(dev).permute(2, 0, 1).unsqueeze(0).float()
        with torch.no_grad():
            d = m2.policy.get_distribution(x)
            bl, bb, _, _ = net(x / 255.0)
        # torch Categorical нормализует логиты (вычитает logsumexp):
        # сравнение с сырыми логитами BCNet давало ложные ~3.3 даже при
        # побитово точном переносе. Сравниваем log_softmax (move + btn).
        maxd = max(maxd,
                   (d.distribution[0].logits - F.log_softmax(bl, -1)).abs().max().item(),
                   (d.distribution[1].logits - F.log_softmax(bb, -1)).abs().max().item())

    print(f"самопроверка переноса: move {match_m:.3f} | btn {match_b:.3f} "
          f"| value corr {v_corr:.3f} | max |Δlogits| {maxd:.5f} "
          f"(должно быть 1.000 / <1e-2)")
    if match_m < 0.99 or match_b < 0.99 or v_corr < 0.99 or maxd >= 1e-2:
        print("!! перенос сломан — PPO НЕ запускать, кинь мне этот вывод")
    else:
        print(f"перенос чист. дальше: train_stage.py (стек {N_STACK}) → PPO → servo → DAgger")


if __name__ == "__main__":
    main()