# dagger_train.py — одна итерация DAgger на 60Гц-семантике деплоя (env.py as-is)»
# iter1: python dagger_train.py --iter 1 --base <свежая модель 60Гц/стек-8> --oracle-share 0.5
# iter2: python dagger_train.py --iter 2 --resume-bc dagger_iter1_bcnet.pt --oracle-share 0.33
# iter3: python dagger_train.py --iter 3 --resume-bc dagger_iter2_bcnet.pt --oracle-share 0.2
# from-scratch (если reverse-check не сошёлся): iter1 без --base.

import glob, time, json, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from sim_env import SimOsuEnv
from bc_train import BCNet, oracle_policy, aux_target

# ---- настройки: всё, кроме источника данных, — как в bc_train ----
TRAIN_EVERY, BATCH, LR, WARMUP, BUF_SIZE = 4, 256, 3e-4, 2048, 30_000
ACT_NOISE_ORACLE = 0.03        # шум исполнения ТОЛЬКО в oracle-эпизодах
BTN_W, VALUE_W, AUX_W, GAMMA = 1.5, 0.5, 1.0, 0.99
N_STACK = 8     # 60Гц/стек-8 — как у текущих моделей; гвард в reverse_transfer проверит совпадение
LAG_P = (0.4, 0.35, 0.25)      # баланс лага: лечит катастрофы lag=2


def reverse_transfer(net, ck, dev):
    """PPO -> BCNet + инлайн-гейт совпадения (= reverse_check)."""
    m = PPO.load(ck, device="cpu")
    base_k = int(m.policy.features_extractor.cnn[0].in_channels)
    if base_k != N_STACK:
        raise SystemExit(
            f"!! База {ck} учится на стеке {base_k}, а DAgger собран на N_STACK={N_STACK}.\n"
            f"   Возьми --base из 60Гц-линии (стек 8) либо поменяй N_STACK на {base_k}.")
    pol = m.policy
    net.cnn.load_state_dict(pol.features_extractor.cnn.state_dict())
    net.linear.load_state_dict(pol.features_extractor.linear[0].state_dict())
    with torch.no_grad():
        net.move_head.weight.copy_(pol.action_net.weight[:27]); net.move_head.bias.copy_(pol.action_net.bias[:27])
        net.btn_head.weight.copy_(pol.action_net.weight[27:]);  net.btn_head.bias.copy_(pol.action_net.bias[27])
        net.value_head.weight.copy_(pol.value_net.weight);      net.value_head.bias.copy_(pol.value_net.bias)
    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(256):
        s = rng.integers(0, 256, size=(120, 160, N_STACK), dtype=np.uint8)
        x = torch.from_numpy(s).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(dev)
        with torch.no_grad():
            lm, lb, _, _ = net(x)
        a, _ = m.predict(s, deterministic=True)
        ok += int(a[0] == int(lm.argmax())) + int(a[1] == int(lb.argmax()))
    acc = ok / 512.0
    print(f"[reverse-check] совпадение: {acc:.3f} (гейт 0.99)")
    if acc < 0.99:
        raise SystemExit("!! перенос не сошёлся — перезапусти iter1 БЕЗ --base (from-scratch)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--base", default=None)
    p.add_argument("--resume-bc", default=None)
    p.add_argument("--oracle-share", type=float, required=True)
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--teleport-p", type=float, default=0.01)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    maps = sorted(glob.glob("sim_maps/*.osu")) + sorted(glob.glob("sim_maps_t1/*.osu"))
    assert maps, "нет карт"

    env = SimOsuEnv(maps, domain_rand=True, teleport_prob=args.teleport_p)
    net = BCNet(n_stack=N_STACK).to(dev)
    if args.resume_bc:
        net.load_state_dict(torch.load(args.resume_bc, map_location=dev))
        print(f"[init] продолжаю с {args.resume_bc}")
    elif args.base:
        reverse_transfer(net, args.base, dev)
    else:
        print("[init] from-scratch (нет --base/--resume-bc)")

    opt = torch.optim.Adam(net.parameters(), lr=LR)
    OUT = f"dagger_iter{args.iter}"

    buf_obs = np.zeros((BUF_SIZE, 120, 160, N_STACK), dtype=np.uint8)
    buf_move = np.zeros(BUF_SIZE, dtype=np.int64)
    buf_btn = np.zeros(BUF_SIZE, dtype=np.int64)
    buf_aux = np.zeros((BUF_SIZE, 2), dtype=np.float32)
    buf_r = np.zeros(BUF_SIZE, dtype=np.float32)
    buf_done = np.zeros(BUF_SIZE, dtype=np.float32)
    fill, ptr = 0, 0

    comp = {"frames": 0, "far": 0, "veryfar": 0, "far_o": 0, "far_p": 0,
            "n_o": 0, "n_p": 0, "lag": [0, 0, 0],
            "ep_oracle": 0, "ep_policy": 0, "oracle_share": args.oracle_share,
            "teleport_p": args.teleport_p, "steps": args.steps}

    obs, _ = env.reset()
    env._lag = int(env.np_random.choice(3, p=LAG_P))
    env._press_off = 10.0          # как в bc_train; известный остаточный мисматч с eval
    driver = "oracle" if env.np_random.random() < args.oracle_share else "policy"
    comp["ep_oracle" if driver == "oracle" else "ep_policy"] += 1
    stack = np.repeat(obs, N_STACK, axis=2)

    btn_w, last_info = None, {}
    stats = {"loss": 0.0, "m": 0, "b": 0, "v": 0.0, "a": 0.0, "n": 0}
    t0 = time.time()

    for step in range(args.steps):
        # 1) метка — ВСЕГДА оракул, на текущем состоянии (онлайн-разметка)
        midx, btn = oracle_policy(env)
        ax_, ay_ = aux_target(env)

        # 2) исполнение по драйверу эпизода
        if driver == "oracle":
            exec_midx, exec_btn = midx, btn
            if env.np_random.random() < ACT_NOISE_ORACLE:
                exec_midx = int(env.np_random.integers(0, 27))
        else:
            x = torch.from_numpy(stack).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(dev)
            with torch.no_grad():
                lm, lb, _, _ = net(x)
            exec_midx = int(torch.multinomial(F.softmax(lm.squeeze(0), 0), 1))
            exec_btn = int(torch.multinomial(F.softmax(lb.squeeze(0), 0), 1))

        # 3) в буфер — состояние и МЕТКУ (не исполнение)
        buf_obs[ptr] = stack
        buf_move[ptr] = midx
        buf_btn[ptr] = btn
        buf_aux[ptr] = (ax_, ay_)

        obs, r, term, trunc, info = env.step(np.array([exec_midx, exec_btn], dtype=np.int64))
        stack = np.concatenate([stack[:, :, 1:], obs], axis=2)
        buf_r[ptr] = r
        buf_done[ptr] = float(term or trunc)
        ptr = (ptr + 1) % BUF_SIZE
        fill = min(fill + 1, BUF_SIZE)
        if info:
            last_info = info
            last_info["driver"] = driver          # кто вёл закончившийся эпизод

        # 4) состав данных
        comp["frames"] += 1
        comp["lag"][env._lag] += 1
        g = env._goal()
        if g is not None:
            d = math.hypot(g[0] - env.cursor_x, g[1] - env.cursor_y)
            if d > 25: comp["far"] += 1
            if d > 50: comp["veryfar"] += 1
            k = "o" if driver == "oracle" else "p"
            comp["n_" + k] += 1
            comp["far_" + k] += int(d > 25)

        if term or trunc:
            obs, _ = env.reset()
            env._lag = int(env.np_random.choice(3, p=LAG_P))
            env._press_off = 10.0
            driver = "oracle" if env.np_random.random() < args.oracle_share else "policy"
            comp["ep_oracle" if driver == "oracle" else "ep_policy"] += 1
            stack = np.repeat(obs, N_STACK, axis=2)

        # 5) веса кнопок (как в bc_train)
        if (btn_w is None and fill >= WARMUP) or \
           (btn_w is not None and (step + 1) % 25_000 == 0):
            counts = np.bincount(buf_btn[:fill], minlength=5)
            w = 1.0 / np.maximum(counts, 1).astype(np.float32)
            w /= w.mean(); w[0] = min(w[0], 0.3)
            btn_w = torch.tensor(np.clip(w, 0.05, 10.0), device=dev)

        # 6) обучение: лосс и гиперпараметры — без изменений
        if btn_w is not None and (step + 1) % TRAIN_EVERY == 0:
            idx = np.random.randint(0, fill, size=BATCH)
            newest = (ptr - 1) % BUF_SIZE if fill == BUF_SIZE else fill - 1
            done_np = buf_done[idx].copy(); done_np[idx == newest] = 1.0
            nxt = (idx + 1) % fill

            x = torch.from_numpy(buf_obs[idx]).to(dev).permute(0, 3, 1, 2).float() / 255.0
            mv = torch.from_numpy(buf_move[idx]).to(dev)
            bt = torch.from_numpy(buf_btn[idx]).to(dev)
            at = torch.from_numpy(buf_aux[idx]).to(dev)
            with torch.no_grad():
                x2 = torch.from_numpy(buf_obs[nxt]).to(dev).permute(0, 3, 1, 2).float() / 255.0
                _, _, v_next, _ = net(x2)
                target = (torch.from_numpy(buf_r[idx]).to(dev)
                          + GAMMA * (1.0 - torch.from_numpy(done_np).to(dev)) * v_next)

            lm, lb, lv, la = net(x)
            v_loss = F.smooth_l1_loss(lv, target)
            loss = (F.cross_entropy(lm, mv)
                    + BTN_W * F.cross_entropy(lb, bt, weight=btn_w)
                    + VALUE_W * v_loss
                    + AUX_W * F.mse_loss(la, at))
            opt.zero_grad(); loss.backward(); opt.step()

            stats["loss"] += loss.item()
            stats["m"] += (lm.argmax(1) == mv).sum().item()
            stats["b"] += (lb.argmax(1) == bt).sum().item()
            stats["a"] += F.mse_loss(la, at).item()
            stats["n"] += BATCH

        if (step + 1) % 10_000 == 0:
            k, nb = max(1, stats["n"]), max(1, stats["n"] // BATCH)
            rate = (step + 1) / (time.time() - t0)
            fo = comp["far_o"] / max(1, comp["n_o"])
            fp = comp["far_p"] / max(1, comp["n_p"])
            print(f"{step+1:>8,} | loss={stats['loss']/nb:.3f} | agree(буфер)={stats['m']/k:.3f} "
                  f"| btn={stats['b']/k:.3f} | AUX={stats['a']/nb:.4f} | ep_acc={last_info.get('acc')}({last_info.get('driver','?')}) "
                  f"| far={comp['far']/max(1,comp['frames']):.2f} (o {fo:.2f} / p {fp:.2f}) "
                  f"veryfar={comp['veryfar']} lag={comp['lag']} "
                  f"| ep o/p={comp['ep_oracle']}/{comp['ep_policy']} | {rate:.0f} ш/с")
            stats = {"loss": 0.0, "m": 0, "b": 0, "v": 0.0, "a": 0.0, "n": 0}
        if (step + 1) % 50_000 == 0:
            torch.save(net.state_dict(), OUT + "_live.pt")
            print(f"[live] чекпоинт: {OUT}_live.pt")

    # ---- сохранение: BCNet (для продолжения) + PPO-обёртка (для eval/play) ----
    torch.save(net.state_dict(), OUT + "_bcnet.pt")
    venv = VecTransposeImage(VecFrameStack(
        DummyVecEnv([lambda: Monitor(SimOsuEnv(maps))]),
        n_stack=N_STACK, channels_order="last"))
    model = PPO("CnnPolicy", venv, n_steps=512, batch_size=512,
                learning_rate=5e-5, ent_coef=0.0, verbose=0, device=dev)
    pol = model.policy
    sd = net.state_dict()
    assert len(pol.mlp_extractor.policy_net) == 0
    assert pol.action_net.weight.shape == (32, 512)
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

    with open(OUT + "_composition.json", "w", encoding="utf-8") as f:
        json.dump(comp, f, indent=1)
    print(f"сохранено: {OUT}.zip + {OUT}_bcnet.pt + {OUT}_composition.json")
    print("состав:", {k: v for k, v in comp.items() if k != "lag"}, "| lag:", comp["lag"])


if __name__ == "__main__":
    main()
