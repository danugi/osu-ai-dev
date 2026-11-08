# dagger_train2.py — DAgger v3 (попытка В): плотность А + свежесть Б + фабрика ближних состояний.
#
# ИСТОРИЯ (30 сидов, det rand, sim_maps):
#   Попытка А (dagger_train.py, кольцо 30k, ~64 прохода/кадр, 200k шагов):
#     ЛУЧШАЯ по замкнутому циклу — acc 0.123±0.073, aim 26/68px, recovery 0.23;
#     но tf-поведение деградировало (nav_tf 3->15px, agree_tf 0.269->0.111).
#   Попытка Б (v2: чанки x 3 эпохи, 150k шагов): tf восстановлен (3px), но цикл
#     на уровне базлайна — acc 0.067±0.047, aim 39/122px; А лучше на 28/30
#     сидов (парный p~9e-7). Вывод: активный ингредиент А — ПЛОТНОСТЬ
#     оптимизации (12.8M градиентных сэмплов против 450k у Б, x28).
#     «3 эпохи» — недоучение (in-sample agree 0.28-0.40), а не честность.
#   Попытка В (эти дефолты; --iter 2 от live200k.pt, 300k шагов): tf 7px,
#     cl nav 68px, acc 0.073 (gap) / 0.085±0.058 (v3; aim 28/87, rec 0.15).
#     Плотность уровня А + целевые телепорты НЕ сломали стену: cl nav 66-68px
#     в пятом прогоне подряд (BC 66, А 61-70, Б 68, В 68). Гипотезы недоучения
#     и дефицита ближних состояний закрыты. Остаточная гипотеза — сеть не
#     выучивает «далеко -> поверни к цели» из пикселей; её проверяет
#     far_probe.py (однотактный тест, см. README). Новые итерации DAgger
#     НЕ запускать до вердикта far_probe.
#
#   Дефицит ближних состояний: far p ~0.96 в ОБОИХ прогонах => политика
#   проводит <25px от цели лишь ~4% своих кадров. Навык «серво у цели из
#   своих диких подходов» получает у Б ~9k сэмплов, у А ~256k — отсюда
#   разница. Оракульных ближних кадров полно (far o 0.11-0.23), но контекст
#   другой — перенос слабый (recovery 0.13-0.23, первый промах на объекте #0).
#
# ПОПЫТКА В — три изменения против Б:
#   1) плотность как у А: чанк 20k x 32 эпохи = 9.6M сэмплов на 300k
#      УНИКАЛЬНЫХ кадров (у А 12.8M на вращающемся окне 30k);
#   2) ЦЕЛЕВЫЕ телепорты (--goal-tp-p): курсор прыгает в кольцо 0-35px вокруг
#      текущей цели => фабрикует «ближние» состояния с дикой историей в стеке;
#      следующая метка оракула = коррекция серво. Заодно тренирует старт
#      эпизода (случайный курсор = тот же телепорт);
#   3) автопробник после каждого чанка: 2000 шагов замкнутого цикла (det
#      argmax, фикс. seed) — тренд nav med и acc виден без ручных замеров.
#
# Запуск (попытка В, ~1 час на GPU):
#   python dagger_train2.py --iter 2 --resume-bc live200k.pt --steps 300000 \
#       --chunk 20000 --epochs 32 --oracle-share 0.33 --teleport-p 0.02 --goal-tp-p 0.10
# iter3 дальше: --resume-bc dagger2_iter2_bcnet.pt --oracle-share 0.2

import glob, time, json, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
from sim_env import SimOsuEnv
from oracle import oracle_policy, aux_target
from net import BCNet, bcnet_to_ppo
from models import infer_stack_file
from core import decode_move
from vision import OSU_W, OSU_H

# ---- настройки по умолчанию = рецепт попытки В ----
STEPS, CHUNK, EPOCHS = 300_000, 20_000, 32
ORACLE_SHARE, TELEPORT_P, GOAL_TP_P = 0.33, 0.02, 0.10
PROBE = 2000            # шагов автопробника после чанка (0 = выкл)
BATCH, LR = 256, 3e-4
ACT_NOISE_ORACLE = 0.03        # шум исполнения ТОЛЬКО в oracle-эпизодах
BTN_W, VALUE_W, AUX_W, GAMMA = 1.5, 0.5, 1.0, 0.99
LAG_P = (0.4, 0.35, 0.25)      # баланс лага: лечит катастрофы lag=2


def probe(net, maps, n_stack, n_steps, seed, dev):
    """Быстрый замер замкнутого цикла (det argmax) на свежем окружении.
    Тот же seed каждый раз -> числа сравнимы между чанками."""
    env = SimOsuEnv(maps, domain_rand=True)
    obs, _ = env.reset(seed=seed)
    env._lag = int(env.np_random.choice(2, p=[0.65, 0.35]))
    env._press_off = 10.0
    stack = np.repeat(obs, n_stack, axis=2)
    nav, accs = [], []
    for _ in range(n_steps):
        midx, btn = oracle_policy(env)
        x = torch.from_numpy(stack).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(dev)
        with torch.no_grad():
            lm, lb, _, _ = net(x)
        pm, pb = int(lm.argmax()), int(lb.argmax())
        d1, s1 = decode_move(midx)
        d2, s2 = decode_move(pm)
        v1 = np.array([d1[0] * s1, d1[1] * s1], dtype=np.float64)
        v2 = np.array([d2[0] * s2, d2[1] * s2], dtype=np.float64)
        nav.append(float(np.hypot(*(v1 - v2))))
        obs, r, term, trunc, info = env.step(np.array([pm, pb], dtype=np.int64))
        stack = np.concatenate([stack[:, :, 1:], obs], axis=2)
        if info:
            accs.append(info.get("acc"))
        if term or trunc:
            obs, _ = env.reset()
            env._lag = int(env.np_random.choice(2, p=[0.65, 0.35]))
            env._press_off = 10.0
            stack = np.repeat(obs, n_stack, axis=2)
    return float(np.median(nav)), (float(np.mean(accs)) if accs else None), len(accs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--resume-bc", required=True,
                   help="live200k.pt (попытка А, измерена 0.123) или dagger2_iter1_bcnet.pt")
    p.add_argument("--oracle-share", type=float, default=ORACLE_SHARE)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--teleport-p", type=float, default=TELEPORT_P)
    p.add_argument("--goal-tp-p", type=float, default=GOAL_TP_P,
                   help="целевые телепорты: курсор -> кольцо 0-35px у цели")
    p.add_argument("--chunk", type=int, default=CHUNK)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--probe", type=int, default=PROBE,
                   help="шагов автопробника после чанка (0 = выкл)")
    p.add_argument("--maps", default="sim_maps,sim_maps_t1",
                   help="папки с картами через запятую")
    args = p.parse_args()

    N_STACK = infer_stack_file(args.resume_bc)
    print(f"[stack] N_STACK={N_STACK}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    if dev != "cuda":
        print("!! GPU не найден — обучение пойдёт на CPU (в разы дольше).")

    maps = []
    for d in (x.strip() for x in args.maps.split(",")):
        maps += sorted(glob.glob(d + "/*.osu"))
    assert maps, "нет карт в " + args.maps

    env = SimOsuEnv(maps, domain_rand=True, teleport_prob=args.teleport_p)
    net = BCNet(n_stack=N_STACK).to(dev)
    net.load_state_dict(torch.load(args.resume_bc, map_location=dev))
    print(f"[init] продолжаю с {args.resume_bc}")
    gs = args.chunk * args.epochs
    print(f"[режим В] чанк={args.chunk} x {args.epochs} эп = {gs/1e6:.2f}M сэмплов/чанк, "
          f"всего {args.steps*args.epochs/1e6:.1f}M; телепорт rand={args.teleport_p} "
          f"целевой={args.goal_tp_p}; oracle-share={args.oracle_share}; probe={args.probe}")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    OUT = f"dagger2_iter{args.iter}"

    ch_obs = np.zeros((args.chunk, 120, 160, N_STACK), dtype=np.uint8)
    ch_move = np.zeros(args.chunk, dtype=np.int64)
    ch_btn = np.zeros(args.chunk, dtype=np.int64)
    ch_aux = np.zeros((args.chunk, 2), dtype=np.float32)
    ch_r = np.zeros(args.chunk, dtype=np.float32)
    ch_done = np.zeros(args.chunk, dtype=np.float32)

    comp = {"far": 0, "veryfar": 0, "far_o": 0, "far_p": 0, "n_o": 0, "n_p": 0,
            "lag": [0, 0, 0], "frames": 0, "ep_oracle": 0, "ep_policy": 0,
            "tot_frames": 0, "tot_veryfar": 0, "tot_far": 0, "tot_lag": [0, 0, 0],
            "chunks": 0, "oracle_share": args.oracle_share,
            "teleport_p": args.teleport_p, "goal_tp_p": args.goal_tp_p,
            "steps": args.steps, "chunk": args.chunk, "epochs": args.epochs,
            "probe": args.probe, "resume": args.resume_bc}

    obs, _ = env.reset()
    env._lag = int(env.np_random.choice(3, p=LAG_P))
    env._press_off = 10.0
    driver = "oracle" if env.np_random.random() < args.oracle_share else "policy"
    comp["ep_oracle" if driver == "oracle" else "ep_policy"] += 1
    stack = np.repeat(obs, N_STACK, axis=2)

    last_info = {}
    t0 = time.time()
    fill, chunk_id = 0, 0

    def train_pass(n, epochs, tag):
        """Ровно `epochs` истинных эпох по первым n кадрам чанка (без повторов
        внутри эпохи), TD-цель по соседу внутри чанка."""
        if n < BATCH:
            return
        counts = np.bincount(ch_btn[:n], minlength=5)
        w = 1.0 / np.maximum(counts, 1).astype(np.float32)
        w /= w.mean(); w[0] = min(w[0], 0.3)
        btn_w = torch.tensor(np.clip(w, 0.05, 10.0), device=dev)
        st = {"loss": 0.0, "m": 0, "b": 0, "n": 0, "nb": 0}
        for _ in range(epochs):
            perm = np.random.permutation(n)
            for b0 in range(0, n - BATCH + 1, BATCH):
                idx = perm[b0:b0 + BATCH]
                nxt = np.minimum(idx + 1, n - 1)
                dn = ch_done[idx].copy()
                dn[idx + 1 >= n] = 1.0
                x = torch.from_numpy(ch_obs[idx]).to(dev).permute(0, 3, 1, 2).float() / 255.0
                mv = torch.from_numpy(ch_move[idx]).to(dev)
                bt = torch.from_numpy(ch_btn[idx]).to(dev)
                at = torch.from_numpy(ch_aux[idx]).to(dev)
                with torch.no_grad():
                    x2 = torch.from_numpy(ch_obs[nxt]).to(dev).permute(0, 3, 1, 2).float() / 255.0
                    _, _, v_next, _ = net(x2)
                    target = (torch.from_numpy(ch_r[idx]).to(dev)
                              + GAMMA * (1.0 - torch.from_numpy(dn).to(dev)) * v_next)
                lm, lb, lv, la = net(x)
                v_loss = F.smooth_l1_loss(lv, target)
                loss = (F.cross_entropy(lm, mv)
                        + BTN_W * F.cross_entropy(lb, bt, weight=btn_w)
                        + VALUE_W * v_loss
                        + AUX_W * F.mse_loss(la, at))
                opt.zero_grad(); loss.backward(); opt.step()
                st["loss"] += loss.item(); st["nb"] += 1
                st["m"] += (lm.argmax(1) == mv).sum().item()
                st["b"] += (lb.argmax(1) == bt).sum().item()
                st["n"] += len(idx)
        print(f"[{tag}] loss={st['loss']/max(1,st['nb']):.3f} "
              f"| agree(чанк in-sample)={st['m']/max(1,st['n']):.3f} "
              f"| btn={st['b']/max(1,st['n']):.3f}", flush=True)

    for step in range(args.steps):
        # 1) метка — ВСЕГДА оракул, на текущем состоянии
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

        # 3) в чанк — состояние и МЕТКУ (не исполнение)
        ch_obs[fill] = stack
        ch_move[fill] = midx
        ch_btn[fill] = btn
        ch_aux[fill] = (ax_, ay_)

        # 3.5) ЦЕЛЕВОЙ телепорт («толчок к цели»): курсор -> кольцо 0-35px
        # вокруг текущей цели. Пара (стек, метка) уже записана — консистентна;
        # прыжок виден в СЛЕДУЮЩЕМ кадре, метка след. шага = коррекция серво.
        # Именно этих состояний в policy-эпизодах ~4% (far p 0.96).
        if args.goal_tp_p > 0 and env.np_random.random() < args.goal_tp_p:
            g = env._goal()
            if g is not None:
                ang = env.np_random.uniform(0.0, 2.0 * math.pi)
                dist = env.np_random.uniform(0.0, 35.0)
                env.cursor_x = min(OSU_W - 20.0, max(20.0, g[0] + math.cos(ang) * dist))
                env.cursor_y = min(OSU_H - 20.0, max(20.0, g[1] + math.sin(ang) * dist))

        obs, r, term, trunc, info = env.step(np.array([exec_midx, exec_btn], dtype=np.int64))
        stack = np.concatenate([stack[:, :, 1:], obs], axis=2)
        ch_r[fill] = r
        ch_done[fill] = float(term or trunc)
        fill += 1
        if info:
            last_info = info
            last_info["driver"] = driver

        # 4) статистика чанка
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

        if (step + 1) % 10_000 == 0 and fill != args.chunk:
            fp = comp["far_p"] / max(1, comp["n_p"])
            print(f"{step+1:>8,} [сбор чанка {chunk_id+1}: {fill}/{args.chunk}] "
                  f"far_p={fp:.2f} ep_acc={last_info.get('acc')}"
                  f"({last_info.get('driver','?')}) "
                  f"| {(step+1)/(time.time()-t0):.0f} ш/с", flush=True)

        # ---- чанк заполнен: эпохи -> автопробник -> печать -> вытеснение ----
        if fill == args.chunk:
            chunk_id += 1
            train_pass(args.chunk, args.epochs, f"чанк {chunk_id} x{args.epochs} эп")
            if args.probe > 0:
                pnav, pacc, pep = probe(net, maps, N_STACK, args.probe, 2024, dev)
                acc_s = f"{pacc:.3f}" if pacc is not None else "—"
                print(f"[probe чанк {chunk_id}] nav med={pnav:.0f}px | acc={acc_s} "
                      f"({pep} эп)", flush=True)
            fo = comp["far_o"] / max(1, comp["n_o"])
            fp = comp["far_p"] / max(1, comp["n_p"])
            print(f"[итог чанка {chunk_id}] шаг {step+1:,} "
                  f"| ep_acc={last_info.get('acc')}({last_info.get('driver','?')}) "
                  f"| far={comp['far']/max(1,comp['frames']):.2f} (o {fo:.2f} / p {fp:.2f}) "
                  f"veryfar={comp['veryfar']} lag={comp['lag']} "
                  f"| ep o/p={comp['ep_oracle']}/{comp['ep_policy']} "
                  f"| {(step+1)/(time.time()-t0):.0f} ш/с", flush=True)
            comp["chunks"] = chunk_id
            comp["tot_frames"] += comp["frames"]
            comp["tot_veryfar"] += comp["veryfar"]
            comp["tot_far"] += comp["far"]
            comp["tot_lag"] = [a + b for a, b in zip(comp["tot_lag"], comp["lag"])]
            comp.update(far=0, far_o=0, far_p=0, n_o=0, n_p=0,
                        frames=0, veryfar=0, lag=[0, 0, 0])
            torch.save(net.state_dict(), OUT + "_live.pt")
            print(f"[live] чекпоинт после чанка {chunk_id}: {OUT}_live.pt", flush=True)
            fill = 0

    # ---- хвост: неполный чанк укороченной прогонкой ----
    if fill >= BATCH:
        train_pass(fill, min(args.epochs, 8), f"хвост {fill} кадров x{min(args.epochs,8)} эп")
        if args.probe > 0:
            pnav, pacc, pep = probe(net, maps, N_STACK, args.probe, 2024, dev)
            acc_s = f"{pacc:.3f}" if pacc is not None else "—"
            print(f"[probe хвост] nav med={pnav:.0f}px | acc={acc_s} ({pep} эп)", flush=True)

    torch.save(net.state_dict(), OUT + "_bcnet.pt")
    bcnet_to_ppo(net, maps, N_STACK, dev, OUT)
    with open(OUT + "_composition.json", "w", encoding="utf-8") as f:
        json.dump(comp, f, indent=1)
    print(f"сохранено: {OUT}.zip + {OUT}_bcnet.pt + {OUT}_composition.json")
    print("состав:", {k: v for k, v in comp.items()
                      if k not in ("lag", "tot_lag", "far", "veryfar", "far_o",
                                   "far_p", "n_o", "n_p", "frames")},
          "| lag(всего):", comp["tot_lag"])


if __name__ == "__main__":
    main()
