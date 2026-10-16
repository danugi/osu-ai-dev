import glob, os, sys, csv, math
import numpy as np
import torch
from stable_baselines3 import PPO
from sim_env import SimOsuEnv
from bc_train import oracle_policy

# Servo v2: до 100 телепортов, трассы, agreement политики с оракулом (P2/P3),
# paste/drift-режимы (P4), направление смещения vs скорость шара (P5).
# Без аргументов — интерактив. С аргументами:
#   python servo_v2.py model.zip [policy|oracle] [jump|paste|drift] [det|sto] [n_seeds]

ATTEMPTS, TIMEOUT = 4, 120
PX_X, PX_Y = (300.0/512.0)*0.5, (224.0/384.0)*0.5   # osu-px -> px входа сети


def ask(prompt, default):
    s = input(f"{prompt} [{default}]: ").strip()
    return s if s else str(default)


def pick_model(default):
    cands = []
    for pat in ("bc_policy*.zip", "*_final.zip", "*_guard.zip",
                "*_interrupted.zip", "best_*/*.zip", "checkpoints_*/*.zip"):
        cands += sorted(glob.glob(pat))
    if not cands:
        return ask("Путь к модели (.zip)", default)
    print("Модели:")
    for i, p in enumerate(cands):
        print(f"  {i}: {p}" + ("   <- по умолчанию" if p == default else ""))
    s = input(f"Номер или свой путь [{default}]: ").strip()
    if s == "":
        return default
    if s.isdigit() and int(s) < len(cands):
        return cands[int(s)]
    return s


if len(sys.argv) > 1:
    N_STACK = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    HZ30 = len(sys.argv) > 7 and sys.argv[7] == "30"
    ck = sys.argv[1]
    DRIVER = sys.argv[2] if len(sys.argv) > 2 else "policy"
    PERTURB = sys.argv[3] if len(sys.argv) > 3 else "jump"
    DET = not (len(sys.argv) > 4 and sys.argv[4] == "sto")
    N_SEEDS = int(sys.argv[5]) if len(sys.argv) > 5 else 25
else:
    print("=== servo_v2 — интерактивный режим ===")
    ck = pick_model("bc_policy.zip")
    DRIVER = ask("Драйвер: policy (ведёт модель) или oracle (ведёт учитель)", "policy")
    PERTURB = ask("Возмущение: jump | paste | drift", "jump")
    DET = ask("Режим: argmax (det) или сэмплинг (sto)", "det") == "det"
    N_SEEDS = int(ask("Сидов (x4 попытки = телепортов)", "25"))
    N_STACK = int(ask("Стек (stage4=8, 30Гц-линия=4)", 8))
    HZ30 = ask("30Hz-патч: 30 или нет", "нет") == "30"
    print(f'\nэквивалент: python servo_v2.py "{ck}" {DRIVER} {PERTURB} '
          f'{"det" if DET else "sto"} {N_SEEDS}')
if HZ30:
    if N_STACK != 4:
        raise SystemExit("30Hz-патч только со стеком 4")
    import sim_env as _sim, bc_train as _bt
    # ВАЖНО: bc_train импортировал DT_MS/MOVE_DISTANCES "по значению" —
    # патчим ОБА модуля, иначе оракул будет целиться на 16.7мс вместо 33
    _sim.MOVE_DISTANCES = [6, 24, 96]; _sim.DT_MS = 1000.0 / 30.0
    _bt.MOVE_DISTANCES = [6, 24, 96];  _bt.DT_MS = 1000.0 / 30.0
    print("[30Hz-патч] MOVE_DISTANCES=[6,24,96], DT_MS=33.33")

model = PPO.load(ck, device="cpu")
maps = sorted(glob.glob("sim_maps/*.osu"))
assert maps, "нет карт в sim_maps"


def policy_stats(s):
    """(p_move, p_btn) распределения политики на стеке s."""
    with torch.no_grad():
        x = torch.from_numpy(s.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        d = model.policy.get_distribution(x)
        return (d.distribution[0].probs.squeeze(0).numpy(),
                d.distribution[1].probs.squeeze(0).numpy())


def shift_img(s, dx, dy):
    """Сдвиг каналов истории (paste): прошлый контент 'был всегда там'."""
    out = np.zeros_like(s)
    h, w = s.shape[:2]
    xs, xe = max(0, dx), min(w, w + dx)
    ys, ye = max(0, dy), min(h, h + dy)
    if xe > xs and ye > ys:
        out[ys:ye, xs:xe] = s[ys-dy:ye-dy, xs-dx:xe-dx]
    return out


rows, bg_agree, bg_plabel = [], [], []
bg_far, bg_near = [], []


def finish(cur, ok, steps, reason):
    ag, pl = cur["agree"], cur["plabel"]
    rows.append({
        "seed": cur["seed"], "att": cur["attempt"], "perturb": PERTURB,
        "driver": DRIVER, "mode": "det" if DET else "sto",
        "dx": cur["dx"], "dy": cur["dy"], "cos": cur["cos"],
        "ok": int(ok), "steps": steps if ok else "", "reason": reason,
        "agree3": round(float(np.mean(ag[:3])), 3) if ag else "",
        "plabel3": round(float(np.mean(pl[:3])), 3) if pl else "",
        "agree10": round(float(np.mean(ag[:10])), 3) if ag else "",
        "plabel10": round(float(np.mean(pl[:10])), 3) if pl else "",
        "trace": ";".join(str(x) for x in cur["trace"][:TIMEOUT])})


for seed in range(N_SEEDS):
    env = SimOsuEnv(maps, domain_rand=False)    # изоляция: только наше возмущение
    obs, _ = env.reset(seed=seed)
    s = np.repeat(obs, N_STACK, axis=2)
    done, attempts, last_sl, cur = False, 0, -1, None

    while not done:
        # 1) метка оракула + распределение политики на текущем кадре
        lm, lb = oracle_policy(env)
        pm, pb = policy_stats(s)
        pred = int(pm.argmax())
        if cur is not None:
            cur["agree"].append(pred == lm)
            cur["plabel"].append(float(pm[lm]))
            if env._open_sliders:
                o = env.hitobjects[env._open_sliders[-1]]
                bx, by = env._slider_pos(o, env.sim_time)
                cur["trace"].append(round(math.hypot(bx - env.cursor_x, by - env.cursor_y), 1))
        else:
            bg_agree.append(pred == lm)
            bg_plabel.append(float(pm[lm]))
            g = env._goal()
            if g is not None:
                _d = math.hypot(g[0] - env.cursor_x, g[1] - env.cursor_y)
                if _d > 25:
                    bg_far.append(pred == lm)
                elif _d < 6:
                    bg_near.append(pred == lm)

        # 2) исполнение
        if DRIVER == "oracle":
            a = np.array([lm, lb], dtype=np.int64)
        elif DET:
            a = np.array([pred, int(pb.argmax())], dtype=np.int64)
        else:
            a, _ = model.predict(s, deterministic=False)

        # 3) drift: толкаем курсор прочь от шара 2px/шаг
        if cur is not None and cur["drift_left"] > 0 and env._open_sliders:
            o = env.hitobjects[env._open_sliders[-1]]
            bx, by = env._slider_pos(o, env.sim_time)
            dx, dy = env.cursor_x - bx, env.cursor_y - by
            n = math.hypot(dx, dy) or 1.0
            env.cursor_x = min(492, max(20, env.cursor_x + dx / n * 2.0))
            env.cursor_y = min(364, max(20, env.cursor_y + dy / n * 2.0))
            cur["drift_left"] -= 1

        # 4) старт jump/paste-попытки: ПОСЛЕ выбора действия (первый шаг вслепую)
        paste = None
        if (cur is None and PERTURB in ("jump", "paste")
                and attempts < ATTEMPTS and env._open_sliders):
            oi = env._open_sliders[-1]
            if oi != last_sl and env.sim_time > env.hitobjects[oi]["time"] + 150:
                o = env.hitobjects[oi]
                ang = env.np_random.uniform(0, 2 * math.pi)
                tx = min(492, max(20, env.cursor_x + math.cos(ang) * 50))
                ty = min(364, max(20, env.cursor_y + math.sin(ang) * 50))
                vx, vy = tx - env.cursor_x, ty - env.cursor_y
                if math.hypot(vx, vy) >= 30:      # clamp у стены не съел смещение
                    env.cursor_x, env.cursor_y = tx, ty
                    bx1, by1 = env._slider_pos(o, env.sim_time)
                    bx0, by0 = env._slider_pos(o, env.sim_time - 100)
                    sp = math.hypot(bx1 - bx0, by1 - by0)
                    cosv = ((vx * (bx1 - bx0) + vy * (by1 - by0))
                            / (math.hypot(vx, vy) * sp)) if sp > 1e-6 else 0.0
                    cur = {"seed": seed, "attempt": attempts, "dx": round(vx, 1),
                           "dy": round(vy, 1), "cos": round(cosv, 2),
                           "agree": [], "plabel": [], "trace": [],
                           "drift_left": 0, "t": 0}
                    if PERTURB == "paste":
                        paste = (int(round(vx * PX_X)), int(round(vy * PX_Y)))
                    last_sl, attempts = oi, attempts + 1

        # 5) старт drift-попытки
        if (cur is None and PERTURB == "drift"
                and attempts < ATTEMPTS and env._open_sliders):
            oi = env._open_sliders[-1]
            if oi != last_sl and env.sim_time > env.hitobjects[oi]["time"] + 150:
                cur = {"seed": seed, "attempt": attempts, "dx": "", "dy": "",
                       "cos": "", "agree": [], "plabel": [], "trace": [],
                       "drift_left": 25, "t": 0}
                last_sl, attempts = oi, attempts + 1

        # 6) шаг и стек (paste: 7 старых каналов сдвигаем, новый кадр честный)
        obs, r, term, trunc, info = env.step(a)
        if paste is not None:
            s = np.concatenate([shift_img(s[:, :, 1:], *paste), obs], axis=2)
        else:
            s = np.concatenate([s[:, :, 1:], obs], axis=2)
        done = term or trunc

        # 7) проверка возврата
        if cur is not None:
            cur["t"] += 1
            if not env._open_sliders:
                finish(cur, False, None, "slider_end"); cur = None
            elif cur["drift_left"] == 0:
                o = env.hitobjects[env._open_sliders[-1]]
                bx, by = env._slider_pos(o, env.sim_time)
                if math.hypot(bx - env.cursor_x, by - env.cursor_y) <= env.cur["r_osu"]:
                    finish(cur, True, cur["t"], "ok"); cur = None
                elif cur["t"] > TIMEOUT:
                    finish(cur, False, None, "timeout"); cur = None

    if cur is not None:
        finish(cur, False, None, "episode_end")

# ---- итоги ----
oks = [r for r in rows if r["ok"]]
print(f"\n=== {ck} | {DRIVER}/{PERTURB}/{'det' if DET else 'sto'} | stack={N_STACK} ===")
print(f"телепортов: {len(rows)} | возвратов: {len(oks)} "
      f"({100.0*len(oks)/max(1,len(rows)):.0f}%)")
if oks:
    print(f"медиана шагов возврата: {int(np.median([r['steps'] for r in oks]))}")
if bg_agree:
    print(f"ФОН (до возмущения):        agree={np.mean(bg_agree):.3f}  P(label)={np.mean(bg_plabel):.3f}")
a3 = [r["agree3"] for r in rows if r["agree3"] != ""]
p3 = [r["plabel3"] for r in rows if r["plabel3"] != ""]
if a3:
    print(f"ПОСТ-ВОЗМУЩЕНИЕ (3 шага):   agree={np.mean(a3):.3f}  P(label)={np.mean(p3):.3f}")
a10 = [r["agree10"] for r in rows if r["agree10"] != ""]
if a10:
    print(f"ПОСТ-ВОЗМУЩЕНИЕ (10 шагов): agree={np.mean(a10):.3f}")
if PERTURB == "jump" and oks and len(rows) > len(oks):
    c_ok = [r["cos"] for r in oks if r["cos"] != ""]
    c_bad = [r["cos"] for r in rows if not r["ok"] and r["cos"] != ""]
    if c_ok and c_bad:
        print(f"P5 cos(смещение, шар): успехи {np.mean(c_ok):+.2f} vs фейлы {np.mean(c_bad):+.2f} "
              f"(0 = направление не влияет)")
if bg_far:
    print(f"ФОН далёко (>25px, подход):     agree={np.mean(bg_far):.3f} (n={len(bg_far)})")
if bg_near:
    print(f"ФОН вплотную (<6px, микрокадры): agree={np.mean(bg_near):.3f} (n={len(bg_near)})")
if rows:
    out = (f"servo_{DRIVER}_{PERTURB}_{'det' if DET else 'sto'}_"
           f"{os.path.basename(ck).replace('.zip','')}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("записано:", out)