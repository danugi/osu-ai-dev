import glob, os, sys, csv, math
import numpy as np
from bisect import bisect_left, bisect_right
from stable_baselines3 import PPO
from sim_env import SimOsuEnv
from cli import ask, ask_int, pick_model, pick_maps_dir
from models import infer_stack

# diagnose v2: стек авто/параметром, форензика (карта/старт/lag/scale/press_off).
# Без аргументов — интерактив. С аргументами:
#   python diagnose_v2.py model.zip [maps_dir] [n_stack=0авто] [n_seeds] [det|sto]


if len(sys.argv) > 1:
    ck = sys.argv[1]
    maps_dir = sys.argv[2] if len(sys.argv) > 2 else "sim_maps_eval"
    N_STACK = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # 0 = авто из модели
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    DET = not (len(sys.argv) > 5 and sys.argv[5] == "sto")
    RAND = not (len(sys.argv) > 6 and sys.argv[6] == "clean")
else:
    print("=== diagnose_v2 — интерактивный режим ===")
    ck = pick_model("bc_policy.zip")
    maps_dir = pick_maps_dir("sim_maps_eval")
    N_STACK = 0                 # авто из модели
    N = ask_int("Сидов", 30)
    DET = ask("Режим: argmax (det) или сэмплинг (sto)", "det") == "det"
    RAND = ask("Домен: rand или clean", "rand") == "rand"

model = PPO.load(ck, device="cpu")
if not N_STACK:
    N_STACK = infer_stack(model)
    print(f"стек модели: {N_STACK} кадров")
maps = sorted(glob.glob(os.path.join(maps_dir, "*.osu")))
assert maps, "нет карт в " + maps_dir

rows = []
for ep in range(N):
    seed = 1000 + ep
    env = SimOsuEnv(maps, domain_rand=RAND)                      # domain_rand=True — как у оракула
    obs, _ = env.reset(seed=seed)
    start_t = round(env.sim_time)
    s = np.repeat(obs, N_STACK, axis=2)
    total, done, prev_l, clicks, overlapped = 0.0, False, False, 0, set()
    while not done:
        a, _ = model.predict(s, deterministic=DET)
        obs, r, term, trunc, info = env.step(a)
        s = np.concatenate([s[:, :, 1:], obs], axis=2)
        total += float(r); done = term or trunc
        if env.btn_l and not prev_l:
            clicks += 1
        prev_l = env.btn_l
        t, m = env.sim_time, env.cur
        lo = bisect_left(env.object_times, t - m["w50"])
        hi = bisect_right(env.object_times, t + m["w50"])
        for i in range(lo, hi):
            o, st = env.hitobjects[i], env.obj_state[i]
            if o["type"] == 0 and not st["judged"] and \
               math.hypot(o["x"]-env.cursor_x, o["y"]-env.cursor_y) <= m["r_osu"]:
                overlapped.add(i)
    rows.append({"seed": seed, "map": os.path.basename(env.cur_map_path),
                 "start_t": start_t, "lag": env._lag,
                 "scale": round(env._scale, 3), "press_off": round(env._press_off, 1),
                 "acc": info.get("acc", 0), "miss": info.get("miss", 0),
                 "clicks": clicks, "over_obj": len(overlapped)})
    print(rows[-1])

accs = np.array([r["acc"] for r in rows])
misses = [r["miss"] for r in rows]
cat = [r for r in rows if r["acc"] < 0.3]
mode = "det" if DET else "sto"
print(f"\n{ck} | {maps_dir} | stack={N_STACK} | {mode} | n={N}")
print(f"acc = {accs.mean():.3f} ± {accs.std():.3f} | miss: med={np.median(misses):.0f} "
      f"max={max(misses)} | кликов: med={np.median([r['clicks'] for r in rows]):.0f} "
      f"| над объектом: med={np.median([r['over_obj'] for r in rows]):.0f}")
print(f"катастроф (acc<0.3): {len(cat)}/{N}"
      + (f" — сиды {[r['seed'] for r in cat]}" if cat else ""))
if cat:
    print("  условия катастроф:")
    for k in ("map", "lag", "scale", "press_off", "start_t"):
        print(f"    {k}: {', '.join(str(r[k]) for r in cat)}")
print("для сравнения: оракул на этих же сидах 0.995 (eval) / 0.998 (t1)")

out = f"diagnose_v2_{os.path.basename(ck).replace('.zip','')}_{mode}.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("записано:", out)