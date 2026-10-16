import glob, sys
import numpy as np
from sim_env import SimOsuEnv
from bc_train import oracle_policy

# G0-f: потолок оракула v3 на eval-папке.
# python oracle_eval.py [maps_dir] [rand|clean] [n_seeds]

maps_dir = sys.argv[1] if len(sys.argv) > 1 else "sim_maps_eval_t1"
rand = (sys.argv[2] != "clean") if len(sys.argv) > 2 else True
N = int(sys.argv[3]) if len(sys.argv) > 3 else 30
TP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0  # добавить после разбора аргументов

maps = sorted(glob.glob(maps_dir + "/*.osu"))
assert maps, "нет карт в " + maps_dir

rows = []
for seed in range(1000, 1000 + N):
    env = SimOsuEnv(maps, domain_rand=rand, teleport_prob=TP)  # в цикле по сидам
    env.reset(seed=seed)
    done = False
    while not done:
        midx, btn = oracle_policy(env)
        _, _, term, trunc, info = env.step(np.array([midx, btn], dtype=np.int64))
        done = term or trunc
    rows.append({"seed": seed, "lag": env._lag, "scale": round(env._scale, 3),
                 "press_off": round(env._press_off, 1),
                 "acc": info.get("acc", 0.0), "miss": info.get("miss", 0)})
    print(rows[-1])

accs = np.array([r["acc"] for r in rows])
print(f"\nОРАКУЛ v3 | {maps_dir} | domain_rand={rand} | n={N}")
print(f"acc = {accs.mean():.3f} ± {accs.std():.3f} | катастроф (acc<0.3): {(accs < 0.3).sum()}/{N}")
for lag in (0, 1, 2):
    sub = [r for r in rows if r["lag"] == lag]
    if sub:
        a = np.mean([r["acc"] for r in sub])
        m = np.median([r["miss"] for r in sub])
        print(f"  lag={lag}: acc={a:.3f} miss_med={m:.0f} (n={len(sub)})")