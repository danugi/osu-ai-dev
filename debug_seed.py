# debug_seed.py — анатомия провального эпизода
import glob, sys, math, os
import numpy as np
import cv2
from bisect import bisect_left, bisect_right
from stable_baselines3 import PPO
from sim_env import SimOsuEnv

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1008
ck = sys.argv[2] if len(sys.argv) > 2 else \
     max(glob.glob("checkpoints_bcft/*_steps.zip"),
         key=lambda p: int(p.split("_")[-2]))
print("Модель:", ck, "| seed:", seed)
model = PPO.load(ck, device="cpu")

maps = sorted(glob.glob("sim_maps_eval/*.osu"))
env = SimOsuEnv(maps)
obs, _ = env.reset(seed=seed)
print(f"lag={env._lag} | scale={env._scale:.2f} | press_off={env._press_off:.0f}ms")

hits, clicks, traj = [], [], []          # шпион на судейство
_orig = env._judge_hit
def _spy(i, pt, head):
    o = env.hitobjects[i]
    hits.append((i, pt - o["time"],
                 math.hypot(o["x"]-env.cursor_x, o["y"]-env.cursor_y)))
    return _orig(i, pt, head)
env._judge_hit = _spy

os.makedirs("debug_frames", exist_ok=True)
s = np.repeat(obs, 4, axis=2)
done, prev_l, step_i, total = False, False, 0, 0.0
while not done:
    t_now, cx, cy = env.sim_time, env.cursor_x, env.cursor_y
    traj.append((t_now, cx, cy))
    a, _ = model.predict(s, deterministic=False)
    obs, r, term, trunc, info = env.step(a)
    if env.btn_l and not prev_l:         # настоящий клик
        clicks.append((t_now, cx, cy))
    prev_l = env.btn_l
    s = np.concatenate([s[:, :, 1:], obs], axis=2)
    total += float(r); step_i += 1; done = term or trunc
    if step_i % 25 == 0:                 # кадры: ~30 шт на эпизод
        cv2.imwrite(f"debug_frames/s{seed}_{step_i:03d}.png",
                    cv2.resize(obs[:, :, 0], (640, 480),
                               interpolation=cv2.INTER_NEAREST))

print(f"reward={total:.1f} | acc={info.get('acc')} | miss={info.get('miss')} "
      f"| кликов={len(clicks)} | хитов={len(hits)}\n")

t0, t1 = traj[0][0], traj[-1][0]
hit_ids = {h[0] for h in hits}
tt = [p[0] for p in traj]
print(f"{'рез':4s} {'время':>7s} {'x':>4s} {'y':>4s} {'дист в окне':>11s}  клики рядом (офсет/дист)")
shown = 0
for i, o in enumerate(env.hitobjects):
    if o["type"] != 0 or not (t0 < o["time"] <= t1):
        continue
    lo = bisect_left(tt, o["time"] - env.cur["w50"])
    hi = bisect_right(tt, o["time"] + env.cur["w50"])
    dmin = min((math.hypot(o["x"]-p[1], o["y"]-p[2]) for p in traj[lo:hi]),
               default=999.0)
    near = [c for c in clicks if abs(c[0] - o["time"]) <= 350]
    st = "HIT " if i in hit_ids else \
         ("MISS" if env.obj_state[i]["judged"] else "OPEN")
    cd = " ".join(f"{c[0]-o['time']:+4.0f}мс/d{math.hypot(o['x']-c[1], o['y']-c[2]):3.0f}"
                  for c in near) or "—"
    print(f"{st:4s} {o['time']:7.0f} {o['x']:4.0f} {o['y']:4.0f} {dmin:10.0f}px  {cd}")
    shown += 1
    if shown >= 20: break   # хочешь больше — подними