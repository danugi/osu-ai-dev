import glob
import numpy as np
import cv2
from stable_baselines3 import PPO
from sim_env import SimOsuEnv

MODEL = "checkpoints_stage1new/stage1new_3000000_steps.zip"
OUT = "sim_episode.mp4"
SPEED = 0.5          # 0.5 = замедлить вдвое, 1.0 = реальная скорость
EP_SEED = 1001       # тот же сид, что в diagnose — эпизод 1, её лучший

model = PPO.load(MODEL, device="cpu")
maps = sorted(glob.glob("sim_maps_eval/*.osu"))
env = SimOsuEnv(maps)

obs, _ = env.reset(seed=EP_SEED)
s = np.repeat(obs, 4, axis=2)

# 160x120 слишком мелкий для просмотра — апскейлим x4
vw, vh = env.obs_shape[1] * 4, env.obs_shape[0] * 4
vwrt = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), 30 * SPEED, (vw, vh), isColor=False)

frame_delay = int(round(1.0 / SPEED))   # при 0.5 каждый кадр пишем дважды
done, frames = False, 0
while not done and frames < 30 * 60:     # минута максимум
    a, _ = model.predict(s, deterministic=False)
    obs, r, term, trunc, info = env.step(a)
    s = np.concatenate([s[:, :, 1:], obs], axis=2)

    big = cv2.resize(obs[:, :, 0], (vw, vh), interpolation=cv2.INTER_NEAREST)
    for _ in range(frame_delay):
        vwrt.write(big)
    frames += 1
    done = term or trunc

vwrt.release()
print(f"Готово: {OUT} | {info}")