import glob
from stable_baselines3 import PPO
for p in sorted(glob.glob("*.zip") + glob.glob("best_*/*.zip") + glob.glob("checkpoints_*/*.zip")):
    try:
        m = PPO.load(p, device="cpu")
        print(f"{p:55s} obs={tuple(m.observation_space.shape)} steps={m.num_timesteps}")
    except Exception as e:
        print(f"{p:55s} FAIL {e}")