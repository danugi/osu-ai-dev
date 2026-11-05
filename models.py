# models.py — реестр моделей: поиск .zip, авто-стек из первой свёртки
# политики, инвентаризация (бывший census.py).
import glob
import os

# Порядок = порядок пунктов меню в pick_model (cli.py).
PATTERNS = ("bc_policy*.zip", "*_final.zip", "*_guard.zip",
            "*_interrupted.zip", "best_*/*.zip", "checkpoints_*/*.zip")


def find_models():
    """Все известные .zip-модели проекта."""
    cands = []
    for pat in PATTERNS:
        cands += sorted(glob.glob(pat))
    return cands


def latest_model():
    """Свежий .zip по дате изменения (для play / record_sim / debug_seed)."""
    cands = find_models()
    return max(cands, key=os.path.getmtime) if cands else None


def infer_stack(model):
    """Сколько кадров ждёт сеть: 8 = 60Гц-линия, 4 = эксперимент стек-4.
    Читаем прямо из первой свёртки политики, чтобы раннер НИКОГДА
    не расходился с тренировкой (краш октября: Stacker(4) против стек-8)."""
    try:
        return int(model.policy.features_extractor.cnn[0].in_channels)
    except Exception:
        sh = tuple(model.observation_space.shape)
        if sh[1:] == (120, 160):
            return int(sh[0])      # CHW: (8,120,160)
        if sh[:2] == (120, 160):
            return int(sh[2])      # HWC: (120,160,8)
        return 8                   # текущая 60Гц-линия


def infer_stack_file(path):
    """Стек сети по файлу: .zip (PPO-модель) или .pt (state_dict BCNet).
    train_stage/dagger_train берут стек из весов, а не из константы."""
    if path.endswith(".pt"):
        import torch
        sd = torch.load(path, map_location="cpu")
        return int(sd["cnn.0.weight"].shape[1])
    from stable_baselines3 import PPO
    return infer_stack(PPO.load(path, device="cpu"))


if __name__ == "__main__":
    # Инвентаризация (бывший census.py): obs-форма -> стек, число шагов.
    from stable_baselines3 import PPO
    for p in sorted(glob.glob("*.zip") + glob.glob("best_*/*.zip")
                    + glob.glob("checkpoints_*/*.zip")):
        try:
            m = PPO.load(p, device="cpu")
            print(f"{p:55s} obs={tuple(m.observation_space.shape)} steps={m.num_timesteps}")
        except Exception as e:
            print(f"{p:55s} FAIL {e}")
