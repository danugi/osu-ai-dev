import os
import sys
import glob
import time
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

from sim_env import SimOsuEnv

try:                      # монитор необязателен: без него обучение не меняется
    import ai_telemetry
except Exception:         # noqa: BLE001
    ai_telemetry = None

N_ENVS = 10
TB_LOG_DIR = "logs_sim"
# SB3: eval_freq/save_freq считаются в ВЫЗОВАХ vec-env.
# 20_000 вызовов * 10 env = 200k шагов среды между событиями.
EVAL_EVERY = 20_000
N_STACK = 4
try:
    import tqdm   # noqa: F401
    import rich   # noqa: F401
    PROGRESS_BAR = True
except ImportError:
    PROGRESS_BAR = False


# ============================================================
# EvalCallback с отдельной кривой в TensorBoard
# ============================================================
class TierEvalCallback(EvalCallback):
    def __init__(self, *args, tb_prefix="eval", **kwargs):
        super().__init__(*args, **kwargs)
        self._prefix = tb_prefix
        stamp = time.strftime("%m%d_%H%M%S")
        self._writer = SummaryWriter(
            log_dir=os.path.join(TB_LOG_DIR, f"{tb_prefix}_{stamp}")
        )

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._writer.add_scalar(
                f"{self._prefix}/mean_reward",
                self.last_mean_reward,
                self.num_timesteps,
            )
            self._writer.flush()
        return result

    def _on_training_end(self) -> None:
        self._writer.close()


# ============================================================
# ЗАЩИТА ОТ САМОРАЗРУШЕНИЯ: ранний стоп, если PPO ест BC
# ============================================================
class GuardCallback(CheckpointCallback):
    """CheckpointCallback + следит за ep_rew_mean: если обучение
    монотонно валит награду с высокого старта — глушим сами,
    не дав PPO доесть политику (урок 400k-прогона)."""
    def __init__(self, *args, patience=12, min_improve=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self._hist = []
        self._patience = patience      # итераций на оценку тренда
        self._min_improve = min_improve
        self._ep_rew = None

    def _on_step(self) -> bool:
        # подглядываем ep_rew_mean из логгера (как это делает сам SB3)
        buf = self.model.ep_info_buffer
        if len(buf) >= 50:                      # среднее по последним эпизодам
            self._ep_rew = float(np.mean([e["r"] for e in buf]))
        result = super()._on_step()
        if self._ep_rew is not None and self.n_calls % self.save_freq == 0:
            self._hist.append(self._ep_rew)
            n = len(self._hist)
            if n >= self._patience:
                first, last = self._hist[-self._patience], self._ep_rew
                if last < first - self._min_improve:
                    print(f"[GUARD] ep_rew_mean {first:.1f} -> {last:.1f} "
                          f"за {self._patience} чекпоинтов — PPO валит политику.")
                    print(f"[GUARD] Сохраняю и выходим.")
                    self.model.save(f"{self.name_prefix}_guard")
                    return False
        return result


# ============================================================
# ТЕЛЕМЕТРИЯ ДЛЯ МОНИТОРА (опционально, ничего не ломает)
# ============================================================
class TelemetryCallback(BaseCallback):
    """Раз в ~0.5с пишет loss/шаг/среднюю награду в telemetry_*.json.
    Если монитор не запущен (ai_telemetry is None) — пустая операция."""
    def __init__(self, model_name, verbose=0):
        super().__init__(verbose)
        self.model_name = model_name
        self._last = 0.0

    def _on_step(self) -> bool:
        if ai_telemetry is None:
            return True
        now = time.time()
        if now - self._last < 0.5:
            return True
        self._last = now
        lv = self.model.logger.name_to_value if self.model.logger else {}
        loss = lv.get("train/policy_gradient_loss")
        if loss is None:
            loss = lv.get("train/value_loss")
        buf = self.model.ep_info_buffer
        ep_rew = float(np.mean([e["r"] for e in buf])) if len(buf) else None
        ai_telemetry.publish(
            "train_stage",
            ai_status="RUNNING", training=True, model=self.model_name,
            model_loaded=True, train_steps=int(self.model.num_timesteps),
            loss=(float(loss) if loss is not None else None),
            episode_reward=ep_rew,
        )
        return True


# ============================================================
# СБОРКА СРЕД
# ============================================================
def build_pool(spec):
    """'dir:вес,dir2:вес' -> список путей (вес = частота сэмплинга)."""
    paths = []
    for part in spec.split(","):
        d, _, w = part.partition(":")
        d = d.strip()
        w = int(w) if w else 1
        maps = sorted(glob.glob(os.path.join(d, "*.osu")))
        if not maps:
            raise SystemExit(f"В папке {d} нет .osu файлов")
        paths.extend(maps * w)
        print(f"  {d}: {len(maps)} карт × вес {w}")
    return paths


def make_vec(paths, n, hr=0.0, teleport=0.0):
    env = SubprocVecEnv(
        [lambda: Monitor(SimOsuEnv(paths, hr_prob=hr, teleport_prob=teleport)) for _ in range(n)]
    )
    env = VecFrameStack(env, n_stack=N_STACK, channels_order="last")  # 60HZ: было 4
    env = VecTransposeImage(env)
    return env


# ============================================================
# ИНТЕРАКТИВНЫЙ РЕЖИМ
# ============================================================
def ask(prompt, default=""):
    tail = f" [{default}]" if default != "" else ""
    s = input(f"{prompt}{tail}: ").strip()
    return s if s else default


def ask_int(prompt, default):
    while True:
        try:
            return int(ask(prompt, str(default)))
        except ValueError:
            print("  Нужно целое число, например 5000000")


def ask_float(prompt, default):
    while True:
        try:
            return float(ask(prompt, str(default)))
        except ValueError:
            print("  Нужно число, например 0.15")


def interactive():
    print("=== train_stage — интерактивный режим ===\n")

    map_dirs = sorted(
        d for d in os.listdir(".")
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.osu"))
    )
    if map_dirs:
        print("Папки с картами :", ", ".join(map_dirs))
    else:
        print("!! Папок с .osu не видно — запускай из корня проекта")

    weights = (
        sorted(glob.glob("bc_policy*.zip"))
        + sorted(glob.glob("*_final.zip"))
        + sorted(glob.glob("*_guard.zip"))
        + sorted(glob.glob("*_interrupted.zip"))
        + sorted(glob.glob("best_*/best_model.zip"))
        + sorted(glob.glob("checkpoints_*/*_steps.zip"))
    )
    if weights:
        print("Найденные веса  :", ", ".join(weights[-8:]))
    print()

    default_pool = ("sim_maps:1,sim_maps_t1:1"
                    if "sim_maps" in map_dirs and "sim_maps_t1" in map_dirs
                    else ("sim_maps:1" if "sim_maps" in map_dirs else "sim_maps:1"))
    default_eval = "sim_maps_eval" if "sim_maps_eval" in map_dirs else (
        map_dirs[0] if map_dirs else "sim_maps_eval")
    default_old = "sim_maps_eval_t1" if "sim_maps_eval_t1" in map_dirs else ""

    pool = ask("Пул карт (папка:вес,...)", default_pool)
    eval_new = ask("Eval нового тира (папка)", default_eval)
    eval_old = ask("Eval старого тира [нет]", default_old) or None
    resume = ask("Веса для старта (.zip), Enter = с нуля", "") or None
    steps = ask_int("Шагов", 4_000_000)
    lr = ask_float("Learning rate (пик)", 1e-5)
    name = ask("Имя стадии", "stage2")
    hr = ask_float("HR-вероятность", 0.0)

    if resume and not os.path.exists(resume):
        raise SystemExit(f"Файл весов не найден: {resume}")

    cmd = (f'python train_stage.py --pool "{pool}" --eval-new {eval_new}'
           + (f" --eval-old {eval_old}" if eval_old else "")
           + (f" --resume {resume}" if resume else "")
           + f" --steps {steps} --lr {lr} --name {name} --hr {hr}")
    print(f"\nЭквивалентная команда:\n  {cmd}\n")
    input("Enter — поехали (Ctrl+C — отмена). ")

    return argparse.Namespace(
        pool=pool, eval_new=eval_new, eval_old=eval_old,
        resume=resume, steps=steps, lr=lr, name=name, hr=hr,
    )


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True, help='"sim_maps:1,sim_maps_t1:1"')
    p.add_argument("--eval-new", required=True, help="папка карт нового тира (eval)")
    p.add_argument("--eval-old", default=None, help="папка старого тира (контроль забывания)")
    p.add_argument("--resume", default=None, help="веса прошлой стадии (.zip)")
    p.add_argument("--steps", type=int, default=4_000_000)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="пик learning rate; спад до 2e-6 в последние 5%%")
    p.add_argument("--name", default="stage2")
    p.add_argument("--hr", type=float, default=0.0, help="вероятность HR-аугментации")
    args = p.parse_args() if len(sys.argv) > 1 else interactive()

    # --- карты ---
    paths = build_pool(args.pool)
    eval_new_maps = sorted(glob.glob(os.path.join(args.eval_new, "*.osu")))
    if not eval_new_maps:
        raise SystemExit(f"Нет .osu карт в {args.eval_new}")

    # --- среды ---
    env = make_vec(paths, N_ENVS, hr=args.hr, teleport=0.004)   # ← добавь teleport=0.004

    callbacks = [
        GuardCallback(
            save_freq=EVAL_EVERY,
            save_path=f"checkpoints_{args.name}",
            name_prefix=args.name,
        ),
        TelemetryCallback(args.name),
        TierEvalCallback(
            make_vec(eval_new_maps, 1),
            best_model_save_path=f"best_{args.name}",
            eval_freq=EVAL_EVERY,
            n_eval_episodes=10,
            deterministic=False,
            tb_prefix=f"eval_{args.name}",
        ),
    ]

    if args.eval_old:
        eval_old_maps = sorted(glob.glob(os.path.join(args.eval_old, "*.osu")))
        if eval_old_maps:
            callbacks.append(
                TierEvalCallback(
                    make_vec(eval_old_maps, 1),
                    eval_freq=3 * EVAL_EVERY,   # старый тир реже: контроль, не метрика
                    n_eval_episodes=10,
                    deterministic=False,
                    tb_prefix="eval_old",
                )
            )
        else:
            print(f"[!] В {args.eval_old} нет .osu — контроль забывания выключен")

    # --- модель ---
    if args.resume:
        print(f"Продолжаем с {args.resume}")
        model = PPO.load(
            args.resume,
            env=env,
            custom_objects={
                # ТИХИЙ СТАРТ (урок 400k-прогона): острый BC (0.999) ломается
                # на 5e-5. Пик 1e-5 + clip 0.1 + kl 0.02.
                "learning_rate": lambda prog: 2e-6 + (args.lr - 2e-6) * min(1.0, prog / 0.05),
                "ent_coef": 0.001,
                "clip_range": 0.1,
                "n_epochs": 4,
                "target_kl": 0.02,
            },
            tensorboard_log=TB_LOG_DIR,
            verbose=1,
        )
    else:
        model = PPO(
            "CnnPolicy", env,
            learning_rate=1e-4,
            n_steps=512,
            batch_size=512,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            target_kl=0.03,
            verbose=1,
            tensorboard_log=TB_LOG_DIR,
            device="auto",
        )

    # --- обучение ---
    try:
        model.learn(
            args.steps,
            callback=callbacks,
            progress_bar=PROGRESS_BAR,
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        print("\n[!] Прервано — сохраняю текущие веса...")
        model.save(f"{args.name}_interrupted")
        if ai_telemetry is not None:
            ai_telemetry.clear("train_stage")
        raise SystemExit(0)

    model.save(f"{args.name}_final")
    if ai_telemetry is not None:
        ai_telemetry.clear("train_stage")
    print(f"\nГотово: {args.name}_final.zip | лучший на eval: best_{args.name}/best_model.zip")