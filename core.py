# core.py — общие константы живой среды и симулятора, один источник правды.
# env.py, sim_env.py, bc_train.py и инструменты читают их отсюда, поэтому
# тренировочный стек (sim -> bc/dagger/train_stage) не тянет за собой
# Windows-код env.py и работает на любой платформе.

TOSU_URL = "http://127.0.0.1:24050/json/v2"

ACTIONS_PER_SECOND = 60          # 60HZ: было 30
DT_MS = 1000.0 / ACTIONS_PER_SECOND   # длительность одного шага среды, мс
MOVE_DISTANCES = [3, 12, 48]     # 60HZ: было [6, 24, 96]; скорости сохранены
DIRS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]

OBS_SCALE = 0.5

R_300, R_100, R_50, R_MISS, R_SB = 1.0, 0.4, 0.1, -1.0, -0.5
R_STEP = -0.002
DENSE_W = 0.25


def decode_move(midx):
    """Индекс комбинированного действия -> (направление, дистанция).
    Кодировка: speed_idx * 9 + dir_idx (экшн-спейс env/sim)."""
    d = DIRS[midx % len(DIRS)]
    speed = MOVE_DISTANCES[min(len(MOVE_DISTANCES) - 1, midx // len(DIRS))]
    return d, speed
