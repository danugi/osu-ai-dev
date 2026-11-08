# coords.py — доп-каналы стека: курсор, цель, кнопки как ВХОД сети.
#
# Причина появления: пять прогонов (BC, А-кольцо, Б-чанки, В-плотность+тп)
# упираются в cl nav 66-68px; far_probe показал, что BCNet не извлекает из
# пикселей направление «курсор -> цель» (bin ~случая в полосе 25-100px,
# aux-голова мертва даже под прямой супервизией). Аудит master добавил:
# состояние кнопок btn_l/btn_r в кадре НЕТ, а метки кнопок от него зависят
# (неоднозначность учитель->сеть). Кнопки/тайминг из пикселей работают,
# сломан прицел + скрытое состояние кнопок.
#
# Устройство стека (HWC, как во всём проекте):
#   каналы [0..7]  — игровые кадры, старые -> новые (без изменений);
#   далее ТОЛЬКО ТЕКУЩИЕ плоскости (не на каждый кадр — экономия памяти):
#     cursor — квадрат 5x5 яркости 255 в позиции курсора;
#     goal   — квадрат 7x7 в позиции env._goal() (кружок/голова/тело
#              слайдера/центр спиннера — то, куда ведёт оракул);
#     line   — (в *line-режимах) линия курсор->цель толщиной 1;
#     btn_l  — вся плоскость 255, если левая зажата (иначе 0);
#     btn_r  — вся плоскость 255, если правая зажата.
#   История курсора и так видна в игровых кадрах (draw_cursor), история
#   кнопок частично видна по последствиям (объекты исчезают по клику).
#
# Режимы (n_stack = 8 + extras): none=8 (старая линия), dots=10,
# dotsline=11, full=12 (+btn_l+btn_r), fullline=13.
# Режим определяется ПО МОДЕЛИ (каналы первой свёртки): все инструменты
# кита зовут coords.make_pusher(n_stack) и работают с любым режимом.
#
# Живая среда: курсор и цель доступны из tosu-стейта (оракул их читает),
# вход переносим на бой без «читерства» — карты известны заранее.
import numpy as np
import cv2

from vision import to_canvas, IMG_W, IMG_H

N_GAME = 8                                   # игровых кадров в стеке
MODES = {"none": 0, "dots": 2, "dotsline": 3, "full": 4, "fullline": 5}
MODE_BY_STACK = {N_GAME + v: k for k, v in MODES.items()}


def n_stack_for(mode):
    return N_GAME + MODES[mode]


def _plane_pos(x_osu, y_osu, h, w):
    """osu-координаты -> (row, col) в разрешении obs (с учётом полей canvas)."""
    cx, cy = to_canvas(x_osu, y_osu)         # canvas 320x240 -> (col, row)
    return int(round(cy * h / IMG_H)), int(round(cx * w / IMG_W))


def _blob(plane, row, col, size, val=255):
    r = size // 2
    h, w = plane.shape
    plane[max(0, row - r):min(h, row + r + 1),
          max(0, col - r):min(w, col + r + 1)] = val


def coord_planes(env, mode, h, w):
    """(h, w, n_extra) uint8: [cursor, goal, (line,) btn_l, btn_r] сейчас."""
    ne = MODES[mode]
    out = np.zeros((h, w, ne), np.uint8)
    r0, c0 = _plane_pos(env.cursor_x, env.cursor_y, h, w)
    _blob(out[:, :, 0], r0, c0, 5)
    r1 = c1 = None
    g = env._goal()
    if g is not None:
        r1, c1 = _plane_pos(g[0], g[1], h, w)
        _blob(out[:, :, 1], r1, c1, 7)
    k = 2
    if "line" in mode:
        if r1 is not None:
            cv2.line(out[:, :, k], (c0, r0), (c1, r1), 255, 1)
        k += 1
    if "full" in mode:                       # full / fullline: кнопки
        if getattr(env, "btn_l", False):
            out[:, :, k] = 255
        if getattr(env, "btn_r", False):
            out[:, :, k + 1] = 255
    return out


def push(stack, frame, env, mode):
    """Шаг стека: сдвиг игровых кадров, доп-каналы — из текущего env."""
    out = np.empty_like(stack)
    out[:, :, :N_GAME - 1] = stack[:, :, 1:N_GAME]
    out[:, :, N_GAME - 1] = frame[:, :, 0]
    if MODES[mode]:
        out[:, :, N_GAME:] = coord_planes(env, mode, stack.shape[0], stack.shape[1])
    return out


def stack0(obs, env, mode):
    """Стартовый стек: 8 копий первого кадра + текущие доп-каналы."""
    base = np.repeat(obs, N_GAME, axis=2)
    if not MODES[mode]:
        return base
    return np.concatenate(
        [base, coord_planes(env, mode, obs.shape[0], obs.shape[1])], axis=2)


def make_pusher(n_stack):
    """-> (mode, push(stack, frame, env), stack0(obs, env)) для любого стека.
    8 каналов — старая линия (поведение идентично прежнему коду), 10/11/12/13
    — coords-режим. Единая точка правды для ВСЕХ инструментов кита."""
    mode = MODE_BY_STACK.get(n_stack)
    if mode is None:
        raise SystemExit(
            f"coords: неподдерживаемое число каналов стека {n_stack} "
            f"(ожидались 8/10/11/12/13)")
    if mode == "none":
        return ("none",
                lambda st, fr, env: np.concatenate([st[:, :, 1:], fr], axis=2),
                lambda ob, env: np.repeat(ob, n_stack, axis=2))
    return (mode,
            lambda st, fr, env: push(st, fr, env, mode),
            lambda ob, env: stack0(ob, env, mode))


# ============ PPO-совместимость: обёртка среды с полным стеком ============
def make_wrapped_venv(maps, mode, n_envs=1):
    """DummyVecEnv + VecTransposeImage над CoordStackEnv: obs (n_stack,h,w).
    Заменяет связку VecFrameStack+VecTransposeImage из net.bcnet_to_ppo для
    ЛЮБОГО режима (включая none — семантика стека та же: новые кадры в конец)."""
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    from sim_env import SimOsuEnv

    class CoordStackEnv(gym.Wrapper):
        """SimOsuEnv -> (h, w, N_GAME+extras): стек ведётся внутри."""

        def __init__(self, env, mode):
            super().__init__(env)
            self.mode = mode
            h, w = env.observation_space.shape[:2]
            self.observation_space = spaces.Box(
                0, 255, (h, w, n_stack_for(mode)), np.uint8)
            self._stack = None

        def reset(self, **kw):
            obs, info = self.env.reset(**kw)
            self._stack = stack0(obs, self.env, self.mode)
            return self._stack.copy(), info

        def step(self, action):
            obs, r, term, trunc, info = self.env.step(action)
            self._stack = push(self._stack, obs, self.env, self.mode)
            return self._stack.copy(), r, term, trunc, info

    return VecTransposeImage(DummyVecEnv(
        [lambda: Monitor(CoordStackEnv(SimOsuEnv(maps), mode)) for _ in range(n_envs)]))


def bcnet_to_ppo_wrap(net, maps, n_stack, dev, out):
    """BCNet -> PPO-zip через coords-обёртку (любой режим стека).
    Перенос весов тот же, что в net.bcnet_to_ppo; отличий семантики нет."""
    from stable_baselines3 import PPO
    mode = MODE_BY_STACK[n_stack]
    venv = make_wrapped_venv(maps, mode)
    model = PPO("CnnPolicy", venv, n_steps=512, batch_size=512,
                learning_rate=5e-5, ent_coef=0.0, verbose=0, device=dev)
    pol = model.policy
    sd = net.state_dict()

    assert len(pol.mlp_extractor.policy_net) == 0, "net_arch не пуст"
    assert pol.action_net.weight.shape == (32, 512)
    assert pol.value_net.weight.shape == (1, 512)

    pol.features_extractor.cnn.load_state_dict(
        {k[len("cnn."):]: v for k, v in sd.items() if k.startswith("cnn.")})
    pol.features_extractor.linear.load_state_dict(
        {"0.weight": sd["linear.weight"], "0.bias": sd["linear.bias"]})
    import torch
    with torch.no_grad():
        pol.action_net.weight[:27].copy_(sd["move_head.weight"].to(dev))
        pol.action_net.bias[:27].copy_(sd["move_head.bias"].to(dev))
        pol.action_net.weight[27:].copy_(sd["btn_head.weight"].to(dev))
        pol.action_net.bias[27:].copy_(sd["btn_head.bias"].to(dev))
        pol.value_net.weight.copy_(sd["value_head.weight"].to(dev))
        pol.value_net.bias.copy_(sd["value_head.bias"].to(dev))
    model.save(out)
    venv.close()
    return out
