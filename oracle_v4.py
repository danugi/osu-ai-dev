# oracle_v4.py — учитель v4: машина состояний кнопок (исправление аудита).
#
# ЧТО СЛОМАНО В v3 (oracle.py, master):
#   1) btn=4 («отпустить правую») НЕ ВЫДАЁТСЯ НИКОГДА: после btn=3 правая
#      остаётся зажатой до конца эпизода; класс 4 не имеет обучающих меток.
#   2) btn=0 тоже не выдаётся: в простое всегда 2 («отпустить левую»), даже
#      когда левая уже отпущена (но-op). Метка «ничего не делать» отсутствует.
#   Следствие для потока кликов: после 3 обе кнопки зажаты -> ветка else=2
#   съедает кадр на отпускание Л, и до конца эпизода клики идут только Л.
#
# v4 (только блок кнопок; вся логика прицела/целей — без изменений):
#   want_click:  не зажата Л -> 1; не зажата П -> 3; обе зажаты -> 2
#                (освободить Л, следующий клик = 1; П отпустится в простое)
#   want_hold:   Л не зажата -> 1; Л зажата и П зажата -> 4 (П не нужна);
#                иначе -> 0
#   простой:     зажата П -> 4 (первой отпускаем вспомогательную);
#                зажата Л -> 2; иначе -> 0
# Все 5 классов получают метки; края нажатий сохранены (1/3 -> press в
# _apply_buttons только по фронту). _predict_pending уже умеет 4 (br=False).
#
# Совместимость: MOVE-метки идентичны v3 (блок прицела не тронут), поэтому
# far_probe-замеры старых и новых моделей сравнимы. BTN-метки отличаются —
# модели, обученные на v3, НЕ переоценивать по btn-agree против v4-учителя.
import math

import numpy as np

from core import decode_move, DT_MS
from vision import OSU_W, OSU_H, slider_pos


# ============Predict_pending: дословно из oracle.py v3 ============
def _predict_pending(env):
    px, py, bl, br = env.cursor_x, env.cursor_y, env.btn_l, env.btn_r
    has_press = False
    pending = list(env._action_q)[-env._lag:] if env._lag else []
    for a in pending:
        if a is None:
            continue
        aa = np.asarray(a).flatten()
        d, sp = decode_move(int(aa[0]))
        px = min(OSU_W, max(0.0, px + d[0] * sp * env._scale))
        py = min(OSU_H, max(0.0, py + d[1] * sp * env._scale))
        b = int(aa[1])
        if b == 1:
            has_press = has_press or not bl
            bl = True
        elif b == 2:
            bl = False
        elif b == 3:
            has_press = has_press or not br
            br = True
        elif b == 4:
            br = False
    return px, py, bl, br, has_press


def _best_move(px, py, tx, ty, scale):
    best, bd = 4, math.hypot(tx - px, ty - py)
    for midx in range(27):
        d, sp = decode_move(midx)
        nx = min(OSU_W, max(0.0, px + d[0] * sp * scale))
        ny = min(OSU_H, max(0.0, py + d[1] * sp * scale))
        dist = math.hypot(tx - nx, ty - ny)
        if dist < bd:
            bd, best = dist, midx
    return best, bd


def oracle_policy(env):
    """Учитель v4 = v3 с исправленными кнопками (см. шапку)."""
    t = env.sim_time
    L, m = env._lag, env.cur
    px, py, bl, br, has_press = _predict_pending(env)

    t_obj_i, hold_i = None, None
    for i in range(env._expire_ptr, min(len(env.hitobjects), env._expire_ptr + 8)):
        o, st = env.hitobjects[i], env.obj_state[i]
        if o["type"] == 0 and not st["judged"]:
            if t_obj_i is None: t_obj_i = i
        elif o["type"] == 1:
            if st["head"] is None:
                if t_obj_i is None: t_obj_i = i
            elif (st["head"] in ("300", "100", "50")
                  and not st["final"]
                  and env.sim_time < o["time"]):
                if hold_i is None: hold_i = i
    t_obj = env.hitobjects[t_obj_i] if t_obj_i is not None else None

    if env._open_sliders:
        o = env.hitobjects[env._open_sliders[-1]]
        tx, ty = slider_pos(o, t + (L + 1) * DT_MS)
    elif hold_i is not None:
        o = env.hitobjects[hold_i]
        tx, ty = float(o["x"]), float(o["y"])
    elif env._open_spinners:
        a = math.atan2(py - 192.0, px - 256.0) + 1.5
        tx, ty = 256.0 + 45.0 * math.cos(a), 192.0 + 45.0 * math.sin(a)
    elif t_obj is not None:
        tx, ty = float(t_obj["x"]), float(t_obj["y"])
    else:
        g = env._goal()
        tx, ty = g if g is not None else (px, py)

    midx, _ = _best_move(px, py, tx, ty, 1.0)
    d, sp = decode_move(midx)
    ax = min(OSU_W, max(0.0, px + d[0] * sp * 1.0))
    ay = min(OSU_H, max(0.0, py + d[1] * sp * 1.0))

    on_obj = (t_obj is not None and
              math.hypot(t_obj["x"] - ax, t_obj["y"] - ay) <= m["r_osu"])
    in_window = (t_obj is not None and -30 < t - t_obj["time"] < 22)
    want_click = on_obj and in_window and not has_press
    pending_slider_hold = (has_press and t_obj is not None and t_obj["type"] == 1)
    want_hold = bool(env._open_sliders or env._open_spinners
                     or hold_i is not None or pending_slider_hold)

    # --- v4: машина состояний кнопок (было: 1/2/3 без 4 и 0) ---
    if want_click:
        if not bl:   btn = 1          # нажать Л
        elif not br: btn = 3          # нажать П (Л занята)
        else:        btn = 2          # обе зажаты: освободить Л, след. клик = 1
    elif want_hold:
        if not bl:   btn = 1          # держать Л
        elif br:     btn = 4          # Л держим, П не нужна — отпустить
        else:        btn = 0          # всё уже в нужном состоянии
    else:
        if br:       btn = 4          # простой: первой отпускаем П
        elif bl:     btn = 2          # затем Л
        else:        btn = 0          # нечего отпускать
    return midx, btn


def aux_target(env):
    """Нормированный вектор (курсор -> текущая цель) — дословно v3."""
    g = env._goal()
    if g is None:
        return 0.0, 0.0
    return (g[0] - env.cursor_x) / 512.0, (g[1] - env.cursor_y) / 384.0
