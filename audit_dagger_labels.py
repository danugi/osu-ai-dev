#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_dagger_labels.py — аудит пайплайна разметки (шаг 2 плана): проверяем,
что метки (obs -> действие оракула), которые получит DAgger/BC, осмысленны.

Почему так: буферы bc_train/dagger_train живут в RAM и на диск не пишутся,
поэтому скрипт умеет (а) сам собрать свежую порцию РОВНО той же семантикой,
что bc_train (оракул-метка, исполнение=метка+3% шум, lag {0,1}, press_off=10),
и (б) тут же её аудировать. Режим (в) --npz — для дампа вашего боевого буфера.

ПРОВЕРКИ:
  A. ЦЕЛОСТНОСТЬ СТЕКА: для последовательных сэмплов одного эпизода кадр k
     сэмпла i+1 должен совпадать с кадром k+1 сэмпла i (ловит off-by-one).
  B. СОГЛАСОВАННОСТЬ МЕТКИ С ЦЕЛЬЮ: cos(вектор движения из метки, aux-вектор
     курсор->цель) по движениям (не «стоять»); мусорные метки дают ~0.
  C. ТАЙМИНГ КНОПОК: для нажатий d = (t + press_off) - t_obj ближайшего
     объекта; медиана/разброс покажут, компенсирован ли press_off в метке.
  D. КОНТАКТ-ЛИСТ PNG: последние кадры стека + стрелка метки + aux-стрелка +
     маркер клика и цель — глазами видно мусор.
  E. СЧЁТ ОРАКУЛА за прогон сбора (по info, как в diagnose).

Запуск:
  python audit_dagger_labels.py --maps sim_maps --collect 8000
  python audit_dagger_labels.py --npz buffer_dump.npz
"""
import argparse
import glob
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_env import SimOsuEnv
from oracle import oracle_policy, aux_target
from core import decode_move

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------- сбор
def collect(maps_dir, n_steps, seed=2024, act_noise=0.03):
    """Сбор ровно как bc_train: метка оракула, исполнение = метка + шум."""
    maps = sorted(glob.glob(os.path.join(maps_dir, "*.osu")))
    assert maps, "нет карт в " + maps_dir
    env = SimOsuEnv(maps, domain_rand=True)
    obs, _ = env.reset(seed=seed)
    env._lag = int(env.np_random.choice(2, p=[0.65, 0.35]))
    env._press_off = 10.0
    n_stack = 8
    stack = np.repeat(obs, n_stack, axis=2)

    obs_l, move_l, btn_l, aux_l = [], [], [], []
    meta_l = []          # (ep, step, t_ms, press_pt, t_obj, obj_x, obj_y)
    ep, last_acc, last_miss = 0, None, None
    for step in range(n_steps):
        midx, btn = oracle_policy(env)
        ax_, ay_ = aux_target(env)
        obs_l.append(stack)
        move_l.append(midx)
        btn_l.append(btn)
        aux_l.append((ax_, ay_))
        # ближайший по времени объект для тайминг-аудита кнопок
        t = env.sim_time
        fut = [o for o in env.hitobjects
               if -80.0 <= (o["time"] - (t + env._press_off)) <= 120.0
               and o["type"] in (0, 1)]
        o = min(fut, key=lambda o: abs(o["time"] - (t + env._press_off))) if fut else None
        press_pt = (t + env._press_off) if btn in (1, 3) else None
        meta_l.append({"ep": ep, "step": step, "t_ms": t,
                       "press_pt": press_pt,
                       "t_obj": (o["time"] if o else None),
                       "obj_x": (o["x"] if o else None),
                       "obj_y": (o["y"] if o else None)})

        exec_midx = midx
        if env.np_random.random() < act_noise:
            exec_midx = int(env.np_random.integers(0, 27))
        obs, r, term, trunc, info = env.step(np.array([exec_midx, btn], dtype=np.int64))
        stack = np.concatenate([stack[:, :, 1:], obs], axis=2)
        if info:
            last_acc, last_miss = info.get("acc"), info.get("miss")
        if term or trunc:
            obs, _ = env.reset()
            env._lag = int(env.np_random.choice(2, p=[0.65, 0.35]))
            env._press_off = 10.0
            stack = np.repeat(obs, n_stack, axis=2)
            ep += 1
    print(f"[сбор] {n_steps} шагов, {ep + 1} эпизодов | оракул за прогон: "
          f"acc={last_acc} miss={last_miss}")
    return (np.stack(obs_l), np.array(move_l), np.array(btn_l),
            np.array(aux_l, dtype=np.float32), meta_l)


# ---------------------------------------------------------------- проверки
def to_hwc(obs):
    """Канонизация: (N,120,160,8) HWC или (N,8,120,160) CHW -> всегда HWC."""
    if obs.ndim == 4 and obs.shape[1] == 8 and obs.shape[3] != 8:
        return np.ascontiguousarray(obs.transpose(0, 2, 3, 1))
    return obs


def check_stack_overlap(obs, meta):
    """A: off-by-one в стеке. Сравниваем сэмплы одного эпизода на соседних шагах.
    obs в HWC: канал -1 = самый свежий кадр (как в буферах bc/dagger_train)."""
    obs = to_hwc(obs)
    checked = bad = 0
    for i in range(1, len(obs)):
        if meta[i]["ep"] == meta[i - 1]["ep"] and meta[i]["step"] == meta[i - 1]["step"] + 1:
            checked += 1
            if not np.array_equal(obs[i][:, :, :-1], obs[i - 1][:, :, 1:]):
                bad += 1
    if not checked:
        print("[A стек] нет последовательных пар (нет meta) — проверка пропущена")
        return
    rate = bad / checked
    verdict = "OK" if rate < 0.01 else ("ПОДОЗРИТЕЛЬНО" if rate < 0.3 else "СЛОМАНО")
    print(f"[A стек] пар {checked} | расхождений {bad} ({rate:.1%}) | {verdict}"
          + ("  <- off-by-one в сборе стека!" if rate >= 0.3 else ""))


def check_label_consistency(obs, move, btn, aux, meta):
    """B: cos(движение метки, aux-вектор к цели) по ненулевым движениям."""
    coses, anti = [], 0
    for i in range(len(move)):
        d, sp = decode_move(int(move[i]))
        if d == (0, 0):
            continue
        v = np.array([d[0] * sp, d[1] * sp], dtype=np.float64)
        a = np.array(aux[i], dtype=np.float64)
        na, nv = np.linalg.norm(a), np.linalg.norm(v)
        if na < 1e-6 or nv < 1e-6:
            continue
        c = float(np.dot(v, a) / (na * nv))
        coses.append(c)
        anti += c < -0.3
    if not coses:
        print("[B метки] нет ненулевых движений — данных мало")
        return
    coses = np.array(coses)
    print(f"[B метки] движений {len(coses)} | cos(метка, цель): med={np.median(coses):+.3f} "
          f"p10={np.percentile(coses, 10):+.3f} | движений ОТ цели (cos<-0.3): {anti} "
          f"({anti / len(coses):.1%})")
    print("        ожидаемо: med > +0.9 (метка ведёт к цели); med около 0 = метки мусорны")


def check_button_timing(btn, meta):
    """C: компенсация press_off. d = (t+press_off) - t_obj для нажатий."""
    ds = [m["press_pt"] - m["t_obj"] for m in meta
          if m["press_pt"] is not None and m["t_obj"] is not None]
    if len(ds) < 10:
        print("[C тайминг] мало нажатий с объектом рядом — пропущено")
        return
    ds = np.array(ds)
    print(f"[C тайминг] нажатий {len(ds)} | d=(t+press_off)-t_obj: med={np.median(ds):+.0f} мс "
          f"| p10={np.percentile(ds, 10):+.0f} | p90={np.percentile(ds, 90):+.0f}")
    print("        med заметно > +30мс = метка кликает слишком рано/поздно относительно "
          "того, как клик придёт в игру")


def contact_sheet(obs, move, btn, aux, meta, path, n=48):
    """D: сетка кадров + стрелки. Синяя = движение метки, зелёная = aux (к цели)."""
    if not HAS_MPL:
        print("[D лист] matplotlib недоступен — пропущено")
        return
    obs = to_hwc(obs)
    idx = np.linspace(0, len(obs) - 1, min(n, len(obs))).astype(int)
    cols = 8
    rows = math.ceil(len(idx) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0), squeeze=False)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for k, i in enumerate(idx):
        ax = axes[k // cols][k % cols]
        ax.imshow(obs[i][:, :, -1], cmap="gray", vmin=0, vmax=255)
        h, w = obs[i].shape[0], obs[i].shape[1]
        cx, cy = w / 2, h / 2
        d, sp = decode_move(int(move[i]))
        v = np.array([d[0] * sp, d[1] * sp], dtype=np.float64)
        a = np.array(aux[i], dtype=np.float64)
        a_px = np.array([a[0] * 512, a[1] * 384]) * (w / 512.0)   # aux в пиксели кадра
        for vec, color, lw in ((v, "deepskyblue", 1.6), (a_px, "limegreen", 1.0)):
            nrm = np.linalg.norm(vec)
            if nrm > 1e-6:
                u = vec / nrm * (0.28 * min(h, w))
                ax.annotate("", xy=(cx + u[0], cy + u[1]), xytext=(cx, cy),
                            arrowprops=dict(arrowstyle="->", color=color, lw=lw))
        if btn[i] in (1, 3):
            ax.add_patch(plt.Circle((cx, cy), 7, fill=False, color="red", lw=1.4))
        m = meta[i]
        if m["obj_x"] is not None:
            ax.plot([m["obj_x"] * w / 512.0], [m["obj_y"] * h / 384.0],
                    marker="x", color="orange", markersize=7)
    fig.suptitle("контакт-лист: синяя стрелка = метка движения, зелёная = цель (aux), "
                 "красный круг = клик, оранжевый x = ближайший объект", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("[D лист]", path)


def summarize(move, btn):
    """E: сводка по действиям."""
    d0 = sum(1 for m in move if decode_move(int(m))[0] == (0, 0))
    print(f"[E сводка] n={len(move)} | click_rate={(btn == 1).mean():.3f} | "
          f"стоять (dir=0,0): {d0 / len(move):.1%} | "
          f"кнопки 0..4: {np.bincount(btn, minlength=5).tolist()}")


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--maps", default="sim_maps")
    p.add_argument("--collect", type=int, default=8000,
                   help="собрать N шагов свежей разметки (как bc_train)")
    p.add_argument("--npz", default=None, help="дамп буфера: obs,move,btn,aux,meta")
    p.add_argument("--out", default="audit_out")
    p.add_argument("--seed", type=int, default=2024)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.npz:
        z = np.load(args.npz, allow_pickle=True)
        obs, move, btn, aux = to_hwc(z["obs"]), z["move"], z["btn"], z["aux"]
        meta = list(z["meta"]) if "meta" in z else [
            {"ep": 0, "step": i, "t_ms": None, "press_pt": None,
             "t_obj": None, "obj_x": None, "obj_y": None} for i in range(len(obs))]
        print(f"[дамп] {len(obs)} сэмплов")
    else:
        obs, move, btn, aux, meta = collect(args.maps, args.collect, seed=args.seed)

    print(f"\n=== АУДИТ РАЗМЕТКИ ({len(obs)} сэмплов) ===")
    check_stack_overlap(obs, meta)
    check_label_consistency(obs, move, btn, aux, meta)
    check_button_timing(btn, meta)
    summarize(move, btn)
    contact_sheet(obs, move, btn, aux, meta, os.path.join(args.out, "contact_sheet.png"))


if __name__ == "__main__":
    main()
