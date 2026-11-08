#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""far_probe.py — однотактный тест стены «далеко -> поверни к цели».

Контекст: пять прогонов (BC, А-кольцо, Б-чанки, В-плотность+целевые телепорты)
упираются в одно и то же: cl nav med 66-68px при живом tf (3-7px). В чанке В
~70% кадров — far-состояния с правильными метками «к цели», плотность как у А.
Последняя стоящая гипотеза: BCNet не выучивает функцию «из далёкого состояния
указать на цель» из пикселей. Этот скрипт меряет её НАПРЯМУЮ, без компаундинга:

  оракул ведёт эпизод (карта/тайминг живые); каждые --tp-every шагов курсор
  телепортируется в случайную точку поля, и кадр с новым курсором дописывается
  в стек (без шага симуляции). На КАЖДОМ шаге спрашиваем модель (один forward,
  argmax; НИЧЕГО от модели не исполняется) и сравниваем:
    - угол(предсказанный ход, метка оракула)      [чему учил DAgger]
    - угол(предсказанный ход, истинное направление на цель)
    - угол(aux-голова, истинное направление)       [только для .pt]
  Всё по бакетам ДИСТАНЦИИ ДО ЦЕЛИ в момент пробы: <25 / 25-50 / 50-100 / 100+.

Запуск (в корне проекта):
  python far_probe.py dagger2_iter2_bcnet.pt                 # полный (с aux)
  python far_probe.py dagger2_iter2.zip                      # без aux-головы
  python far_probe.py dagger2_iter2_bcnet.pt --tp-every 0    # контроль: без телепортов
  python far_probe.py ORACLE                                 # санити: ~0гр везде

ЧТЕНИЕ РЕЗУЛЬТАТА:
  near-бакет должен быть чистым (это tf-режим; большие углы там = баг пробы).
  ГЛАВНЫЙ столбец — «уг.move~метка» (чему учили); «уг.move~цель» вторичен и
  шумнее (оракул иногда целится не в env._goal(), плюс квантование 9 векторов
  на 45гр). Решают far (50-100) и 100+:
    угол(move) med >= 60гр -> направление из far НЕ выучено. Это и есть стена.
       + aux med <= 25гр   -> фичи направление ЗНАЮТ, маппинг нет: дешёвый
                              фикс — aux/координаты на вход move-головы.
       + aux med >= 45гр   -> направление не кодируется вообще: координаты
                              курсора/цели на вход сети + переучить BC.
    угол(move) med <= 30гр -> far-функция выучена, а цикл всё равно мёртв:
                              компаундинг/история стека — копать стек, не данные.
"""
import argparse
import glob
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_env import SimOsuEnv
from oracle import oracle_policy
from core import decode_move, MOVE_DISTANCES, DIRS
from vision import OSU_W, OSU_H
from models import infer_stack_file
from coords import make_pusher

N_DIRS = len(DIRS)                    # 9
BUCKETS = [(0.0, 25.0, "near <25"), (25.0, 50.0, "mid 25-50"),
           (50.0, 100.0, "far 50-100"), (100.0, 1e9, "xfar 100+")]


def load_predictor(name):
    """-> (predict(stack)->(midx, aux|None), метка режима). Дет. argmax всегда."""
    if name.upper() == "ORACLE":
        return (lambda s: (int(oracle_policy(CUR[0])[0]), None)), "ORACLE(санити)"
    if name.endswith(".pt"):
        import torch
        from net import BCNet
        n_stack = infer_stack_file(name)
        net = BCNet(n_stack=n_stack)
        net.load_state_dict(torch.load(name, map_location="cpu"))
        net.eval()

        def pred_pt(stack):
            x = torch.from_numpy(stack).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
            with torch.no_grad():
                lm, _, _, la = net(x)
            return int(lm.argmax()), (float(la[0, 0]), float(la[0, 1]))
        return pred_pt, os.path.basename(name) + " (.pt, aux есть)"
    from stable_baselines3 import PPO
    model = PPO.load(name, device="cpu")

    def pred_zip(stack):
        a, _ = model.predict(stack, deterministic=True)
        return int(np.asarray(a).flatten()[0]), None
    return pred_zip, os.path.basename(name) + " (.zip, aux нет)"


CUR = [None]      # костыль для ORACLE-санити: текущий env для лямбды выше


def ang_err_deg(v1, v2):
    """Угол между 2D-векторами в градусах; None, если какой-то ~нулевой/stand."""
    if v1 is None or v2 is None:
        return None
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    c = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def move_vec(midx):
    """Класс -> вектор движения; (0,0)-направление = «стоять» -> None."""
    d, sp = decode_move(int(midx))
    return (d[0] * sp, d[1] * sp) if d != (0, 0) else None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="model_bcnet.pt | model.zip | ORACLE")
    p.add_argument("--maps", default="sim_maps")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--tp-every", type=int, default=3,
                   help="телепорт каждые N шагов (0 = контроль без телепортов)")
    p.add_argument("--lag-p", default="0.65,0.35,0",
                   help="распределение лага 0/1/2 (по умолчанию — как старый bc_train, "
                        "для сравнимости; новая линия bc3: 0.4,0.35,0.25)")
    args = p.parse_args()
    lag_p = tuple(float(x) for x in args.lag_p.split(","))
    assert len(lag_p) == 3 and abs(sum(lag_p) - 1.0) < 1e-6, "--lag-p: три доли, сумма 1"

    maps = sorted(glob.glob(os.path.join(args.maps, "*.osu")))
    assert maps, "нет карт в " + args.maps
    predict, label = load_predictor(args.model)
    n_stack = 8 if args.model.upper() == "ORACLE" else infer_stack_file(args.model)
    _, pushf, s0f = make_pusher(n_stack)     # coords-стек (8/10/11/12/13 каналов)

    env = SimOsuEnv(maps, domain_rand=True)
    obs, _ = env.reset(seed=args.seed)
    env._lag = int(env.np_random.choice(3, p=lag_p))
    env._press_off = 10.0
    stack = s0f(obs, env)
    CUR[0] = env

    rows = []
    for step in range(args.steps):
        # --- телепорт: курсор в случайную точку, кадр дописываем без шага t ---
        if args.tp_every > 0 and step > 0 and step % args.tp_every == 0:
            env.cursor_x = float(env.np_random.uniform(40, OSU_W - 40))
            env.cursor_y = float(env.np_random.uniform(40, OSU_H - 40))
            frz = env._render()
            stack = pushf(stack, frz, env)

        g = env._goal()
        if g is not None:
            midx, btn = oracle_policy(env)          # метка на ТЕКУЩЕМ состоянии
            pm, pa = predict(stack)                 # мнение модели, 1 forward
            tx, ty = g[0] - env.cursor_x, g[1] - env.cursor_y
            rows.append({
                "dist": math.hypot(tx, ty),
                "lab": int(midx), "pred": int(pm), "aux": pa,
                "true_dir": (tx, ty),
            })

        # --- исполняем всегда ОРАКУЛ: эпизод остаётся живым/типовым ---
        midx, btn = oracle_policy(env)
        obs, r, term, trunc, info = env.step(np.array([midx, btn], dtype=np.int64))
        stack = pushf(stack, obs, env)
        if term or trunc:
            obs, _ = env.reset()
            env._lag = int(env.np_random.choice(3, p=lag_p))
            env._press_off = 10.0
            stack = s0f(obs, env)

    # ------------------------------------------------------------ статистика
    print(f"\n=== far_probe: {label} | {args.maps} | steps={args.steps} "
          f"| tp_every={args.tp_every} | seed={args.seed} ===")
    hdr = (f"{'бакет':>12} | {'n':>5} | {'уг.move~метка':>13} | {'уг.move~цель':>13} "
           f"| {'бин':>5} | {'скор':>5} | {'стоять':>6} | {'уг.aux':>12}")
    print(hdr)
    print("-" * len(hdr))
    far_pool = {"mv": [], "au": []}
    for lo, hi, nm in BUCKETS:
        rs = [r for r in rows if lo <= r["dist"] < hi]
        if not rs:
            print(f"{nm:>12} | {'0':>5} |  —")
            continue
        e_lab = [ang_err_deg(move_vec(r["pred"]), move_vec(r["lab"]))
                 for r in rs]
        e_lab = [e for e in e_lab if e is not None]
        e_tru = [ang_err_deg(move_vec(r["pred"]), r["true_dir"]) for r in rs]
        e_tru = [e for e in e_tru if e is not None]
        e_aux = []
        if rs[0]["aux"] is not None:
            for r in rs:
                t = (r["true_dir"][0] / OSU_W, r["true_dir"][1] / OSU_H)
                e = ang_err_deg(r["aux"], t)
                if e is not None:
                    e_aux.append(e)
        binh = np.mean([decode_move(r["pred"])[0] == decode_move(r["lab"])[0]
                        for r in rs])
        spdh = np.mean([(r["pred"] // N_DIRS) == (r["lab"] // N_DIRS) for r in rs])
        stand = np.mean([decode_move(r["pred"])[0] == (0, 0) for r in rs])
        if lo >= 50:                       # far+xfar -> пул для вердикта
            # (v1.1: было hi >= 50 — затягивало бакет mid 25-50, чей hi ровно 50;
            #  аудит поймал по расхождению n: 3403 строк против 3678 углов)
            far_pool["mv"] += e_lab
            far_pool["au"] += e_aux
        f = (lambda v: f"{np.median(v):5.0f}г p90 {np.percentile(v, 90):5.0f}г"
              if v else "   —     ")
        print(f"{nm:>12} | {len(rs):>5} | {f(e_lab):>13} | {f(e_tru):>13} "
              f"| {binh:5.2f} | {spdh:5.2f} | {stand:6.1%} | "
              f"{(f'{np.median(e_aux):5.0f}г' if e_aux else '—'):>12}")

    # распределение скоростей из far: хватает ли 48
    far_rs = [r for r in rows if r["dist"] >= 50]
    if far_rs:
        sp = np.bincount([r["pred"] // N_DIRS for r in far_rs], minlength=3)
        print(f"скорости модели из far (n={len(far_rs)}): "
              + " / ".join(f"{MOVE_DISTANCES[i]}px {sp[i]/max(1,len(far_rs)):5.1%}"
                           for i in range(3)))

    # --------------------------------------------------------------- вердикт
    print()
    if not far_pool["mv"]:
        print("нет far-проб (мало шагов?) — увеличь --steps")
        return
    mv = float(np.median(far_pool["mv"]))
    print(f"[far+xfar pooled] угол(move~метка) med = {mv:.0f}г "
          f"(n={len(far_pool['mv'])})")
    if mv <= 30:
        print("ВЕРДИКТ: far-функция «повернуть к цели» ВЫУЧЕНА — стена не в ней.")
        print("  Дальше копать компаундинг/историю стека (короткий стек, живой")
        print("  стек на старте, шум исполнения), а не данные/DAgger.")
    elif mv >= 60:
        if far_pool["au"]:
            au = float(np.median(far_pool["au"]))
            print(f"[far+xfar pooled] угол(aux~цель) med = {au:.0f}г")
            if au <= 25:
                print("ВЕРДИКТ: направление НЕ выучено move-головой, но aux-голова")
                print("  его ЗНАЕТ -> фичи кодируют цель, маппинг нет. Дешёвый фикс:")
                print("  aux/координаты на вход move-головы (переучить BC).")
            else:
                print("ВЕРДИКТ: направление не кодируется НИГДЕ из пикселей.")
                print("  Маршрут: координаты курсора/цели на вход сети (доп. каналы")
                print("  или вектор-вход) + переучить BC с нуля; DAgger закрываем.")
        else:
            print("ВЕРДИКТ: направление из far НЕ выучено (>=60г). Для решения по")
            print("  aux-голове прогони на _bcnet.pt (у .zip её нет).")
    else:
        print("ВЕРДИКТ: промежуточная зона (30-60г) — частичное знание; см. оба")
        print("  столбца и aux; вероятно, поможет и вход координат, и больше far-данных.")


if __name__ == "__main__":
    main()
