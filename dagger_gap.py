#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dagger_gap.py — измеритель DAgger-разрыва: главная улика «поможет ли DAgger».

Считаем поведение модели в двух режимах на ОДНИХ картах:
  teacher-forced: оракул ведёт эпизод, модель лишь предсказывает   -> agree_tf
  closed-loop:    модель ведёт сама, оракул размечает её состояния -> agree_cl
Разрыв — количественная мера covariate shift.

ВЕРДИКТ v2 — по НАВЕДЕНИЮ и acc, не по exact-agree (27 классов действий
содержат близкие эквиваленты: у здоровой модели agree 0.15–0.30 — это норма,
а не «не копирует учителя»). Радиус попадания r_osu = 54.4-4.48*CS ≈ 32-41px,
поэтому «подражание ок» = наведение med ≤ 15px (пол-радиуса).
acc в teacher-forced режиме всегда ≈ 1.0 по построению (исполняет оракул) —
это санити-чек пайплайна, а не метрика модели.

УЧИТЕЛЬ (--teacher, дефолт v4): метки кнопок у oracle_v4 и oracle (v3)
РАЗНЫЕ (v4: машина состояний, все 5 классов; v3: без 0 и 4). Модель линии
bc3+ меряется против v4; веса СТАРОЙ линии (bc_policy*, dagger2* до bc3) —
против v3, иначе agree btn будет мусорным числом (пример: bc3_policy против
v3-учителя дал 0.037 — это артефакт меток, не качество кнопок). MOVE-метки
у v3/v4 идентичны, agree move сравним в обоих режимах. Дополнительно
печатается btn recall по классам 0..4 (доля верных срабатываний на метках
класса) — прямая метрика клик-триггера (класс 1 = нажать Л).

Запуск:
  python dagger_gap.py model.zip --maps sim_maps --steps 4000
  python dagger_gap.py model_bcnet.pt --maps sim_maps --steps 4000   # без PPO
  python dagger_gap.py ORACLE --maps sim_maps --steps 500   # санити: разрыв 0
  python dagger_gap.py old_model.pt --teacher v3 ...         # веса до bc3
"""
import argparse
import glob
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_env import SimOsuEnv
from core import decode_move
from models import infer_stack_file
from coords import make_pusher


def load_teacher(which):
    """v4 = oracle_v4 (линия bc3+), v3 = oracle (старая линия)."""
    if which == "v4":
        from oracle_v4 import oracle_policy
    else:
        from oracle import oracle_policy
    return oracle_policy

# пороги вердикта v2 (osu-px)
NAV_TF_OK = 15.0   # подражание учителю: nav med на его траекториях
NAV_CL_OK = 25.0   # замкнутый цикл жив: nav med на своих состояниях
ACC_LIVE = 0.5     # ... и acc эпизодов не ниже этого
ACC_DEAD = 0.1     # ниже — «цикл мёртв» (при здоровом подражании = сдвиг)


def load_predictor(name, det, oracle_policy):
    """-> (policy(env, stack) -> [midx, btn], метка). .zip через PPO,
    .pt напрямую через BCNet (обходит вопросы нормализации SB3)."""
    if name.upper() == "ORACLE":
        return lambda env, s: np.array(oracle_policy(env), dtype=np.int64), "ORACLE"
    if name.endswith(".pt"):
        import torch
        from net import BCNet
        ns = infer_stack_file(name)
        net = BCNet(n_stack=ns)
        net.load_state_dict(torch.load(name, map_location="cpu"))
        net.eval()
        def policy_pt(env, s):
            x = torch.from_numpy(s).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
            with torch.no_grad():
                lm, lb, _, _ = net(x)
            return np.array([int(lm.argmax()), int(lb.argmax())], dtype=np.int64)
        return policy_pt, f"{os.path.basename(name)} (.pt, стек {ns})"
    from stable_baselines3 import PPO
    model = PPO.load(name, device="cpu")
    def policy(env, s):
        a, _ = model.predict(s, deterministic=det)
        return np.asarray(a).flatten()
    return policy, os.path.basename(name)


def measure(maps, policy, driver, steps, seed=2024, act_noise=0.03,
           n_stack=8, lag_p=(0.65, 0.35, 0.0), oracle_policy=None):
    """driver: 'oracle' | 'policy'. Возвращает per-step записи."""
    env = SimOsuEnv(maps, domain_rand=True)
    obs, _ = env.reset(seed=seed)
    env._lag = int(env.np_random.choice(3, p=lag_p))
    env._press_off = 10.0
    _, pushf, s0f = make_pusher(n_stack)     # coords-стек по числу каналов
    stack = s0f(obs, env)
    recs = []
    for step in range(steps):
        midx, btn = oracle_policy(env)             # метка — всегда оракул
        pred = policy(env, stack)                  # предсказание модели
        pm, pb = int(pred[0]), int(pred[1])
        d1, s1 = decode_move(midx)
        d2, s2 = decode_move(pm)
        v1 = np.array([d1[0] * s1, d1[1] * s1], dtype=np.float64)
        v2 = np.array([d2[0] * s2, d2[1] * s2], dtype=np.float64)
        recs.append({"agree_move": pm == midx, "agree_btn": pb == btn,
                     "btn_y": btn, "btn_p": pb,
                     "nav_err_px": float(np.hypot(*(v1 - v2)))})
        # исполнение
        if driver == "oracle":
            exec_midx, exec_btn = midx, btn
            if env.np_random.random() < act_noise:
                exec_midx = int(env.np_random.integers(0, 27))
        else:
            exec_midx, exec_btn = pm, pb
        obs, r, term, trunc, info = env.step(np.array([exec_midx, exec_btn], dtype=np.int64))
        stack = pushf(stack, obs, env)
        if info:
            recs[-1]["acc"] = info.get("acc")
        if term or trunc:
            obs, _ = env.reset()
            env._lag = int(env.np_random.choice(3, p=lag_p))
            env._press_off = 10.0
            stack = s0f(obs, env)
    return recs


def _sgn(x):
    """Разрыв к печати: падение с минусом, рост с плюсом (без ‘−-0.011’)."""
    return f"−{x:.3f}" if x >= 0 else f"+{-x:.3f}"


def _btn_recall(recs):
    """Строка per-class recall кнопок: доля предсказаний == метке внутри
    класса, с числом меток. '—' = меток класса нет."""
    ys = np.array([r["btn_y"] for r in recs])
    ps = np.array([r["btn_p"] for r in recs])
    parts = []
    for c in range(5):
        m = ys == c
        parts.append(f"{np.mean(ps[m] == c):.2f}({m.sum()})" if m.sum() else "—")
    return " ".join(parts)


def report(name, recs, label):
    am = np.mean([r["agree_move"] for r in recs])
    ab = np.mean([r["agree_btn"] for r in recs])
    nes = [r["nav_err_px"] for r in recs]
    accs = [r["acc"] for r in recs if "acc" in r]
    ne, ne90 = float(np.median(nes)), float(np.percentile(nes, 90))
    acc = float(np.mean(accs)) if accs else None
    print(f"[{label}] n={len(recs)} | agree move={am:.3f} | agree btn={ab:.3f} "
          f"| ошибка наведения med={ne:.0f}px p90={ne90:.0f}px"
          + (f" | acc эпизодов={acc:.3f}" if acc is not None else ""))
    print(f"    btn recall 0..4 (меток): {_btn_recall(recs)}   "
          f"[0=ничего, 1=нажатьЛ, 2=отпуститьЛ, 3=нажатьП, 4=отпуститьП]")
    return {"am": float(am), "ab": float(ab), "ne": ne, "ne90": ne90, "acc": acc}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--maps", default="sim_maps")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--lag-p", default="0.65,0.35,0",
                   help="распределение лага 0/1/2 (дефолт — как старый bc_train, "
                        "для сравнимости; линия bc3: 0.4,0.35,0.25)")
    p.add_argument("--teacher", choices=["v4", "v3"], default="v4",
                   help="учитель МЕТОК: v4 = oracle_v4 (линия bc3+, дефолт); "
                        "v3 = oracle (веса СТАРОЙ линии до bc3 — иначе "
                        "agree btn будет артефактом разницы меток)")
    args = p.parse_args()
    lag_p = tuple(float(x) for x in args.lag_p.split(","))
    assert len(lag_p) == 3 and abs(sum(lag_p) - 1.0) < 1e-6, "--lag-p: три доли, сумма 1"

    maps = sorted(glob.glob(os.path.join(args.maps, "*.osu")))
    assert maps, "нет карт в " + args.maps
    oracle_policy = load_teacher(args.teacher)
    policy, name = load_predictor(args.model, det=True, oracle_policy=oracle_policy)
    n_stack = 8 if args.model.upper() == "ORACLE" else infer_stack_file(args.model)

    print(f"=== DAgger-разрыв: {name} | {args.maps} | {args.steps} шагов "
          f"| лаг {lag_p} | учитель oracle_{args.teacher} ===")
    tf = report(name, measure(maps, policy, "oracle", args.steps, seed=args.seed,
                              n_stack=n_stack, lag_p=lag_p,
                              oracle_policy=oracle_policy),
                "teacher-forced: оракул ведёт")
    cl = report(name, measure(maps, policy, "policy", args.steps, seed=args.seed,
                              n_stack=n_stack, lag_p=lag_p,
                              oracle_policy=oracle_policy),
                "closed-loop: политика ведёт сама")

    print(f"\nРАЗРЫВ: move {tf['am']:.3f} -> {cl['am']:.3f} ({_sgn(tf['am']-cl['am'])}) | "
          f"btn {tf['ab']:.3f} -> {cl['ab']:.3f} ({_sgn(tf['ab']-cl['ab'])})")
    ratio = cl["ne"] / max(tf["ne"], 1e-9)
    acc_line = (f" | acc {tf['acc']:.3f} -> {cl['acc']:.3f}"
                if tf["acc"] is not None and cl["acc"] is not None else "")
    print(f"НАВЕДЕНИЕ: med {tf['ne']:.0f} -> {cl['ne']:.0f}px (x{ratio:.1f})"
          + acc_line)

    if args.model.upper() == "ORACLE":
        print("санити: обе стороны — оракул, разрыв должен быть 0")
        return
    acc_cl = cl["acc"] if cl["acc"] is not None else 0.0
    if tf["ne"] > NAV_TF_OK:
        print(f"ВЕРДИКТ: подражание отсутствует или ПОТЕРЯНО (наведение med "
              f"{tf['ne']:.0f}px > {NAV_TF_OK:.0f}px). Если это DAgger-итерация и раньше "
              "было лучше (3–9px) — итерация переобучает буфер (replay-ratio 64): "
              "переходи на dagger_train2 (чанки x эпохи).")
    elif cl["ne"] <= NAV_CL_OK and acc_cl >= ACC_LIVE:
        print("ВЕРДИКТ: замкнутый цикл ЖИВ — разрыв закрыт.")
    elif acc_cl < ACC_DEAD:
        print(f"ВЕРДИКТ: подражание учителю есть (med {tf['ne']:.0f}px), но замкнутый "
              f"цикл мёртв (med {cl['ne']:.0f}px, acc {acc_cl:.3f}) — сдвиг состояний. "
              "DAgger показан / продолжай итерации.")
    else:
        print(f"ВЕРДИКТ: частичный прогресс (acc {acc_cl:.3f}, наведение med "
              f"{cl['ne']:.0f}px) — разрыв закрывается, но не закрыт; продолжать.")


if __name__ == "__main__":
    main()
