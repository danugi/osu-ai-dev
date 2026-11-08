# -*- coding: utf-8 -*-
# diagnose_v3.py — v2 + инструменты разбора «BC мёртв, оракул жив» (шаги 0/0b аудита).
#
# ЧТО НОВОГО ОТНОСИТЕЛЬНО diagnose_v2:
#   1. model = ORACLE | RANDOM — политика подменяется оракулом/случайной (калибровка
#      потолка и пола среды; оракул и так гоняется через тот же возмущённый пайплайн).
#   2. 6-й аргумент clean теперь НАСТОЯЩИЙ чистый: lag=0, scale=1.0, press_off=0,
#      noise=0 (в v2/sim_env clean оставлял press_off=10.0 — мисматч с BC-сбором был и там).
#   3. --warmup N — прогрев стека: вместо 8 копий первого кадра стек заполняется N
#      кадрами, отрендеренными в прошлом (sim_time - k*DT_MS). Лечит «холодный стек».
#   4. --sweep scale|lag|press — сетка по одной оси (остальные чистые): порог отказов.
#   5. Per-object журнал (кружки + головы слайдеров): aim_err_px, timing_err_ms,
#      first_miss, макс. каскад промахов, recovery — видно, КАК политика умирает.
#
# СОВМЕСТИМОСТЬ: старый вызов даёт те же числа, что v2:
#   python diagnose_v3.py model.zip sim_maps 0 30 det rand
# ПОРЯДОК РАЗБОРА (см. README кита):
#   python diagnose_v3.py bc_policy_fresh.zip sim_maps 0 30 det clean            # 0a
#   python diagnose_v3.py bc_policy_fresh.zip sim_maps 0 30 det clean --warmup 7 # 0b
#   python diagnose_v3.py bc_policy_fresh.zip sim_maps 0 30 det rand --warmup 7  # 0c
#   python diagnose_v3.py ORACLE sim_maps 0 30 det rand                          # 1 (контроль)
#   python diagnose_v3.py bc_policy_fresh.zip sim_maps 0 30 det clean --sweep scale  # 0d
import argparse
import csv
import glob
import math
import os
import sys
from bisect import bisect_left, bisect_right

import numpy as np

from core import DT_MS, decode_move
from sim_env import SimOsuEnv
from oracle import oracle_policy

try:
    from stable_baselines3 import PPO
    from models import infer_stack
except ImportError:            # орakul/RANDOM работают без torch/sb3
    PPO, infer_stack = None, None


# ---------------------------------------------------------------- аргументы
def parse_args():
    p = argparse.ArgumentParser(
        description="diagnose_v3: как v2 + clean/warmup/sweep/per-object метрики",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", nargs="?", default="bc_policy.zip",
                   help="model.zip | ORACLE | RANDOM")
    p.add_argument("maps", nargs="?", default="sim_maps_eval")
    p.add_argument("n_stack", nargs="?", type=int, default=0,
                   help="0 = авто из модели")
    p.add_argument("n", nargs="?", type=int, default=30)
    p.add_argument("mode", nargs="?", default="det", choices=["det", "sto"])
    p.add_argument("domain", nargs="?", default="rand", choices=["rand", "clean"])
    p.add_argument("--warmup", type=int, default=0,
                   help="кадров прогрева стека (7 = весь стек живой); 0 = как v2")
    p.add_argument("--sweep", choices=["scale", "lag", "press"], default=None)
    p.add_argument("--sweep-grid", default=None,
                   help="своя сетка, напр. '0.98,0.99,1.0,1.01' (для lag: '0,1,2')")
    p.add_argument("--sweep-n", type=int, default=10, help="сидов на точку сетки")
    p.add_argument("--csv", default=None)
    return p.parse_args()


# ---------------------------------------------------------------- политика
def make_policy(name, det, seed0):
    """-> callable(env, stack) -> action. Экономит импорт torch для ORACLE/RANDOM."""
    if name.upper() == "ORACLE":
        return lambda env, s: np.array(oracle_policy(env), dtype=np.int64), "ORACLE", 0
    if name.upper() == "RANDOM":
        rng = np.random.default_rng(seed0)
        def random_policy(env, s):
            return np.array([int(rng.integers(0, 27)), int(rng.integers(0, 5))],
                            dtype=np.int64)
        return random_policy, "RANDOM", 0
    if PPO is None:
        raise SystemExit("модель требует stable_baselines3 (нет в окружении)")
    model = PPO.load(name, device="cpu")
    n_stack = infer_stack(model)
    def policy(env, s):
        a, _ = model.predict(s, deterministic=det)
        return a
    return policy, os.path.basename(name), n_stack


# ---------------------------------------------------------------- стек
def build_stack(env, obs0, n_stack, warmup):
    """Стек на старте. warmup=0 → repeat (как v2/bc_train). warmup=N>0 → N кадров
    прошлого (рендер при sim_time-k*DT_MS, состояние не трогаем) + текущий."""
    if warmup <= 0:
        return np.repeat(obs0, n_stack, axis=2)
    t0 = env.sim_time
    frames = []
    for k in range(warmup, 0, -1):
        env.sim_time = max(0.0, t0 - k * DT_MS)
        frames.append(env._render())
    env.sim_time = t0
    frames.append(obs0)
    frames = frames[-n_stack:]
    if len(frames) < n_stack:
        pad = np.repeat(frames[0], n_stack - len(frames), axis=2)
        return np.concatenate([pad] + frames, axis=2)
    return np.concatenate(frames, axis=2)


# ---------------------------------------------------------------- эпизод
def run_episode(policy, maps, seed, det, clean, warmup, n_stack,
                override=None):
    """Один сегмент. override — dict вида {'scale': 1.02, ...}: выставляется
    после reset (для sweep), остальные оси чистые."""
    env = SimOsuEnv(maps, domain_rand=not clean)
    obs, _ = env.reset(seed=seed)
    start_t = env.sim_time
    if clean:                        # НАСТОЯЩИЙ чистый (v2 оставлял press_off=10)
        env._lag, env._scale, env._press_off, env._noise = 0, 1.0, 0.0, 0.0
    if override:
        for k, v in override.items():
            setattr(env, "_%s" % k, v)

    # --- per-object журнал: хуки судейства ---
    hits, presses = {}, []           # i -> (pt, head); [(pt, cx, cy)]
    _jh, _jp = env._judge_hit, env._judge_press
    def jh(i, pt, head):
        hits[i] = (pt, head); _jh(i, pt, head)
    def jp(pt):
        presses.append((pt, env.cursor_x, env.cursor_y)); _jp(pt)
    env._judge_hit, env._judge_press = jh, jp
    min_aim = {}                     # i -> мин. дистанция курсор-центр в окне

    cut0 = bisect_right(env.object_times, start_t)   # объекты до старта — skip
    s = build_stack(env, obs, n_stack, warmup)

    clicks, prev_l, overlapped = 0, False, set()
    done, info = False, {}
    while not done:
        a = policy(env, s)
        obs, r, term, trunc, info = env.step(a)
        s = np.concatenate([s[:, :, 1:], obs], axis=2)
        done = term or trunc
        if env.btn_l and not prev_l:
            clicks += 1
        prev_l = env.btn_l
        t, m = env.sim_time, env.cur
        lo = bisect_left(env.object_times, t - m["w50"])
        hi = bisect_right(env.object_times, t + m["w50"])
        for i in range(lo, hi):
            o, st = env.hitobjects[i], env.obj_state[i]
            if o["type"] == 0 and not st["judged"]:
                d = math.hypot(o["x"] - env.cursor_x, o["y"] - env.cursor_y)
                if i in min_aim:
                    min_aim[i] = min(min_aim[i], d)
                else:
                    min_aim[i] = d
                if d <= m["r_osu"]:
                    overlapped.add(i)
            elif o["type"] == 1 and st["head"] is None:
                d = math.hypot(o["x"] - env.cursor_x, o["y"] - env.cursor_y)
                min_aim[i] = min(min_aim.get(i, 1e9), d)

    # --- per-object записи: кружки + головы слайдеров ---
    m = env.cur
    recs = []
    for i in range(cut0, len(env.hitobjects)):
        o, st = env.hitobjects[i], env.obj_state[i]
        t_obj = o["time"]
        if o["type"] == 0:
            if not st["judged"]:            # trunc оборвал эпизод до судейства
                continue
            hit = i in hits and not hits[i][1]
        elif o["type"] == 1:
            if st["head"] is None:
                continue                    # не судён (trunc)
            hit = i in hits and hits[i][1]
        else:
            continue                        # спиннеры в журнал не входят
        h = hits.get(i)
        clicked = any(abs(pt - t_obj) <= m["w50"] for pt, _, _ in presses)
        recs.append({
            "i": i, "type": o["type"], "t_obj": t_obj, "hit": bool(hit),
            "clicked": clicked,
            "aim": min_aim.get(i, float("nan")),
            "tim": (h[0] - t_obj) if h else float("nan"),
        })

    row = {
        "seed": seed, "map": os.path.basename(env.cur_map_path),
        "start_t": round(start_t), "lag": env._lag,
        "scale": round(env._scale, 3), "press_off": round(env._press_off, 1),
        "acc": info.get("acc", 0.0), "miss": info.get("miss", 0),
        "clicks": clicks, "over_obj": len(overlapped),
        "r_osu": round(m["r_osu"], 1),
    }
    row.update(journal_stats(recs))
    return row


def journal_stats(recs):
    """Per-object агрегаты по кружкам/головам."""
    circles = [r for r in recs if r["type"] == 0]
    n = len(recs)
    if not n:
        return {"j_total": 0, "j_hits": 0, "aim_med": float("nan"),
                "aim_p90": float("nan"), "tim_med": float("nan"),
                "first_miss": -1, "cascade": 0, "recovery": float("nan")}
    hits_n = sum(1 for r in recs if r["hit"])
    aims = sorted(r["aim"] for r in circles if r["aim"] == r["aim"])
    tims = sorted(r["tim"] for r in recs if r["tim"] == r["tim"])
    first_miss = next((k for k, r in enumerate(recs) if not r["hit"]), -1)
    cascade = cur = 0
    for r in recs:
        cur = 0 if r["hit"] else cur + 1
        cascade = max(cascade, cur)
    if 0 <= first_miss < n - 1:
        tail = recs[first_miss + 1:]
        recovery = sum(1 for r in tail if r["hit"]) / len(tail)
    else:
        recovery = float("nan")
    def pct(xs, q):
        return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))] if xs else float("nan")
    return {"j_total": n, "j_hits": hits_n,
            "aim_med": round(pct(aims, 50), 1), "aim_p90": round(pct(aims, 90), 1),
            "tim_med": round(pct(tims, 50), 1) if tims else float("nan"),
            "first_miss": first_miss, "cascade": cascade,
            "recovery": round(recovery, 2) if recovery == recovery else float("nan")}


# ---------------------------------------------------------------- вывод
def med(xs):
    xs = [x for x in xs if x == x]
    return float(np.median(xs)) if xs else float("nan")


def print_summary(rows, title):
    accs = np.array([r["acc"] for r in rows], dtype=float)
    print(f"\n{title}")
    print(f"acc = {accs.mean():.3f} ± {accs.std():.3f} | miss: med={med([r['miss'] for r in rows]):.0f} "
          f"max={max(r['miss'] for r in rows)} | кликов: med={med([r['clicks'] for r in rows]):.0f} "
          f"| над объектом: med={med([r['over_obj'] for r in rows]):.0f}")
    r_osu = med([r["r_osu"] for r in rows])
    aim_m, aim_p = med([r["aim_med"] for r in rows]), med([r["aim_p90"] for r in rows])
    print(f"журнал: hits med={med([r['j_hits'] for r in rows]):.0f}/{med([r['j_total'] for r in rows]):.0f} "
          f"| aim med/p90 = {aim_m:.0f}/{aim_p:.0f} osu-px (радиус кружка ~{r_osu:.0f}) "
          f"| timing med = {med([r['tim_med'] for r in rows]):.0f} ms")
    print(f"первый промах: объект #{med([r['first_miss'] for r in rows]):.0f} "
          f"| макс. каскад med={med([r['cascade'] for r in rows]):.0f} "
          f"| recovery med={med([r['recovery'] for r in rows]):.2f}")
    cat = [r for r in rows if r["acc"] < 0.3]
    print(f"катастроф (acc<0.3): {len(cat)}/{len(rows)}"
          + (f" — сиды {[r['seed'] for r in cat]}" if cat else ""))
    # авто-интерпретация
    tips = []
    if accs.mean() < 0.3:
        if aim_m > 3 * r_osu:
            tips.append(f"прицел: медианная дистанция до кружков {aim_m:.0f}px при радиусе {r_osu:.0f}px — "
                        "курсор систематически не доходит (пространственный отказ)")
        if med([r["first_miss"] for r in rows]) <= 2:
            tips.append("срыв с самых первых объектов (first_miss<=2): похоже на холодный стек/старт, "
                        "прогоните с --warmup 7 и сравните")
        rec = med([r["recovery"] for r in rows])
        if rec == rec and rec < 0.2:
            tips.append(f"recovery {rec:.2f}: после первого промаха политика почти не возвращается — "
                        "классический covariate shift (то, что лечит DAgger)")
        if med([r["clicks"] for r in rows]) == 0:
            tips.append("кликов нет вообще: отказ кнопок, а не прицела")
    else:
        tips.append("acc>=0.3: политика жива в этом режиме")
    for t in tips:
        print("  »", t)


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("записано:", path)


# ---------------------------------------------------------------- sweep
def default_grid(axis):
    return {"scale": [0.92, 0.95, 0.97, 0.99, 1.0, 1.01, 1.02, 1.03, 1.05, 1.08],
            "lag": [0, 1, 2, 3, 4],
            "press": [0, 5, 10, 15, 20, 30, 50]}[axis]


def run_sweep(policy, maps, args, n_stack):
    grid = ([float(x) for x in args.sweep_grid.split(",")] if args.sweep_grid
            else default_grid(args.sweep))
    print(f"sweep по {args.sweep} (остальные оси чистые), {args.sweep_n} сидов на точку, "
          f"warmup={args.warmup}")
    print(f"{'значение':>8} | {'acc':>6} | {'aim med':>8} | {'перв.промах':>11} | {'acc>=0.3':>8}")
    out = []
    for g in grid:
        accs, aims, fms = [], [], []
        for ep in range(args.sweep_n):
            row = run_episode(policy, maps, 1000 + ep, args.mode == "det", True,
                              args.warmup, n_stack,
                              override={args.sweep: (int(g) if args.sweep == "lag" else g)})
            accs.append(row["acc"]); aims.append(row["aim_med"]); fms.append(row["first_miss"])
        a = float(np.mean(accs))
        print(f"{g:>8} | {a:>6.3f} | {med(aims):>8.0f} | {med(fms):>11.0f} | "
              f"{sum(1 for x in accs if x >= 0.3):>4}/{args.sweep_n}")
        out.append({"axis": args.sweep, "value": g, "acc": round(a, 3),
                    "aim_med": round(med(aims), 1), "first_miss": round(med(fms), 1),
                    "n_ok": sum(1 for x in accs if x >= 0.3), "n": args.sweep_n})
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader(); w.writerows(out)
        print("записано:", args.csv)


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    maps = sorted(glob.glob(os.path.join(args.maps, "*.osu")))
    assert maps, "нет карт в " + args.maps

    policy, name, auto_stack = make_policy(args.model, args.mode == "det", 1000)
    n_stack = args.n_stack or auto_stack or 8
    if args.model.upper() not in ("ORACLE", "RANDOM"):
        print(f"стек модели: {n_stack} кадров")
    tag = (f"{args.model if args.model.upper() in ('ORACLE', 'RANDOM') else name} | {args.maps} "
           f"| stack={n_stack} | {args.mode} | {args.domain} | warmup={args.warmup} | n={args.n}")

    if args.sweep:
        run_sweep(policy, maps, args, n_stack)
        return

    rows = []
    for ep in range(args.n):
        seed = 1000 + ep
        row = run_episode(policy, maps, seed, args.mode == "det",
                          args.domain == "clean", args.warmup, n_stack)
        rows.append(row)
        print({k: row[k] for k in ("seed", "map", "start_t", "lag", "scale",
                                   "press_off", "acc", "miss", "clicks", "over_obj")},
              "| aim", row["aim_med"], "| 1st_miss", row["first_miss"],
              "| casc", row["cascade"], "| rec", row["recovery"])

    print_summary(rows, tag)
    out = args.csv or f"diagnose_v3_{name.replace('.zip', '')}_{args.mode}_{args.domain}" \
                      f"{'_w' + str(args.warmup) if args.warmup else ''}.csv"
    write_csv(rows, out)


if __name__ == "__main__":
    main()
