import copy
import math
from bisect import bisect_left, bisect_right
from collections import deque

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import vision
# Импортируем константы ЖИВОЙ среды — экшн-спейс и награды гарантированно совпадают
from env import (ACTIONS_PER_SECOND, MOVE_DISTANCES, DIRS, OBS_SCALE,
                 R_300, R_100, R_50, R_MISS, R_SB, R_STEP, DENSE_W)

DT_MS = 1000.0 / ACTIONS_PER_SECOND

# --- судейство (упрощённое, но последовательное) ---
FOLLOW_RADIUS_MULT = 1.5     # ← stage2c: было 2.4. Тighter follow = офсет на слайдере
                             # БОЛИТ, у PPO появляется градиент учить серво по курсору
R_FOLLOW_STEP = 0.005        # ± за шаг следования по слайдеру
R_OVERLAP = 0.08   # бонус за шаг, когда курсор реально над не судимым ещё кружком
R_SLIDER_END  = 0.3          # бонус за доведённый до конца слайдер
R_SPIN_STEP   = 0.01         # за шаг активного вращения
R_SPIN_END    = 0.5          # выполненный спиннер

# --- домен-рандомизация (главный мост сим -> реал) ---
LAG_STEPS_PROBS = (0.6, 0.3, 0.1)   # задержка ввода: 0/1/2 шага
SCALE_JITTER    = 0.08              # ±8% к дистанции (ошибка калибровки MOUSE_SCALE)
MOVE_NOISE      = 1.5               # гауссов шум на движение, osu-px
PRESS_OFFSET_MS = (5.0, 15.0)       # когда клик реально "приходит" в игру
DT_JITTER       = 0.02              # ±2% к длительности шага


def _windows_from_od(od):
    w300 = max(5.0, 80.0 - 6.0 * od)
    w100 = max(10.0, 140.0 - 8.0 * od)
    w50 = (200.0 - 10.0 * od) if od <= 5.0 else (190.0 - 8.0 * od)
    return w300, w100, w50


class SimOsuEnv(gym.Env):
    """Полностью офлайн-среда: карта + время симулируются, судейство локальное."""

    def __init__(self, map_paths, max_episode_steps=900, domain_rand=True, hr_prob=0.0,
                 teleport_prob=0.0):        # ← stage2c: телепорты курсора (по умолчанию ВЫКЛ —
        super().__init__()                  #    diagnose/eval остаются сравнимы с лестницей)
        if not map_paths:
            raise ValueError("Передай список .osu файлов")
        self.maps = [self._load_map(p) for p in map_paths]
        self.max_episode_steps = max_episode_steps
        self.domain_rand = domain_rand
        self.hr_prob = hr_prob
        self.teleport_prob = teleport_prob   # ← stage2c

        h, w = int(vision.IMG_H * OBS_SCALE), int(vision.IMG_W * OBS_SCALE)
        self.obs_shape = (h, w, 1)
        # РОВНО как в env.py
        self.action_space = spaces.MultiDiscrete([len(DIRS) * len(MOVE_DISTANCES), 5])
        self.observation_space = spaces.Box(0, 255, self.obs_shape, np.uint8)

    def _load_map(self, path):
        objs, ar_ms, cs, w50 = vision.parse_osu_file(path)
        od = (200.0 - w50) / 10.0 if w50 >= 150.0 else (190.0 - w50) / 8.0
        return {"objs": objs, "times": [o["time"] for o in objs],
                "ar_ms": ar_ms, "cs": cs, "od": od, "path": path}   # ← добавить

    def _finalize(self, objs, ar_ms, cs, od):
        w300, w100, w50 = _windows_from_od(od)
        r_osu = 54.4 - 4.48 * cs
        return {
            "objs": objs,
            "times": [o["time"] for o in objs],
            "ar_ms": ar_ms,
            "r_osu": r_osu,
            "cs_radius_px": max(2.0, r_osu * vision.PLAY_W / vision.OSU_W),
            "w300": w300, "w100": w100, "w50": w50,
        }

    # ---------------- gym API ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        vision.ACTIVE_MODS = 0

        base = self.maps[self.np_random.integers(0, len(self.maps))]
        self.cur_map_path = base["path"]                            # ← добавить
        objs = copy.deepcopy(base["objs"])

        # опциональная HR-аугментация (реальная среда HR тоже поддерживает)
        if self.hr_prob > 0 and self.np_random.random() < self.hr_prob:
            ar_ms, cs = vision.apply_mods(objs, base["ar_ms"], base["cs"], vision.MOD_HR)
            self.cur = self._finalize(objs, ar_ms, cs, min(10.0, base["od"] * 1.4))
        else:
            self.cur = self._finalize(objs, base["ar_ms"], base["cs"], base["od"])

        self.hitobjects = self.cur["objs"]
        self.object_times = self.cur["times"]

        # случайный старт внутри карты: больше событий на шаг обучения
        if len(self.hitobjects) > 1:
            idx = self.np_random.integers(0, len(self.hitobjects))
            start_t = max(0.0, self.hitobjects[idx]["time"] - 400.0)
        else:
            start_t = 0.0
        self.sim_time = float(start_t)

        # стартовая позиция курсора случайна — политика обязана уметь
        # «подхватывать» курсор по точке в кадре, а не предполагать центр
        self.cursor_x = float(self.np_random.uniform(40, vision.OSU_W - 40))
        self.cursor_y = float(self.np_random.uniform(40, vision.OSU_H - 40))
        self.btn_l = self.btn_r = False

        # домен-рандомизация на эпизод
        self._lag = int(self.np_random.choice(3, p=LAG_STEPS_PROBS)) if self.domain_rand else 0
        self._scale = 1.0 + self.np_random.uniform(-SCALE_JITTER, SCALE_JITTER) if self.domain_rand else 1.0
        self._press_off = float(self.np_random.uniform(*PRESS_OFFSET_MS)) if self.domain_rand else 10.0
        self._noise = MOVE_NOISE if self.domain_rand else 0.0
        self._action_q = deque(maxlen=3)

        # состояние объектов
        self.obj_state = [{
            "judged": False, "final": False, "head": None,
            "follow_ok": 0, "follow_tot": 0,
            "last_ang": None, "spun": 0.0,
        } for _ in self.hitobjects]
        self.n300 = self.n100 = self.n50 = self.nmiss = self.nsb = 0
        self.n300h = self.n100h = self.n50h = 0   # головы слайдеров (для acc_legacy)
        self.nmiss_c = 0                           # миссы на кружках (разбивка по типам)
        self.nmiss_sp = 0

        # всё, что было ДО точки старта, помечаем пропущенным БЕЗ штрафа
        cut = bisect_right(self.object_times, self.sim_time)
        for i in range(cut):
            self.obj_state[i].update(judged=True, final=True, head="skip")
        self._expire_ptr = cut
        self._slide_ptr = cut
        self._open_sliders, self._open_spinners = [], []

        self._steps = 0
        self._prev_goal = None
        self._reward = 0.0
        return self._render(), {}

    def step(self, action):
        self._reward = R_STEP

        # 1) действие (с учётом задержки ввода)
        act = self._dequeue(action)
        press = False
        if act is not None:
            a = np.asarray(act).flatten()
            midx, btn = int(a[0]), int(a[1])
            d = DIRS[midx % len(DIRS)]
            speed = MOVE_DISTANCES[min(len(MOVE_DISTANCES) - 1, midx // len(DIRS))]
            press = self._apply_buttons(btn)
            self._move(d, speed)

        # 2) клик «приходит» в игру чуть позже наблюдения
        if press:
            self._judge_press(self.sim_time + self._press_off)

        # 2.5) ← stage2c: ТЕЛЕПОРТ КУРСОРА — внешнее возмущение («толчок мыши»).
        # Смещение появится в СЛЕДУЮЩЕМ кадре — политика обязана его увидеть
        # и вернуть курсор к шарику/цели. Именно этому её раньше не учили.
        if self.teleport_prob > 0 and self.np_random.random() < self.teleport_prob:
            ang = self.np_random.uniform(0.0, 2.0 * math.pi)
            dist = self.np_random.uniform(30.0, 90.0)
            self.cursor_x = min(vision.OSU_W - 20.0, max(20.0,
                self.cursor_x + math.cos(ang) * dist))
            self.cursor_y = min(vision.OSU_H - 20.0, max(20.0,
                self.cursor_y + math.sin(ang) * dist))

        # 3) время течёт
        jit = self.np_random.uniform(-DT_JITTER, DT_JITTER) if self.domain_rand else 0.0
        self.sim_time += DT_MS * (1.0 + jit)

        # 4) активируем начавшиеся объекты, ведём слайдеры/спины
        self._activate_new()
        self._update_sliders()
        self._update_spinners()
        self._expire_objects()

        # 5) shaping ровно как в env.py
        goal = self._goal()
        if goal is not None:
            nd = math.hypot(goal[0] - self.cursor_x, goal[1] - self.cursor_y) / 512.0
            # гвард: скачок дистанции > 0.25 = смена цели, не штрафуем
            if self._prev_goal is not None and abs(self._prev_goal - nd) < 0.25:
                self._reward += DENSE_W * (self._prev_goal - nd)
            self._prev_goal = nd
        # бонус за нахождение над объектом (лечит "кружит рядом, не доезжает")
        m = self.cur
        lo = bisect_left(self.object_times, self.sim_time - m["w50"])
        hi = bisect_right(self.object_times, self.sim_time + m["w50"])
        for i in range(lo, hi):
            o, st = self.hitobjects[i], self.obj_state[i]
            if o["type"] == 0 and not st["judged"] and \
               math.hypot(o["x"] - self.cursor_x, o["y"] - self.cursor_y) <= m["r_osu"]:
                self._reward += R_OVERLAP
                break   # один бонус за шаг, независимо от числа кружков под курсором
        self._steps += 1
        terminated = self._map_done()
        truncated = self._steps >= self.max_episode_steps
        info = self._stats() if (terminated or truncated) else {}
        return self._render(), self._reward, terminated, truncated, info

    # ---------------- действия ----------------
    def _dequeue(self, action):
        self._action_q.append(action)
        idx = len(self._action_q) - 1 - self._lag
        return self._action_q[idx] if idx >= 0 else None

    def _apply_buttons(self, btn):
        press = False
        if btn == 1 and not self.btn_l:
            self.btn_l = True; press = True
        elif btn == 2 and self.btn_l:
            self.btn_l = False
        elif btn == 3 and not self.btn_r:
            self.btn_r = True; press = True
        elif btn == 4 and self.btn_r:
            self.btn_r = False
        return press

    def _move(self, d, speed):
        ux = d[0] * speed * self._scale
        uy = d[1] * speed * self._scale
        if self._noise:
            nx, ny = self.np_random.normal(0.0, self._noise, 2)
            ux += nx; uy += ny
        self.cursor_x = min(vision.OSU_W, max(0.0, self.cursor_x + ux))
        self.cursor_y = min(vision.OSU_H, max(0.0, self.cursor_y + uy))

    # ---------------- судейство ----------------
    def _judge_press(self, pt):
        m = self.cur
        lo = bisect_left(self.object_times, pt - m["w50"])
        hi = bisect_right(self.object_times, pt + m["w50"])
        for i in range(lo, hi):
            o, st = self.hitobjects[i], self.obj_state[i]
            if o["type"] == 0:
                if st["judged"]:
                    continue
                if math.hypot(o["x"] - self.cursor_x, o["y"] - self.cursor_y) <= m["r_osu"]:
                    self._judge_hit(i, pt, head=False)
                    return
            elif o["type"] == 1:
                if st["head"] is not None:
                    continue
                if math.hypot(o["x"] - self.cursor_x, o["y"] - self.cursor_y) <= m["r_osu"]:
                    self._judge_hit(i, pt, head=True)
                    return

    def _judge_hit(self, i, pt, head):
        m, o, st = self.cur, self.hitobjects[i], self.obj_state[i]
        dtm = abs(pt - o["time"])
        if dtm <= m["w300"]:
            q = "300"; self._reward += R_300
        elif dtm <= m["w100"]:
            q = "100"; self._reward += R_100
        else:
            q = "50"; self._reward += R_50
        if head:
            st["head"] = q
            # голова слайдера — только в ЛЕГАСИ-счётчики (для сравнения со старыми прогонами)
            if q == "300":   self.n300h += 1
            elif q == "100": self.n100h += 1
            else:            self.n50h += 1
        else:
            st["judged"] = True
            # кружок — в основные счётчики (один объект = один результат)
            if q == "300":   self.n300 += 1
            elif q == "100": self.n100 += 1
            else:            self.n50 += 1

    def _activate_new(self):
        t = self.sim_time
        while self._slide_ptr < len(self.hitobjects) and self.object_times[self._slide_ptr] <= t:
            i = self._slide_ptr
            if self.hitobjects[i]["type"] == 1:
                self._open_sliders.append(i)
            elif self.hitobjects[i]["type"] == 2:
                self._open_spinners.append(i)
            self._slide_ptr += 1

    def _slider_pos(self, o, t):
        repeats = max(1, int(o.get("repeats", 1)))
        total = max(1.0, float(o["duration"]))
        rd = total / repeats
        el = min(total, max(0.0, t - o["time"]))
        ci = min(repeats - 1, int(el // rd))
        lp = max(0.0, min(1.0, (el - ci * rd) / rd))
        prog = lp if ci % 2 == 0 else 1.0 - lp
        return vision.get_point_on_curve(o["curve_points"], prog)

    def _update_sliders(self):
        m, t = self.cur, self.sim_time
        held = self.btn_l or self.btn_r
        for i in list(self._open_sliders):
            o, st = self.hitobjects[i], self.obj_state[i]
            if t > o["time"] + o["duration"]:
                self._finalize_slider(i)
                self._open_sliders.remove(i)
                continue
            st["follow_tot"] += 1
            bx, by = self._slider_pos(o, t)
            if held and math.hypot(bx - self.cursor_x, by - self.cursor_y) <= m["r_osu"] * FOLLOW_RADIUS_MULT:
                st["follow_ok"] += 1
                self._reward += R_FOLLOW_STEP
            else:
                self._reward -= R_FOLLOW_STEP

    def _finalize_slider(self, i):
        st = self.obj_state[i]
        st["judged"] = st["final"] = True
        head_ok = st["head"] in ("300", "100", "50")
        frac = st["follow_ok"] / max(1, st["follow_tot"])
        if head_ok and frac >= 0.85:
            self._reward += R_SLIDER_END;
            self.n300 += 1
        elif head_ok and frac >= 0.3:
            self._reward += 0.1;
            self.n100 += 1
        elif head_ok:
            # sb: комбо-брейк, в acc — 50 (как в stable, не завышаем)
            self._reward += R_SB;
            self.nsb += 1;
            self.n50 += 1
        # голова промазана: miss уже посчитан в _expire_objects — 1 объект = 1 запись

    def _update_spinners(self):
        t = self.sim_time
        held = self.btn_l or self.btn_r
        for i in list(self._open_spinners):
            o, st = self.hitobjects[i], self.obj_state[i]
            if t > o["time"] + o["duration"]:
                self._finalize_spinner(i)
                self._open_spinners.remove(i)
                continue
            ang = math.atan2(self.cursor_y - 192.0, self.cursor_x - 256.0)
            if st["last_ang"] is not None and held:
                d = ang - st["last_ang"]
                while d > math.pi: d -= 2 * math.pi
                while d < -math.pi: d += 2 * math.pi
                st["spun"] += abs(d)
                self._reward += R_SPIN_STEP * min(1.0, abs(d) / 0.4)
            st["last_ang"] = ang

    def _finalize_spinner(self, i):
        o, st = self.hitobjects[i], self.obj_state[i]
        st["judged"] = st["final"] = True
        need = (o["duration"] / 1000.0) * (1.5 + 0.4 * self.cur_od()) * 2 * math.pi
        if st["spun"] >= need:
            self._reward += R_SPIN_END; self.n300 += 1
        else:
            self._reward += R_MISS; self.nmiss += 1
            self.nmiss_sp += 1
    def cur_od(self):
        return (200.0 - self.cur["w50"]) / 10.0 if self.cur["w50"] >= 150.0 else (190.0 - self.cur["w50"]) / 8.0

    def _expire_objects(self):
        m, t = self.cur, self.sim_time
        hi = bisect_right(self.object_times, t - m["w50"])
        for i in range(self._expire_ptr, hi):
            o, st = self.hitobjects[i], self.obj_state[i]
            if o["type"] == 0 and not st["judged"]:
                st["judged"] = True; self.nmiss += 1; self._reward += R_MISS
                self.nmiss_c += 1
            elif o["type"] == 1 and st["head"] is None:
                st["head"] = "miss"; self.nmiss += 1; self._reward += R_MISS
        self._expire_ptr = hi

    def _goal(self):
        t = self.sim_time
        # ФИКС: активный слайдер/спиннер ещё играет, хотя его старт уже
        # позади _expire_ptr — ведём до конца, а не бросаем на полпути
        if self._open_sliders:
            o = self.hitobjects[self._open_sliders[-1]]
            return self._slider_pos(o, t)
        if self._open_spinners:
            return 256.0, 192.0
        start = max(0, self._expire_ptr)
        for i in range(start, min(len(self.hitobjects), start + 8)):
            o, st = self.hitobjects[i], self.obj_state[i]
            if o["type"] == 0:
                if st["judged"]:
                    continue
                return float(o["x"]), float(o["y"])
            if o["type"] == 1:
                if st["final"]:
                    continue
                if t < o["time"]:
                    return float(o["x"]), float(o["y"])
                return self._slider_pos(o, t)
            if o["type"] == 2:
                if st["final"]:
                    continue
                return 256.0, 192.0
        return None

    def _map_done(self):
        return (self._expire_ptr >= len(self.hitobjects)
                and not self._open_sliders and not self._open_spinners)

    # ---------------- рендер: 1-в-1 как env._get_obs ----------------
    def _render(self):
        m = self.cur
        canvas = np.zeros((vision.IMG_H, vision.IMG_W), np.uint8)
        t = self.sim_time
        lf, rt = t - vision.PAST_MS, t + m["ar_ms"]
        l = max(0, bisect_left(self.object_times, lf) - 2)
        r = min(len(self.hitobjects), bisect_right(self.object_times, rt) + 2)
        for o in self.hitobjects[l:r]:
            if vision.object_visible(o, t, m["ar_ms"]):
                try:
                    vision.render_hitobject(canvas, o, t, m["ar_ms"], m["cs_radius_px"])
                except Exception:
                    pass
        vision.draw_cursor(canvas, self.cursor_x, self.cursor_y)
        if OBS_SCALE != 1.0:
            canvas = cv2.resize(canvas, (self.obs_shape[1], self.obs_shape[0]),
                                interpolation=cv2.INTER_AREA)
        return canvas[:, :, None].copy()

    def _stats(self):
        # --- новая: один объект = один судейский результат ---
        total = self.n300 + self.n100 + self.n50 + self.nmiss
        acc = (300*self.n300 + 100*self.n100 + 50*self.n50) / (300*total) if total else 0.0
        # --- легаси: двойной счёт голов; sb не добавлял записей ---
        l300 = self.n300 + self.n300h
        l100 = self.n100 + self.n100h
        l50 = self.n50 - self.nsb + self.n50h      # nsb: каждый = ровно один лишний 50
        ltotal = l300 + l100 + l50 + self.nmiss
        acc_legacy = (300*l300 + 100*l100 + 50*l50) / (300*ltotal) if ltotal else 0.0
        return {
            "acc": round(acc, 3), "acc_legacy": round(acc_legacy, 3),
            "300": self.n300, "100": self.n100, "50": self.n50,
            "miss": self.nmiss, "sb": self.nsb,
            "miss_c": self.nmiss_c,
            "miss_s": self.nmiss - self.nmiss_c - self.nmiss_sp,
            "miss_sp": self.nmiss_sp,
        }


if __name__ == "__main__":
    import glob, sys, time
    maps = sorted(glob.glob("sim_maps/*.osu"))
    if not maps:
        print("Положи .osu файлы в папку sim_maps/"); sys.exit()
    env = SimOsuEnv(maps)
    obs, _ = env.reset()
    print("obs:", obs.shape, "| action space:", env.action_space)
    t0, steps = time.time(), 0
    for ep in range(5):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            steps += 1
            done = term or trunc
        print(f"эпизод {ep}: {info}")
    dt = time.time() - t0
    print(f"скорость одного env: {steps/dt:.0f} шагов/сек (рандомная политика)")