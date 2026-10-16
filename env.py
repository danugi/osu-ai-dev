import os
import copy
import math
import time
import ctypes
import ctypes.wintypes
import cv2
import numpy as np
import requests
import gymnasium as gym
from gymnasium import spaces
from bisect import bisect_left, bisect_right
import json
import input_ctrl as ic
import vision

TOSU_URL = "http://127.0.0.1:24050/json/v2"

ACTIONS_PER_SECOND = 60          # 60HZ: было 30
MOVE_DISTANCES = [3, 12, 48]     # 60HZ: было [6, 24, 96]; скорости сохранены
RETRY_VK = 0xC0            # ` — Quick Retry
STALL_TIMEOUT_SEC = 6.0
SHOW_WINDOW = True         # окно показывает ровно то, что видит сеть
OBS_SCALE = 0.5

R_300, R_100, R_50, R_MISS, R_SB = 1.0, 0.4, 0.1, -1.0, -0.5
R_STEP = -0.002
DENSE_W = 0.25

DIRS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()   # честные физические px при масштабе Windows ≠ 100%
WINDOW_NAME = "Neuro Vision"


class Playfield:
    """Экран <-> поле osu!. Константы из старого проекта (проверены реальной игрой):
    поле 4:3 высотой 80% экрана, по горизонтали центр, по вертикали +8 osu-px вниз."""
    def __init__(self):
        W = user32.GetSystemMetrics(0)
        H = user32.GetSystemMetrics(1)
        self.pf_h = H * 0.80
        self.pf_w = self.pf_h * 4.0 / 3.0
        self.left = (W - self.pf_w) / 2.0
        self.top = (H - self.pf_h) / 2.0 + self.pf_h * (8.0 / 384.0)
        self.scale = self.pf_h / 384.0          # экранных px в одном osu-px
        # калибровка поверх дефолта (пишет pf_calib.py):
        try:
            with open("pf_calib.json") as f:
                c = json.load(f)
            self.left, self.top, self.scale = c["left"], c["top"], c["scale"]
        except Exception:
            pass
    def screen_to_osu(self, sx, sy):
        return (sx - self.left) / self.scale, (sy - self.top) / self.scale

    def osu_to_screen(self, x, y):
        return self.left + x * self.scale, self.top + y * self.scale


PF = Playfield()
MOUSE_SCALE = PF.scale            # 1080p → 2.25 (было 2.8 — перелёт 24%)


class OsuEnv(gym.Env):
    def __init__(self):
        super().__init__()
        ctypes.windll.winmm.timeBeginPeriod(1)   # таймер Windows 1мс вместо 15.6
        h, w = int(vision.IMG_H * OBS_SCALE), int(vision.IMG_W * OBS_SCALE)
        self.obs_shape = (h, w, 1)
        self.action_space = spaces.MultiDiscrete([len(DIRS) * len(MOVE_DISTANCES), 5])
        self.observation_space = spaces.Box(0, 255, self.obs_shape, np.uint8)

        # Окно-зрение (то же, что в старом визуализаторе)
        if SHOW_WINDOW:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 800, 600)
            hwnd = user32.FindWindowW(None, WINDOW_NAME)
            if hwnd:
                user32.SetWindowPos(hwnd, 0xFFFF, 0, 0, 0, 0,
                                    0x0001 | 0x0002 | 0x0010 | 0x00040000)

        # Сырой разбор (без модов) + текущая сборка под моды
        self._raw_md5 = None
        self._raw_objects, self._raw_ar_ms, self._raw_cs = [], 1200.0, 4.0
        self._map_key = None
        self.hitobjects, self.object_times = [], []
        self.ar_ms, self.cs_radius_px, self.mods = 1200.0, 18.0, 0

        self.cursor_x, self.cursor_y = 256.0, 192.0
        self.btn_l = self.btn_r = False
        self._prev_hits, self._prev_goal_dist = {}, None
        self._stall, self._last_t, self._time = 0.0, None, 0.0
        self._state, self._fails = None, 0
        self._window_quit = False
        self._session = requests.Session()
        self._tick = time.perf_counter()

    def _fetch_state(self):
        try:
            r = self._session.get(TOSU_URL, timeout=1)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def _screen_to_osu(self):
        """Реальный системный курсор -> координаты поля osu!."""
        pt = ctypes.wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        ox = (pt.x - PF.left) / PF.scale
        oy = (pt.y - PF.top) / PF.scale
        return (min(vision.OSU_W, max(0.0, ox)),
                min(vision.OSU_H, max(0.0, oy)))

    def _sync_map(self, st):
        md5 = st.get("beatmap", {}).get("checksum", "")
        mods = vision.get_mod_number(st)
        if not md5:
            return

        # Сырой парс — один раз на карту
        if md5 != self._raw_md5:
            base = st.get("folders", {}).get("songs", "")
            rel = st.get("files", {}).get("beatmap", "")
            path = os.path.join(base, rel) if base and rel else None
            if not path or not os.path.exists(path):
                return
            self._raw_objects, self._raw_ar_ms, self._raw_cs, _od = vision.parse_osu_file(path)
            self._raw_md5 = md5
            self._map_key = None  # форсируем пересборку

        # Пересборка при смене карты ИЛИ МОДОВ (md5 от модов не меняется!)
        if self._map_key != (md5, mods):
            objs = copy.deepcopy(self._raw_objects)
            ar_ms, cs = vision.apply_mods(objs, self._raw_ar_ms, self._raw_cs, mods)

            self.hitobjects = objs
            self.object_times = [o["time"] for o in objs]
            self.ar_ms = ar_ms
            self.cs_radius_px = max(2.0, (54.4 - 4.48 * cs) * vision.PLAY_W / vision.OSU_W)
            self.mods = mods
            self._map_key = (md5, mods)
            vision.ACTIVE_MODS = mods  # HD включается в рендере

            names = [n for b, n in ((vision.MOD_HD, "HD"), (vision.MOD_HR, "HR"),
                     (vision.MOD_DT, "DT"), (vision.MOD_NC, "NC"),
                     (vision.MOD_HT, "HT"), (vision.MOD_EZ, "EZ")) if mods & b]
            print(f"[env] Карта: {len(objs)} об. | AR={ar_ms:.0f}ms | CS={cs:.2f} | "
                  f"моды: {'+'.join(names) if names else 'nomod'}")

            if (mods & (vision.MOD_DT | vision.MOD_NC)) and ACTIONS_PER_SECOND < 45:
                print("[env] DT/NC при 30 шагах/сек тесно — подними ACTIONS_PER_SECOND до 45–60.")

    def _get_obs(self):
        st = self._fetch_state()
        canvas = np.zeros((vision.IMG_H, vision.IMG_W), np.uint8)
        if st is not None:
            self._sync_map(st)
            self._time = vision.get_current_time(st)
            self._state = st
            self._fails = 0
            # ЗАМКНУТАЯ ПЕТЛЯ: belief = реальный курсор каждый тик.
            # Политика видит истинный результат своих команд и корректирует
            # дрейф — ровно то, к чему её готовил scale-jitter в симе.
            self.cursor_x, self.cursor_y = self._screen_to_osu()
            if self.hitobjects:
                lf, rt = self._time - vision.PAST_MS, self._time + self.ar_ms
                l = max(0, bisect_left(self.object_times, lf) - 2)
                r = min(len(self.hitobjects), bisect_right(self.object_times, rt) + 2)
                for o in self.hitobjects[l:r]:
                    if vision.object_visible(o, self._time, self.ar_ms):
                        try:
                            vision.render_hitobject(canvas, o, self._time,
                                                    self.ar_ms, self.cs_radius_px)
                        except Exception:
                            pass
        else:
            self._fails += 1
            # tosu молчит -> belief живёт интегралом команд (dead reckoning)

        # проприоцепция: где курсор (теперь — где он РЕАЛЬНО)
        vision.draw_cursor(canvas, self.cursor_x, self.cursor_y)

        # человек смотрит ровно на вход сети
        if SHOW_WINDOW:
            cv2.imshow(WINDOW_NAME, canvas)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                self._window_quit = True

        if OBS_SCALE != 1.0:
            canvas = cv2.resize(canvas, (self.obs_shape[1], self.obs_shape[0]),
                                interpolation=cv2.INTER_AREA)
        return canvas[:, :, None].copy()

    def _slider_pos(self, o, t):
        repeats = max(1, int(o.get("repeats", 1)))
        total = max(1.0, float(o["duration"]))
        rd = total / repeats
        el = min(total, max(0.0, t - o["time"]))
        ci = min(repeats - 1, int(el // rd))
        lp = max(0.0, min(1.0, (el - ci * rd) / rd))
        prog = lp if ci % 2 == 0 else 1.0 - lp
        return vision.get_point_on_curve(o["curve_points"], prog)

    def _next_goal(self):
        t = self._time
        if not self.hitobjects:
            return None
        i = bisect_right(self.object_times, t - 150)
        # ФИКС: активный слайдер/спиннер мог начаться раньше точки среза —
        # не бросаем его на полпути, ведём до конца
        while i > 0:
            p = self.hitobjects[i - 1]
            if p["type"] != 0 and p["time"] + p["duration"] >= t:
                i -= 1
            else:
                break
        for o in self.hitobjects[i:i + 6]:
            if o["time"] + o["duration"] >= t - 150:
                if o["type"] == 2:
                    return 256.0, 192.0
                if o["type"] == 1 and t >= o["time"]:
                    return self._slider_pos(o, t)
                return float(o["x"]), float(o["y"])
        return None

    def _reward_done(self):
        reward, terminated, truncated = R_STEP, False, False
        st = self._state
        if st is not None:
            gp = st.get("play") or st.get("gameplay") or {}
            hits = gp.get("hits", {}) if isinstance(gp, dict) else {}

            def h(*names):
                for n in names:
                    if isinstance(hits, dict) and n in hits:
                        try:
                            return int(hits[n])
                        except (TypeError, ValueError):
                            pass
                return None

            cur = {"300": h("300", "c300"), "100": h("100", "c100"),
                   "50": h("50", "c50"), "miss": h("miss", "cMiss", "misses"),
                   "sb": h("sliderBreaks", "slider_breaks")}
            weights = {"300": R_300, "100": R_100, "50": R_50,
                       "miss": R_MISS, "sb": R_SB}
            for k, v in cur.items():
                if v is not None and self._prev_hits.get(k) is not None:
                    d = v - self._prev_hits[k]
                    if d:
                        reward += weights[k] * d
            self._prev_hits = cur

            goal = self._next_goal()
            if goal is not None:
                nd = math.hypot(goal[0] - self.cursor_x, goal[1] - self.cursor_y) / 512.0
                # гвард: скачок дистанции > 0.25 = смена цели, не штрафуем
                if self._prev_goal_dist is not None and abs(self._prev_goal_dist - nd) < 0.25:
                    reward += DENSE_W * (self._prev_goal_dist - nd)
                self._prev_goal_dist = nd

            if self._last_t is not None and self._time <= self._last_t + 1e-6:
                self._stall += 1.0 / ACTIONS_PER_SECOND
            else:
                self._stall = 0.0
            self._last_t = self._time
            if self._stall > STALL_TIMEOUT_SEC:
                truncated = True

            hp = gp.get("hp", {}) if isinstance(gp, dict) else {}
            hpv = hp.get("normal", 1.0) if isinstance(hp, dict) else hp
            if isinstance(hpv, (int, float)) and hpv <= 0.0:
                terminated = True

        if self._fails > 90:
            truncated = True
        return reward, terminated, truncated

    def _move(self, ux, uy):
        ic.mouse_move(ux * MOUSE_SCALE, uy * MOUSE_SCALE)
        # belief обновит _get_obs из реального положения (петля);
        # эти строки — dead reckoning на случай, если tosu молчит
        self.cursor_x = min(vision.OSU_W, max(0.0, self.cursor_x + ux))
        self.cursor_y = min(vision.OSU_H, max(0.0, self.cursor_y + uy))

    def _apply_button(self, btn):
        if btn == 1 and not self.btn_l:
            ic.mouse_button(True, True); self.btn_l = True
        elif btn == 2 and self.btn_l:
            ic.mouse_button(False, True); self.btn_l = False
        elif btn == 3 and not self.btn_r:
            ic.mouse_button(True, False); self.btn_r = True
        elif btn == 4 and self.btn_r:
            ic.mouse_button(False, False); self.btn_r = False

    def _retry(self):
        ic.key_press(RETRY_VK)
        deadline = time.time() + 15
        while time.time() < deadline:
            st = self._fetch_state()
            if st is None:
                time.sleep(0.25)
                continue
            t = vision.get_current_time(st)
            # ждём не "карта сброшена", а "музыка пошла" — первый объект ещё впереди
            if t < -50:      # всё ещё lead-in (отрицательное время)
                time.sleep(0.1)
                continue
            if t >= -50 and t < 800:   # lead-in кончился, старт!
                time.sleep(0.3)        # дать кадру устаканиться
                return
            time.sleep(0.25)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.btn_l: ic.mouse_button(False, True); self.btn_l = False
        if self.btn_r: ic.mouse_button(False, False); self.btn_r = False
        self._retry()
        # АНКЕР: belief стартует там, где реально стоит мышь (не в центре)
        self.cursor_x, self.cursor_y = self._screen_to_osu()
        self._prev_goal_dist, self._stall, self._last_t = None, 0.0, None
        self._prev_hits, self._window_quit = {}, False
        st = self._fetch_state()
        if st is not None:
            self._sync_map(st)
        self._tick = time.perf_counter()
        return self._get_obs(), {}

    def step(self, action):
        a = np.asarray(action).flatten()
        move_idx, btn = int(a[0]), int(a[1])
        d = DIRS[move_idx % len(DIRS)]
        speed = MOVE_DISTANCES[min(len(MOVE_DISTANCES) - 1, move_idx // len(DIRS))]
        self._move(d[0] * speed, d[1] * speed)
        self._apply_button(btn)

        self._tick += 1.0 / ACTIONS_PER_SECOND
        now = time.perf_counter()
        if self._tick > now:
            time.sleep(self._tick - now)
        else:
            self._tick = now

        obs = self._get_obs()
        reward, terminated, truncated = self._reward_done()
        if self._window_quit:
            truncated = True
        return obs, reward, terminated, truncated, {}


if __name__ == "__main__":
    print("ВНИМАНИЕ: смоук-тест двигает РЕАЛЬНУЮ мышь и жмёт Quick Retry.")
    print("Запускай только с открытым osu + tosu. Выход — q в окне Neuro Vision.")
    env = OsuEnv()
    obs, _ = env.reset()
    print("obs:", obs.shape, "| action space:", env.action_space)
    total = 0.0
    for i in range(1200):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        total += r
        if term or trunc:
            print(f"Эпизод кончился на шаге {i}, reward={total:.2f}")
            obs, _ = env.reset()
            total = 0.0
    print("Среда работает.")