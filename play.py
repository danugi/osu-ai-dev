import argparse
import time
import threading
import ctypes
import numpy as np
from stable_baselines3 import PPO
from env import OsuEnv
import input_ctrl as ic
import vision

try:                      # монитор необязателен: без него play.py работает как раньше
    import ai_telemetry
except Exception:         # noqa: BLE001
    ai_telemetry = None

STOP = False
PAUSED = False            # ручная пауза из UI (не путать с паузой самой osu!)


def _stop_listener():
    global STOP
    GetAsyncKeyState = ctypes.windll.user32.GetAsyncKeyState
    while not STOP:
        if GetAsyncKeyState(0x77) & 0x8000:   # F8
            STOP = True
            print("\n[!] F8 — стоп. Отпускаю кнопки...")
            return
        time.sleep(0.05)


class Stacker:
    def __init__(self, k):
        self.k, self.buf = k, None
    def reset(self, obs):
        self.buf = np.repeat(obs, self.k, axis=2)
        return self.buf
    def push(self, obs):
        self.buf = np.concatenate([self.buf[:, :, 1:], obs], axis=2)
        return self.buf


def release_buttons(env):
    if env.btn_l: ic.mouse_button(False, True); env.btn_l = False
    if env.btn_r: ic.mouse_button(False, False); env.btn_r = False


def soft_start(env):
    """Подхват игры, которую запустил человек: БЕЗ Quick Retry.
    Якорим belief на реальный курсор, обнуляем статистику эпизода."""
    env.cursor_x, env.cursor_y = env._screen_to_osu()
    env._prev_goal_dist, env._stall, env._last_t = None, 0.0, None
    env._prev_hits, env._window_quit = {}, False
    env._tick = time.perf_counter()
    return env._get_obs()


# ---------------- интеграция с монитором (вся опциональна) ----------------
def _publish(**fields):
    if ai_telemetry is not None:
        ai_telemetry.publish("play", **fields)


def _handle_commands(model, args):
    """Выполнить новую команду из UI. Возвращает (model, note|None).

    PAUSE/RESUME лишь ставят флаг (главный цикл сам отпускает кнопки);
    SAVE/RELOAD идут через существующий SB3-интерфейс модели.
    """
    if ai_telemetry is None:
        return model, None
    cmd = ai_telemetry.read_command(getattr(_handle_commands, "last_id", 0))
    if not cmd:
        return model, None
    _handle_commands.last_id = int(cmd.get("id", 0))
    c = cmd.get("cmd")
    global PAUSED
    if c == "pause":
        PAUSED = True
    elif c == "resume":
        PAUSED = False
    elif c == "save":
        try:
            model.save("play_manual_save")
            print("[monitor] модель сохранена: play_manual_save.zip")
        except Exception as e:            # noqa: BLE001
            print(f"[monitor] save failed: {e}")
    elif c == "reload":
        try:
            model = PPO.load(args.model, device="cpu")
            print(f"[monitor] модель перечитана: {args.model}")
        except Exception as e:            # noqa: BLE001
            print(f"[monitor] reload failed: {e}")
    else:
        return model, None
    return model, c


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="best_stage2b/stage2b_800000_steps.zip")
    p.add_argument("--stochastic", action="store_true")
    args = p.parse_args()

    threading.Thread(target=_stop_listener, daemon=True).start()

    env = OsuEnv()
    model = PPO.load(args.model, device="cpu")
    stack = Stacker(4)

    in_game, ep, total = False, 0, 0.0
    last_t, frozen, last_num = None, 0, None

    # --- телеметрия для монитора: счётчики FPS/инференса, троттлинг записей ---
    _fps_cnt, _fps_t = 0, time.perf_counter()
    _fps = None
    _infer_ms = None
    _last_pub = 0.0
    _step_reward = 0.0
    _publish(model_loaded=True, model=args.model, ai_status="RUNNING",
             step_reward=0.0, episode_reward=0.0)

    print("Жду, пока ты запустишь карту. В меню/на паузе — политика спит, кнопки отпущены.")

    def _pub_throttled(force=False):
        global _last_pub
        now = time.time()
        if force or now - _last_pub >= 0.15:
            _last_pub = now
            _publish(
                ai_status="PAUSED" if PAUSED else "RUNNING",
                model=args.model, model_loaded=True,
                fps=_fps, inference_ms=_infer_ms,
                step_reward=_step_reward, episode_reward=round(total, 2),
                action=[int(a) for a in _last_action] if _last_action is not None else None,
                cursor={"x": float(env.cursor_x), "y": float(env.cursor_y)},
                ai_connection=env._state is not None,
                telemetry_errors=int(getattr(env, "_fails", 0)),
            )

    _last_action = None

    try:
        while not STOP:
            # ---- команды из UI (pause/resume/save/reload) ----
            model, note = _handle_commands(model, args)

            # ---- вратарь: идёт ли игра? ----
            st = env._fetch_state()
            num = st.get("state", {}).get("number", -1) if st else -1
            name = st.get("state", {}).get("name", "?") if st else "?"
            t = vision.get_current_time(st) if st else None

            if last_num != num and num != 2:
                print(f"[ждём] state={num} ({name})")
            last_num = num

            # пауза = время в игре стоит
            paused = False
            if num == 2 and t is not None:
                if last_t is not None and t <= last_t + 0.5:
                    frozen += 1
                else:
                    frozen = 0
                last_t = t
                paused = frozen >= 3          # ~0.1с время не двигается
            else:
                frozen, last_t = 0, None

            # ручная пауза из UI ведёт себя как "пауза игры": отпускаем кнопки
            if num != 2 or paused or PAUSED:
                if in_game:
                    ep += 1
                    print(f"эпизод {ep}: reward={total:.1f} — пауза/конец, кнопки отпущены")
                    total, in_game = 0.0, False
                release_buttons(env)
                env._get_obs()                # окно Neuro Vision живёт, q работает
                if env._window_quit:
                    break
                _step_reward = 0.0
                _last_action = None
                _pub_throttled()
                time.sleep(0.15)
                continue

            if not in_game:
                # человек сам нажал Play — подхватываем без ретрая
                obs = soft_start(env)
                s = stack.reset(obs)
                in_game = True
                print("игра пошла — подхватил")
                _pub_throttled(force=True)
                continue

            # ---- обычный игровой шаг ----
            _t0 = time.perf_counter()
            action, _ = model.predict(s, deterministic=not args.stochastic)
            _infer_ms = (time.perf_counter() - _t0) * 1000.0
            obs, r, term, trunc, _ = env.step(action)
            s = stack.push(obs)
            total += float(r)
            _step_reward = float(r)
            _last_action = action
            _fps_cnt += 1

            if term or trunc:
                ep += 1
                print(f"эпизод {ep}: reward={total:.1f} (env: term/trunc)")
                total, in_game = 0.0, False
                release_buttons(env)
                _pub_throttled(force=True)
                if env._window_quit:          # q = полный выход
                    break

            # FPS считаем окном ~0.5с
            _now = time.perf_counter()
            if _now - _fps_t >= 0.5:
                _fps = round(_fps_cnt / (_now - _fps_t), 1)
                _fps_cnt, _fps_t = 0, _now
            _pub_throttled()
    finally:
        ic.mouse_button(False, True)
        ic.mouse_button(False, False)
        if ai_telemetry is not None:
            ai_telemetry.clear("play")     # монитор переведёт AI в STOPPED
        print("Выход. Кнопки отпущены.")