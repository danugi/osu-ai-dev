# pf_live.py — калибровка ПО ЖИВОЙ ИГРЕ (без паузы, без вопросов)
# 1) удали pf_calib.json, если лежит (мусор от прошлых попыток)
# 2) в osu: лёгкая карта (1.5-2★), мод NoFail
# 3) python pf_live.py -> переключись в osu -> Play -> НЕ ТРОГАЙ мышь 12 сек
# 4) результат в консоли + calib_debug.png (зелёный = прогноз, красный = найдено)
import os, ctypes, json, math, time
from collections import defaultdict
import numpy as np
import cv2
from PIL import ImageGrab

if os.path.exists("pf_calib.json"):
    os.remove("pf_calib.json")     # старый мусор от невалидных попыток
    print("старый pf_calib.json удалён")

import env as env_mod
env_mod.SHOW_WINDOW = False         # окно Neuro Vision не должно попасть в скрин
from env import PF, OsuEnv
import vision

user32 = ctypes.windll.user32
print("Жду, пока карта реально пойдёт (запусти её в osu)...")
env = OsuEnv()
last_t = None
while True:
    st = env._fetch_state()
    if st is None:
        time.sleep(0.2); continue
    env._sync_map(st)
    t = vision.get_current_time(st)
    if last_t is not None and t > last_t + 5 and t > 0:
        break
    last_t = t
    time.sleep(0.1)
print(f"игра идёт: {len(env.hitobjects)} об., AR={env.ar_ms:.0f}мс. Собираю 12 сек, мышь не трогай.")

user32.SetCursorPos(30, 30)   # курсор в угол, подальше от поля


def blob_center(img, px, py, win=200):
    h, w = img.shape
    x0, y0 = max(0, px-win), max(0, py-win)
    x1, y1 = min(w, px+win), min(h, py+win)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0 or crop.max() < 80:
        return None
    mask = (crop > crop.max()*0.55).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cx0, cy0 = px-x0, py-y0
    best, best_d = None, 1e9
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 2000:
            continue
        bx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]/2.0
        by = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]/2.0
        d = math.hypot(bx-cx0, by-cy0)
        if d < best_d:
            best, best_d = (x0+bx, y0+by), d
    return best


det = defaultdict(list)
debug_img = None
t0 = time.perf_counter()
frames = 0
while time.perf_counter() - t0 < 12.0:
    st = env._fetch_state()
    if st is None:
        time.sleep(0.05); continue
    t = vision.get_current_time(st)
    img = np.array(ImageGrab.grab().convert("L"))
    frames += 1
    debug_img = img.copy()
    # твёрдо видимые будущие кружки (запас от границ фейда и от латентности)
    for i, o in enumerate(env.hitobjects):
        if o["type"] != 0:
            continue
        if not (t + 0.25*env.ar_ms <= o["time"] <= t + 0.85*env.ar_ms):
            continue
        px, py = PF.osu_to_screen(o["x"], o["y"])
        c = blob_center(img, int(px), int(py))
        if c is not None:
            det[i].append(c)
    time.sleep(0.08)

print(f"кадров: {frames}, кружков задето: {len(det)}")

pairs = []
for i, pts in det.items():
    if len(pts) < 3:
        continue
    a = np.array(pts)
    if a.std(axis=0).max() > 8:      # позиция прыгала — мусор
        continue
    o = env.hitobjects[i]
    pairs.append((o["x"], o["y"], float(a[:, 0].mean()), float(a[:, 1].mean()), len(pts)))

for p in pairs:
    print(f"  osu({p[0]:.0f},{p[1]:.0f}) -> экр({p[2]:.0f},{p[3]:.0f}) [{p[4]} кадров]")

if len(pairs) < 3:
    raise SystemExit("стабильных кружков < 3 — приложи calib_debug.png и вывод, разберём")

xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
if max(xs)-min(xs) < 60 or max(ys)-min(ys) < 60:
    raise SystemExit("кружки почти на одной линии — возьми карту с разбросом")

fx = np.polyfit([p[0] for p in pairs], [p[2] for p in pairs], 1)
fy = np.polyfit([p[1] for p in pairs], [p[3] for p in pairs], 1)
scale_x, left = fx[0], fx[1]
scale_y, top = fy[0], fy[1]
scale = (scale_x + scale_y) / 2.0

res = [math.hypot(p[2]-(left+p[0]*scale_x), p[3]-(top+p[1]*scale_y)) for p in pairs]
H = user32.GetSystemMetrics(1)

print(f"\nИЗМЕРЕНО: scale={scale:.3f}  left={left:.1f}  top={top:.1f}")
print(f"БЫЛО:     scale={PF.scale:.3f}  left={PF.left:.1f}  top={PF.top:.1f}")
print(f"разница:  dleft={left-PF.left:+.0f}px  dtop={top-PF.top:+.0f}px  "
      f"dscale={scale-PF.scale:+.3f}  (H/480={H/480:.3f})")
print(f"остаточная: mean={np.mean(res):.1f}px  max={np.max(res):.1f}px  (кружков {len(pairs)})")

if debug_img is not None:
    d = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
    for p in pairs:
        px, py = PF.osu_to_screen(p[0], p[1])
        cv2.circle(d, (int(px), int(py)), 60, (0, 255, 0), 3)
        cv2.circle(d, (int(p[2]), int(p[3])), 60, (0, 0, 255), 3)
    cv2.imwrite("calib_debug.png", d)
    print("calib_debug.png: зелёный = наш прогноз, красный = найдено на экране")

if np.mean(res) < 8 and abs(scale_x-scale_y) < 0.05*scale and 1.5 < scale < 3.5:
    json.dump({"left": float(left), "top": float(top), "scale": float(scale)},
              open("pf_calib.json", "w"))
    print("\nвалидация ok -> pf_calib.json сохранён")
    print("дальше: python vision_replay.py -> потом play_real")
else:
    print("\n!! фит не прошёл. Приложи сюда calib_debug.png + вывод целиком.")