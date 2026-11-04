# viewer.py — standalone-просмотрщик синтетического зрения
# (бывший __main__-блок vision.py; вынесен, чтобы ядро зрения было
# кросс-платформенным и без WinAPI на импорт-пути).
# Показывает ровно то, что видит сеть, по данным tosu.
# Запуск: python viewer.py — выход q в окне. Windows-only.
import ctypes
import copy
import os
import time
from bisect import bisect_left, bisect_right

import cv2
import numpy as np
import requests

import vision
from vision import (IMG_H, IMG_W, PAST_MS, PLAY_W, OSU_W,
                    MOD_EZ, MOD_HD, MOD_HR, MOD_DT, MOD_HT, MOD_NC,
                    parse_osu_file, apply_mods, get_current_time,
                    get_mod_number, object_visible, render_hitobject,
                    circle_radius_osu)
from core import TOSU_URL

user32 = ctypes.windll.user32

# ============================================================
# MAIN (standalone viewer) — запускается ТОЛЬКО как `python vision.py`
# При import vision этот блок не выполняется.
# ============================================================
if __name__ == "__main__":
    window_name = "Neuro Vision"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    hwnd = user32.FindWindowW(None, window_name)
    if hwnd:
        user32.SetWindowPos(
            hwnd, 0xFFFF, 0, 0, 0, 0,
            0x0001 | 0x0002 | 0x0010 | 0x00040000,
        )

    # Сырой разбор (один раз на карту) + текущая сборка под конкретные моды.
    raw_md5 = None
    raw_objects, raw_ar_ms, raw_cs, raw_od = [], 1200.0, 4.0, 9.0
    map_key = None
    hitobjects, object_times = [], []
    ar_ms, cs_radius_px = 1200.0, 18.0

    print("Ожидание tosu...")

    while True:
        try:
            response = requests.get(TOSU_URL, timeout=2)
            response.raise_for_status()
            state = response.json()

            beatmap = state.get("beatmap", {})
            md5 = beatmap.get("checksum", "")
            current_time = get_current_time(state)
            mods = get_mod_number(state)

            # Сырой парс — один раз на новую карту.
            if md5 and md5 != raw_md5:
                base_path = state.get("folders", {}).get("songs", "")
                rel_path = state.get("files", {}).get("beatmap", "")
                filepath = os.path.join(base_path, rel_path) if base_path and rel_path else None

                if filepath and os.path.exists(filepath):
                    raw_objects, raw_ar_ms, raw_cs, raw_od = parse_osu_file(filepath)
                    raw_md5 = md5
                    map_key = None  # форсируем пересборку
                else:
                    print("Файл .osu не найден по пути из tosu.      ", end="\r")

            # Пересборка при смене карты ИЛИ МОДОВ (md5 от модов не меняется!).
            if raw_md5 and map_key != (md5, mods):
                objs = copy.deepcopy(raw_objects)
                ar_ms, cs = apply_mods(objs, raw_ar_ms, raw_cs, mods)
                vision.ACTIVE_MODS = mods  # включает HD в рендере

                hitobjects = objs
                object_times = [o["time"] for o in objs]
                cs_radius_px = max(2.0, circle_radius_osu(cs) * PLAY_W / OSU_W)
                map_key = (md5, mods)

                names = [n for b, n in (
                    (MOD_EZ, "EZ"), (MOD_HD, "HD"), (MOD_HR, "HR"),
                    (MOD_DT, "DT"), (MOD_HT, "HT"), (MOD_NC, "NC"),
                ) if mods & b]
                print(
                    f"\nКарта: {len(objs)} об. | AR={ar_ms:.0f} ms | "
                    f"CS={cs:.2f} | OD={raw_od:.2f} | "
                    f"моды: {'+'.join(names) if names else 'nomod'}"
                )

            if hitobjects:
                # Pure grayscale observation. No osu! UI and no cursor.
                canvas = np.zeros((IMG_H, IMG_W), dtype=np.uint8)

                visible_from = current_time - PAST_MS
                visible_to = current_time + ar_ms

                left = max(0, bisect_left(object_times, visible_from) - 2)
                right = min(len(hitobjects), bisect_right(object_times, visible_to) + 2)

                for obj in hitobjects[left:right]:
                    if not object_visible(obj, current_time, ar_ms):
                        continue
                    try:
                        render_hitobject(
                            canvas,
                            obj,
                            current_time,
                            ar_ms,
                            cs_radius_px,
                        )
                    except Exception as render_error:
                        print(
                            f"Render error for object at {obj.get('time', '?')}: "
                            f"{render_error}"
                        )

                cv2.imshow(window_name, canvas)

                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            print(
                f"Time: {current_time:8.1f} ms | "
                f"Objects: {len(hitobjects):5d}   ",
                end="\r",
            )

        except requests.exceptions.RequestException:
            print("Нет связи с tosu. Запущен ли tosu.exe?   ", end="\r")
            time.sleep(0.5)
        except ValueError as error:
            print(f"Ошибка JSON: {error}                      ", end="\r")
            time.sleep(0.2)
        except Exception as error:
            print(f"Ошибка: {error}                      ", end="\r")
            time.sleep(0.2)

    cv2.destroyAllWindows()