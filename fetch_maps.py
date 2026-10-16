import requests
import json
import os
import time
import vision

# ============================================================
# НАСТРОЙКИ
# ============================================================
API_KEY = ""   # ключ v1 (osu.ppy.sh/home/account/edit -> Legacy API). Нужен только для set/map

MAP_DIR = "sim_maps"
MANIFEST = os.path.join(MAP_DIR, "manifest.json")
HEADERS = {"User-Agent": "Mozilla/5.0"}

STAR_RANGE = (1.5, 2.0)   # фильтр звёздности (режимы set/map)
LEN_RANGE = (20, 150)     # фильтр длины, секунды
PAUSE = 1.5               # пауза между скачиваниями
MIN_OBJECTS = 15          # меньше объектов — мусор


# ============================================================
# MANIFEST
# ============================================================
def load_manifest():
    if os.path.exists(MANIFEST):
        with open(MANIFEST, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(m):
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)


def maps_only(manifest):
    return {k: v for k, v in manifest.items() if not k.startswith("__")}


# ============================================================
# API / СКАЧИВАНИЕ
# ============================================================
def api_get_beatmaps(**params):
    params["k"] = API_KEY
    try:
        r = requests.get("https://osu.ppy.sh/api/get_beatmaps",
                         params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            print("  API вернул не список (проверь ключ): " + str(data)[:80])
            return None
        return data
    except Exception as e:
        print(f"  Сетевая ошибка API: {e}")
        return None


def download_osu(beatmap_id):
    """Прямая ссылка, ключ не нужен. Сохраняем байты как есть."""
    try:
        r = requests.get(f"https://osu.ppy.sh/osu/{beatmap_id}",
                         headers=HEADERS, timeout=15)
    except Exception as e:
        print(f"  {beatmap_id}: сеть: {e}")
        return None
    if r.status_code != 200 or b"[HitObjects]" not in r.content:
        print(f"  {beatmap_id}: не скачался (код {r.status_code})")
        return None
    path = os.path.join(MAP_DIR, f"{beatmap_id}.osu")
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def parse_check(path):
    """Страховка: карта должна парситься твоим vision.py и быть не пустой.
    Возвращает (objs, ar_ms, cs, od) или None."""
    try:
        import vision
        objs, ar_ms, cs, w50 = vision.parse_osu_file(path)
        if len(objs) >= MIN_OBJECTS:
            od = (200.0 - w50) / 10.0 if w50 >= 150.0 else (190.0 - w50) / 8.0
            return objs, ar_ms, cs, od
    except Exception:
        pass
    return None


def save_entry(manifest, bid, sr, ar, cs, od, length, version, set_id):
    manifest[bid] = {
        "sr": sr, "ar": round(ar, 2), "cs": round(cs, 2),
        "od": round(od, 2), "length": length,
        "version": version, "set": str(set_id),
    }
    save_manifest(manifest)


# ============================================================
# РЕЖИМЫ ЗАГРУЗКИ
# ============================================================
def try_fetch(entry, manifest):
    bid = str(entry["beatmap_id"])
    if bid in manifest:
        print(f"  {bid}: уже есть, скип")
        return
    if int(entry.get("mode", -1)) != 0:
        return  # не osu! standard (конверты и т.п.)

    sr = float(entry.get("difficultyrating", 0.0))
    length = int(entry.get("total_length", 0))
    if not (STAR_RANGE[0] <= sr <= STAR_RANGE[1]):
        print(f"  {bid}: sr={sr:.2f} вне {STAR_RANGE[0]}–{STAR_RANGE[1]}, скип")
        return
    if not (LEN_RANGE[0] <= length <= LEN_RANGE[1]):
        print(f"  {bid}: длина {length}s, скип")
        return

    path = download_osu(bid)
    if path is None:
        return
    parsed = parse_check(path)
    if parsed is None:
        print(f"  {bid}: парсер не съел, удаляю")
        os.remove(path)
        return

    save_entry(manifest, bid, sr,
               float(entry.get("diff_approach", 0)),
               float(entry.get("diff_size", 0)),
               float(entry.get("diff_overall", 0)),
               length, entry.get("version", ""), entry.get("beatmapset_id", ""))
    print(f"  {bid}: OK  sr={sr:.2f}  [{entry.get('version','')}]  {length}s")
    time.sleep(PAUSE)


def fetch_set(set_id, manifest):
    print(f"Сет {set_id}:")
    entries = api_get_beatmaps(s=set_id, m=0, limit=100)
    if entries is None:
        return
    if not entries:
        print("  пусто. Может, это ID сложности, а не сета? Попробуй режим map.")
        return
    for e in entries:
        try_fetch(e, manifest)


def fetch_map(beatmap_id, manifest):
    print(f"Карта {beatmap_id}:")
    entries = api_get_beatmaps(b=str(beatmap_id), m=0)
    if entries is None:
        return
    if not entries:
        print("  пусто. Может, это ID сета? Попробуй режим set.")
        return
    try_fetch(entries[0], manifest)


def fetch_raw(beatmap_id, manifest):
    """Без API: качаем .osu напрямую, параметры достаём из самого файла.
    SR в файле нет — фильтруй по звёздам на сайте сам."""
    bid = str(beatmap_id)
    print(f"Raw {bid}:")
    if bid in manifest:
        print(f"  {bid}: уже есть, скип")
        return
    path = download_osu(bid)
    if path is None:
        return
    parsed = parse_check(path)
    if parsed is None:
        print("  парсер не съел, удаляю")
        os.remove(path)
        return
    objs, ar_ms, cs, od = parsed
    ar = vision.ar_value_from_ms(ar_ms)
    length = int((objs[-1]["time"] + objs[-1]["duration"]) / 1000) + 1
    save_entry(manifest, bid, None, ar, cs, od, length, "raw", "")
    print(f"  {bid}: OK  AR={ar:.1f} CS={cs:.2f} OD={od:.1f}  {length}s  "
          f"({len(objs)} объектов)")
    time.sleep(PAUSE)


# ============================================================
# СТАТИСТИКА
# ============================================================
def print_stats(manifest):
    maps = maps_only(manifest)
    if not maps:
        print("Датасет пуст.")
        return
    srs = sorted(v["sr"] for v in maps.values() if v.get("sr") is not None)
    ars = sorted(v["ar"] for v in maps.values())
    css = sorted(v["cs"] for v in maps.values())
    ods = sorted(v["od"] for v in maps.values())
    total = sum(v["length"] for v in maps.values())
    sets = len({v["set"] for v in maps.values() if v["set"]})
    print(f"  Карт: {len(maps)} (из {sets} сетов)")
    if srs:
        print(f"  SR: {srs[0]:.2f}–{srs[-1]:.2f}   "
              f"(медиана {srs[len(srs)//2]:.2f})")
    print(f"  AR: {ars[0]:.1f}–{ars[-1]:.1f} | CS: {css[0]:.1f}–{css[-1]:.1f} | "
          f"OD: {ods[0]:.1f}–{ods[-1]:.1f}")
    print(f"  Суммарная длина: {total // 60} мин")


# ============================================================
# ИНТЕРАКТИВНЫЙ ЦИКЛ
# ============================================================
def help_text():
    print("Команды:")
    print("  set            — режим мапсетов (по умолчанию)")
    print("  map            — режим отдельных сложностей (с SR-фильтром)")
    print("  raw            — качать без API-ключа (SR-фильтра нет)")
    print("  range 1.5 2.0  — сменить фильтр звёздности")
    print("  len 20 150     — сменить фильтр длины (сек)")
    print("  stats          — сводка датасета")
    print("  q              — выход")
    print("ID вводи по одному или пачкой через пробел/запятую. Пустая строка — ничего.")
    print("  dir maps_20    — сменить папку назначения")

if __name__ == "__main__":
    os.makedirs(MAP_DIR, exist_ok=True)
    manifest = load_manifest()
    mode = "set"

    print("=== Сборщик карт для симулятора ===")
    print(f"Фильтр: SR {STAR_RANGE[0]}–{STAR_RANGE[1]}★, "
          f"длина {LEN_RANGE[0]}–{LEN_RANGE[1]}с, пауза {PAUSE}с")
    print(f"В датасете сейчас: {len(maps_only(manifest))} карт")
    if not API_KEY and input("API-ключ не задан. Использовать режим raw? (y/n): ").lower().startswith("y"):
        mode = "raw"
    help_text()

    while True:
        try:
            raw = input(f"\n[{mode}] id> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            continue
        low = raw.lower()

        if low in ("q", "quit", "exit", "выход"):
            break
        if low in ("s", "set"):
            mode = "set"
            print(f"Режим: мапсеты (ID из URL beatmapsets/...)")
            continue
        if low in ("m", "map"):
            mode = "map"
            print("Режим: отдельные сложности (ID из URL ...#osu/123456)")
            continue
        if low == "raw":
            mode = "raw"
            print("Режим: raw, без API. SR фильтруй сам на сайте.")
            continue
        if low in ("h", "help", "?"):
            help_text()
            continue
        if low == "stats":
            print_stats(manifest)
            continue
        if low.startswith("dir"):
            parts = raw.split()
            if len(parts) >= 2:
                MAP_DIR = parts[1]
                MANIFEST = os.path.join(MAP_DIR, "manifest.json")
                os.makedirs(MAP_DIR, exist_ok=True)
                manifest = load_manifest()
                print(f"Папка: {MAP_DIR} (карт: {len(maps_only(manifest))})")
            else:
                print(f"Текущая папка: {MAP_DIR}")
            continue
        if low.startswith("range"):
            parts = raw.split()
            try:
                STAR_RANGE = (float(parts[1]), float(parts[2]))
                print(f"SR-фильтр: {STAR_RANGE[0]}–{STAR_RANGE[1]}")
            except (IndexError, ValueError):
                print("Пример: range 1.5 2.0")
            continue
        if low.startswith("len"):
            parts = raw.split()
            try:
                LEN_RANGE = (int(parts[1]), int(parts[2]))
                print(f"Фильтр длины: {LEN_RANGE[0]}–{LEN_RANGE[1]}с")
            except (IndexError, ValueError):
                print("Пример: len 20 150")
            continue

        ids = [t for t in raw.replace(",", " ").split() if t.isdigit()]
        if not ids:
            print("Не понял. Вводи числовые ID или команду (help — список).")
            continue

        for i in ids:
            try:
                if mode == "set":
                    fetch_set(i, manifest)
                elif mode == "map":
                    fetch_map(i, manifest)
                else:
                    fetch_raw(i, manifest)
            except Exception as e:
                print(f"  {i}: ошибка: {e}")
                time.sleep(1)

    print("\n=== Итог ===")
    print_stats(manifest)
    print("\nДальше: python sim_env.py  (проверка карт + замер скорости)")
