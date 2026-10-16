"""Локальный веб-монитор AI для наложения через OBS Browser Source.

Архитектура (ничего не ломает, только читает):

    play.py / train_stage.py / bc_train.py  --(telemetry_*.json)-->  monitor.py
    tosu  http://127.0.0.1:24050/json/v2     --(read-only HTTP poll)->  monitor.py
    monitor.py  --(WebSocket 127.0.0.1:8765)-->  monitor_ui.html  -->  OBS

    UI  --(ws msg {cmd})-->  monitor.py  --(monitor_command.json)-->  play.py

Сервер слушает ТОЛЬКО 127.0.0.1:8765. Порт 24050 не занимается и не проксируется:
мы лишь читаем уже существующий tosu-endpoint ровно так же, как это делает env.py.

Запуск:  python monitor.py
Останов: Ctrl+C.
"""
import asyncio
import glob
import json
import os
import subprocess
import time

import requests
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

import ai_telemetry

# ------------------------------------------------------------------
# НАСТРОЙКИ
# ------------------------------------------------------------------
HOST = "127.0.0.1"          # строго loopback: недоступно из сети
PORT = 8765                 # отдельный порт, 24050 не трогаем
TOSU_URL = "http://127.0.0.1:24050/json/v2"   # читаем тот же endpoint, что и env.py

PUSH_HZ = 6                 # частота рассылки снапшота в UI
TOSU_POLL_SEC = 0.10        # как часто опрашиваем tosu
AI_STALE_SEC = 3.0          # сердцебиение AI старше этого -> считаем STOPPED
GPU_QUERY_SEC = 2.0         # nvidia-smi не дёргаем слишком часто

_HERE = os.path.dirname(os.path.abspath(__file__))
UI_HTML = os.path.join(_HERE, "monitor_ui.html")

PRODUCERS = ("play", "train_stage", "bc_train")

# фаза psutil примается один раз
try:
    import psutil
except Exception:
    psutil = None
    _cpu_prime = False
else:
    psutil.cpu_percent(interval=None)  # prime


# ------------------------------------------------------------------
# ЧТЕНИЕ TOSU (только read-only HTTP, без второго tosu)
# ------------------------------------------------------------------
_tosu_errors = 0            # пропуск/ошибки обновления телеметрии на стороне монитора


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _pick(d, *keys):
    """Вернуть первое непустое значение из набора возможных ключей."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def fetch_tosu(session):
    """Один read-only запрос к tosu. Возвращает dict с игровыми метриками."""
    global _tosu_errors
    try:
        r = session.get(TOSU_URL, timeout=1)
        r.raise_for_status()
        st = r.json()
        _tosu_errors = 0
    except Exception:
        _tosu_errors += 1
        return {"connected": False, "errors": _tosu_errors}

    play = st.get("play") or st.get("gameplay") or {}
    hits = play.get("hits", {}) if isinstance(play, dict) else {}

    def h(*names):
        v = _pick(hits, *names)
        return _num(v)

    state = st.get("state", {})
    name = _pick(state, "name") if isinstance(state, dict) else None
    number = _pick(state, "number") if isinstance(state, dict) else None

    bm = st.get("beatmap", {}) or {}
    map_id = _pick(bm, "id", "beatmap_id")
    title = _pick(bm, "title", "displayTitle")
    artist = _pick(bm, "artist", "displayArtist")
    diff = _pick(bm, "difficulty", "version")
    map_label = None
    if title:
        map_label = "{} - {} [{}]".format(title, artist or "?", diff or "?")
    elif map_id is not None:
        map_label = "beatmap #{}".format(map_id)

    # моды: tosu то ли даёт готовый список, то ли число
    mods_raw = _pick(play, "mods", "mod_array", "mods_number") if isinstance(play, dict) else None
    mods_str = None
    if isinstance(mods_raw, list):
        mods_str = " ".join(str(m) for m in mods_raw) if mods_raw else "nomod"
    elif isinstance(mods_raw, dict):
        nm = mods_raw.get("acronym") or mods_raw.get("name")
        if nm:
            mods_str = str(nm)
        else:
            num = _num(mods_raw.get("number"))
            mods_str = "mods={}".format(int(num)) if num is not None else None
    elif isinstance(mods_raw, (int, float)):
        mods_str = "mods={}".format(int(mods_raw))
    if mods_str == "":
        mods_str = "nomod"

    acc = _num(_pick(play, "accuracy", "accuracy_exact"))
    if acc is not None and acc <= 1.0:
        acc = acc * 100.0  # tosu иногда отдаёт долю, иногда процент

    return {
        "connected": True,
        "errors": 0,
        "state": name if name is not None else ("playing" if number == 2 else "other"),
        "state_number": _num(number),
        "score": _num(_pick(play, "score", "display_score", " DisplayScore")),
        "accuracy": round(acc, 2) if acc is not None else None,
        "combo": _num(_pick(play, "combo", "current_combo", "currentCombo")),
        "max_combo": _num(_pick(play, "max_combo", "maxCombo")),
        "hits": {
            "300": h("300", "count300", "c300", "n300"),
            "100": h("100", "count100", "c100", "n100"),
            "50": h("50", "count50", "c50", "n50"),
            "miss": h("miss", "countmiss", "cMiss", "misses", "nmiss"),
        },
        "mods": mods_str,
        "map": map_label,
    }


# ------------------------------------------------------------------
# СИСТЕМНЫЕ МЕТРИКИ (опционально: psutil / nvidia-smi). N/A если нет.
# ------------------------------------------------------------------
_gpu_cache = {"ts": 0.0, "gpu": None, "vram_used": None, "vram_total": None}


def read_cpu():
    if psutil is None:
        return None
    try:
        return round(psutil.cpu_percent(interval=None), 1)
    except Exception:
        return None


def read_gpu():
    """Утилизация GPU и VRAM через nvidia-smi (без новых python-зависимостей)."""
    now = time.time()
    if now - _gpu_cache["ts"] < GPU_QUERY_SEC:
        return _gpu_cache
    _gpu_cache["ts"] = now
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=1.5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        line = out.stdout.strip().splitlines()[0]
        util, used, total = [float(x.strip()) for x in line.split(",")[:3]]
        _gpu_cache["gpu"] = round(util, 0)
        _gpu_cache["vram_used"] = int(used)
        _gpu_cache["vram_total"] = int(total)
    except Exception:
        _gpu_cache["gpu"] = None
        _gpu_cache["vram_used"] = None
        _gpu_cache["vram_total"] = None
    return _gpu_cache


# ------------------------------------------------------------------
# ЧТЕНИЕ МЕТРИК AI из telemetry_*.json
# ------------------------------------------------------------------
def _load_telemetry():
    now = time.time()
    best = None
    best_age = 1e9
    for p in PRODUCERS:
        path = os.path.join(_HERE, "telemetry_{}.json".format(p))
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        age = now - float(d.get("ts", 0) or 0)
        d["_age"] = age
        if age < AI_STALE_SEC and age < best_age:
            best, best_age = d, age
    return best


def build_ai_block():
    d = _load_telemetry()
    if not d:
        return {"status": "STOPPED", "producer": None, "model_loaded": False,
                "training": False, "fps": None, "inference_ms": None,
                "step_reward": None, "episode_reward": None, "loss": None,
                "train_steps": None, "model": None, "action": None,
                "cursor": None, "ai_connection": None, "telemetry_errors": None}

    producer = d.get("producer")
    training = producer in ("train_stage", "bc_train")
    if producer == "play":
        status = d.get("ai_status") or "RUNNING"
    else:
        status = d.get("ai_status") or "RUNNING"
    return {
        "status": status,
        "producer": producer,
        "training": training,
        "model_loaded": bool(d.get("model_loaded")),
        "model": d.get("model"),
        "fps": d.get("fps"),
        "inference_ms": d.get("inference_ms"),
        "step_reward": d.get("step_reward"),
        "episode_reward": d.get("episode_reward"),
        "loss": d.get("loss"),
        "train_steps": d.get("train_steps"),
        "action": d.get("action"),
        "cursor": d.get("cursor"),
        "ai_connection": d.get("ai_connection"),
        "telemetry_errors": d.get("telemetry_errors"),
    }


def build_snapshot(tosu):
    gpu = read_gpu()
    return {
        "t": time.time(),
        "ai": build_ai_block(),
        "tosu": tosu,
        "sys": {
            "cpu": read_cpu(),
            "gpu": gpu["gpu"],
            "vram_used_mb": gpu["vram_used"],
            "vram_total_mb": gpu["vram_total"],
        },
    }


# ------------------------------------------------------------------
# УПРАВЛЕНИЕ: команды из UI -> monitor_command.json (выполняет только play.py)
# ------------------------------------------------------------------
_cmd_id = 0


def post_command(cmd):
    global _cmd_id
    if cmd not in ("pause", "resume", "save", "reload"):
        return
    _cmd_id += 1
    try:
        ai_telemetry._atomic_write_json(
            ai_telemetry.COMMAND_PATH,
            {"cmd": cmd, "id": _cmd_id, "ts": time.time()},
        )
    except Exception:
        pass


# ------------------------------------------------------------------
# WEB-СЕРВЕР (HTTP + WebSocket на одном порту)
# ------------------------------------------------------------------
CLIENTS = set()


def _serve_html(connection, request):
    """Отдать HTML по обычному GET; пустой ответ -> продолжить WS-handshake."""
    upgrade = (request.headers.get("Upgrade") or "").lower()
    if upgrade == "websocket":
        return None  # это запрос на открытие WebSocket
    try:
        with open(UI_HTML, "rb") as f:
            body = f.read()
    except Exception:
        body = b"<h3>monitor_ui.html not found next to monitor.py</h3>"
    headers = Headers({
        "Content-Type": "text/html; charset=utf-8",
        "Cache-Control": "no-store",
        "Access-Control-Allow-Origin": "*",
    })
    return Response(200, "OK", headers, body)


async def handler(ws):
    CLIENTS.add(ws)
    try:
        async for msg in ws:
            try:
                obj = json.loads(msg)
                if obj.get("type") == "command":
                    post_command(obj.get("cmd"))
            except Exception:
                pass
    finally:
        CLIENTS.discard(ws)


async def broadcaster():
    session = requests.Session()
    last_tosu = {"connected": False, "errors": _tosu_errors}
    last_poll = 0.0
    period = 1.0 / PUSH_HZ
    while True:
        loop = asyncio.get_running_loop()
        now = time.time()
        if now - last_poll >= TOSU_POLL_SEC:
            last_poll = now
            try:
                last_tosu = await loop.run_in_executor(None, fetch_tosu, session)
            except Exception:
                last_tosu = {"connected": False, "errors": _tosu_errors}
        snap = await loop.run_in_executor(None, build_snapshot, last_tosu)
        payload = json.dumps(snap, ensure_ascii=False)
        for ws in list(CLIENTS):
            try:
                await ws.send(payload)
            except Exception:
                CLIENTS.discard(ws)
        await asyncio.sleep(period)


async def amain():
    os.chdir(_HERE)   # telemetry-файлы ищем рядом с проектом
    async with serve(handler, HOST, PORT, process_request=_serve_html):
        print(f"[monitor] UI: http://{HOST}:{PORT}/   (OBS Browser Source -> Local URL)")
        print(f"[monitor] tosu читается read-only из {TOSU_URL} (порт не занимается)")
        await broadcaster()


if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\n[monitor] остановлен.")
