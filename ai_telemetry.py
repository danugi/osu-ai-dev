"""Лёгкий мост "AI-процесс -> монитор".

Только stdlib, никаких тяжёлых зависимостей и никаких связей с PyTorch.
AI-процессы (play.py / train_stage.py / bc_train.py) вызывают publish(),
а отдельный процесс monitor.py читает файлы и отдаёт данные в веб-UI.

Файлы-сердцебиения (пишутся атомарно, во временный файл + os.replace):
    telemetry_<producer>.json   - метрики конкретного продюсера
    monitor_command.json        - команды управления из UI (пишет monitor.py)

ВАЖНО: все функции молча переживают любые ошибки записи/чтения. Если
монитор не запущен или файл повреждён - AI работает как раньше.
"""
import json
import os
import time

_HERE = os.path.dirname(os.path.abspath(__file__))


def _telemetry_path(producer):
    return os.path.join(_HERE, "telemetry_{}.json".format(producer))


COMMAND_PATH = os.path.join(_HERE, "monitor_command.json")


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def publish(producer, **fields):
    """Записать/обновить сердцебиение продюсера. Никогда не бросает наружу.

    producer: короткое имя ("play", "train_stage", "bc_train").
    fields:   любые JSON-сериализуемые метрики (fps, loss, action, ...).
    """
    try:
        path = _telemetry_path(producer)
        data = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, dict):
                        data = {}
            except Exception:
                data = {}
        data.update(fields)
        data["producer"] = producer
        data["ts"] = time.time()
        _atomic_write_json(path, data)
    except Exception:
        pass


def clear(producer):
    """Убрать файл сердцебиения при корректном выходе продюсера."""
    try:
        path = _telemetry_path(producer)
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def read_command(last_handled_id=0):
    """Прочитать новую команду из UI. Возвращает dict или None.

    Никогда не бросает наружу. Команда считается новой, если её id больше
    last_handled_id - так процесс не выполняет одну команду дважды.
    """
    try:
        if not os.path.exists(COMMAND_PATH):
            return None
        with open(COMMAND_PATH, "r", encoding="utf-8") as f:
            cmd = json.load(f)
        if not isinstance(cmd, dict):
            return None
        cid = int(cmd.get("id", 0))
        if cid <= last_handled_id:
            return None
        return cmd
    except Exception:
        return None
