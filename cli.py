# cli.py — общие REPL-хелперы интерактивных режимов
# (train_stage / diagnose_v2 / servo_v2).
import glob
import os

from models import find_models


def ask(prompt, default=""):
    tail = f" [{default}]" if default != "" else ""
    s = input(f"{prompt}{tail}: ").strip()
    return s if s else default


def ask_int(prompt, default):
    while True:
        try:
            return int(ask(prompt, str(default)))
        except ValueError:
            print("  Нужно целое число, например 5000000")


def ask_float(prompt, default):
    while True:
        try:
            return float(ask(prompt, str(default)))
        except ValueError:
            print("  Нужно число, например 0.15")


def pick_model(default):
    cands = find_models()
    if not cands:
        return ask("Путь к модели (.zip)", default)
    print("Модели:")
    for i, p in enumerate(cands):
        print(f"  {i}: {p}" + ("   <- по умолчанию" if p == default else ""))
    s = input(f"Номер или свой путь [{default}]: ").strip()
    if s == "":
        return default
    if s.isdigit() and int(s) < len(cands):
        return cands[int(s)]
    return s


def pick_maps_dir(default):
    dirs = sorted(d for d in os.listdir(".")
                  if os.path.isdir(d) and glob.glob(os.path.join(d, "*.osu")))
    if dirs:
        print("Папки с картами:", ", ".join(dirs))
    return ask("Папка карт", default)
