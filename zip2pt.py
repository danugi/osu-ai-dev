# zip2pt.py — восстановление BCNet-весов (.pt) из PPO-zip (live200k.zip и т.п.).
# Обратный ход к live2zip.py:(zip был создан из live-чекпоинта, .pt удалён).
#
# Запуск:  python zip2pt.py live200k.zip live200k.pt
# Встроенный гейт: перенос проверяется на 256 случайных состояниях (0.99).
#
# Примечание: aux-голова в PPO-zip не живёт — восстановится случайной.
# Это мягкий регуляризатор (AUX_W=1.0 на 2 выхода), move/btn/value — целые.
import sys

import torch

from net import BCNet, reverse_transfer


def main():
    if len(sys.argv) < 2:
        print("использование: python zip2pt.py <model.zip> [out.pt]")
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(".zip", ".pt")
    n_stack = 8
    net = BCNet(n_stack=n_stack)
    reverse_transfer(net, src, "cpu", n_stack)   # копия весов + гейт совпадения
    torch.save(net.state_dict(), dst)
    print(f"сохранено: {dst} (N_STACK={n_stack})")


if __name__ == "__main__":
    main()
