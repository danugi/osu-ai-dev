# live2zip.py — конвертация live-чекпоинта DAgger (BCNet .pt) в PPO-zip,
# чтобы мерить разрыв (dagger_gap / diagnose_v3) НЕ останавливая обучение.
#
# Запуск (в корне проекта, рядом с dagger_train.py):
#   Copy-Item dagger_iter1_live.pt live50k.pt   # снапшот: файл перезапишется на 100k
#   python live2zip.py live50k.pt               # -> live50k.zip + самопроверка переноса
#   python dagger_gap.py live50k.zip --maps sim_maps --steps 4000
#
# Работает на CPU и не трогает обучающий процесс на GPU.
import glob
import sys

import numpy as np
import torch
from stable_baselines3 import PPO

from net import BCNet, bcnet_to_ppo
from models import infer_stack_file


def main():
    if len(sys.argv) < 2:
        print("использование: python live2zip.py <..._live.pt> [папка_карт]")
        sys.exit(1)
    src = sys.argv[1]
    maps_dir = sys.argv[2] if len(sys.argv) > 2 else "sim_maps"
    maps = sorted(glob.glob(maps_dir + "/*.osu"))
    assert maps, "нет карт в " + maps_dir

    n_stack = infer_stack_file(src)              # стек — из весов, как в dagger_train
    net = BCNet(n_stack=n_stack)
    net.load_state_dict(torch.load(src, map_location="cpu"))
    out = src[:-3] if src.endswith(".pt") else src + "_conv"
    if n_stack == 8:
        bcnet_to_ppo(net, maps, n_stack, "cpu", out)
    else:
        # coords-стеки (10/11/12/13 каналов): обёртка CoordStackEnv вместо
        # VecFrameStack — семантика та же, каналы доп-плоскостей из env
        from coords import bcnet_to_ppo_wrap
        bcnet_to_ppo_wrap(net, maps, n_stack, "cpu", out)
        print(f"[coords] стек {n_stack} каналов — PPO собран через CoordStackEnv")

    # самопроверка переноса (как reverse-check в dagger_train): argmax BCNet == predict PPO
    m = PPO.load(out, device="cpu")
    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(128):
        s = rng.integers(0, 256, size=(120, 160, n_stack), dtype=np.uint8)
        x = torch.from_numpy(s).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
        with torch.no_grad():
            lm, lb, _, _ = net(x)
        a, _ = m.predict(s, deterministic=True)
        ok += int(a[0] == int(lm.argmax())) + int(a[1] == int(lb.argmax()))
    acc = ok / 256.0
    print(f"[check] перенос BCNet->PPO: {acc:.3f} (гейт 0.99)")
    if acc < 0.99:
        print("!! перенос не сошёлся — не доверяй этому zip, пришли лог")
        sys.exit(1)
    print(f"готово: {out}.zip (N_STACK={n_stack})")
    print(f"дальше: python dagger_gap.py {out}.zip --maps {maps_dir} --steps 4000")


if __name__ == "__main__":
    main()
