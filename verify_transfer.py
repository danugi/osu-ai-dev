#!/usr/bin/env python3
"""Оффлайн-проверка переноса BCNet -> PPO по ФАЙЛАМ весов.

Закрывает вопрос «перенос чист?» без хвоста лога обучения (например,
окно консоли уже закрыто, или прогон печатал старый гейт со ложной
тревогой — см. CHANGES-STAGE3.md п.3). Гейт здесь уже исправленный
(log_softmax с обеих сторон), тот же, что в bc_train stage3.

Запуск (в корне osu-ai-dev):

    python verify_transfer.py bc_policy_fresh

Аргумент — имя весов БЕЗ расширения; читаются <имя>.zip и <имя>_bcnet.pt.
Стек берётся из самих весов (models.infer_stack_file).
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO

from models import infer_stack_file
from net import BCNet


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "bc_policy_fresh"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    n_stack = infer_stack_file(base + ".zip")
    print(f"файлы: {base}.zip + {base}_bcnet.pt | стек из весов: {n_stack}")

    # 3.2.3: сеть — на тот же dev, что и вход; иначе на CUDA-машине
    # (Colab, GTX 1060) вход cuda против весов cpu -> RuntimeError
    net = BCNet(n_stack=n_stack).to(dev)
    net.load_state_dict(torch.load(base + "_bcnet.pt", map_location="cpu"))
    net.eval()
    m2 = PPO.load(base, device=dev)

    rng = np.random.default_rng(0)
    batch_hwc = rng.integers(0, 256, size=(512, 120, 160, n_stack), dtype=np.uint8)
    with torch.no_grad():
        x = torch.from_numpy(batch_hwc).to(dev).permute(0, 3, 1, 2).float() / 255.0
        lm, lb, lv, _ = net(x)
        x_raw = torch.from_numpy(batch_hwc).to(dev).permute(0, 3, 1, 2).float()
        v_ppo = m2.policy.predict_values(x_raw).squeeze(-1)

    a2, _ = m2.predict(batch_hwc, deterministic=True)
    match_m = (a2[:, 0] == lm.argmax(1).cpu().numpy()).mean()
    match_b = (a2[:, 1] == lb.argmax(1).cpu().numpy()).mean()
    v_corr = (float(np.corrcoef(v_ppo.cpu().numpy(), lv.cpu().numpy())[0, 1])
              if lv.std() > 1e-8 and v_ppo.std() > 1e-8 else 0.0)

    # Δlogits-гейт (исправленный): распределения PPO и BCNet обязаны
    # совпасть после log_softmax (torch Categorical вычитает logsumexp)
    maxd = 0.0
    for _ in range(64):
        s = rng.integers(0, 256, size=(120, 160, n_stack), dtype=np.uint8)
        xx = torch.from_numpy(s).to(dev).permute(2, 0, 1).unsqueeze(0).float()
        with torch.no_grad():
            d = m2.policy.get_distribution(xx)
            bl, bb, _, _ = net(xx / 255.0)
        maxd = max(maxd,
                   (d.distribution[0].logits - F.log_softmax(bl, -1)).abs().max().item(),
                   (d.distribution[1].logits - F.log_softmax(bb, -1)).abs().max().item())

    print(f"match move {match_m:.3f} | btn {match_b:.3f} | value corr {v_corr:.3f} "
          f"| max |Δlogits| {maxd:.2e}")
    ok = match_m > 0.99 and match_b > 0.99 and v_corr > 0.99 and maxd < 1e-2
    print("ИТОГ:", "PASS — перенос чист (хвост лога не нужен)" if ok
          else "FAIL — пришли мне этот вывод целиком")


if __name__ == "__main__":
    main()
