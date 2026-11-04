# net.py — BCNet и перенос весов PPO <-> BCNet (единая реализация;
# ранее 5 копий: bc_train, dagger_train x2, reverse_check, logit_probe).
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from sim_env import SimOsuEnv


# ============ СЕТЬ (вход N_STACK каналов; AUX: доп-голова) ============
class BCNet(nn.Module):
    def __init__(self, n_stack=8):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_stack, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, n_stack, 120, 160)).shape[1]
        self.linear = nn.Linear(n, 512)
        self.move_head = nn.Linear(512, 27)
        self.btn_head = nn.Linear(512, 5)
        self.value_head = nn.Linear(512, 1)
        self.aux_head = nn.Linear(512, 2)      # AUX: [dx, dy] к цели

    def forward(self, x):
        f = F.relu(self.linear(self.cnn(x)))
        return (self.move_head(f), self.btn_head(f),
                self.value_head(f).squeeze(-1), self.aux_head(f))


def ppo_to_bcnet(net, pol, dev):
    """Копия весов PPO(CnnPolicy) -> BCNet (без проверок)."""
    net.cnn.load_state_dict(pol.features_extractor.cnn.state_dict())
    net.linear.load_state_dict(pol.features_extractor.linear[0].state_dict())
    with torch.no_grad():
        net.move_head.weight.copy_(pol.action_net.weight[:27]); net.move_head.bias.copy_(pol.action_net.bias[:27])
        net.btn_head.weight.copy_(pol.action_net.weight[27:]);  net.btn_head.bias.copy_(pol.action_net.bias[27:])
        net.value_head.weight.copy_(pol.value_net.weight);      net.value_head.bias.copy_(pol.value_net.bias)


def reverse_transfer(net, ck, dev, n_stack):
    """PPO -> BCNet + инлайн-гейт совпадения (= reverse_check)."""
    m = PPO.load(ck, device="cpu")
    base_k = int(m.policy.features_extractor.cnn[0].in_channels)
    if base_k != n_stack:
        raise SystemExit(
            f"!! База {ck} учится на стеке {base_k}, а DAgger собран на N_STACK={n_stack}.\n"
            f"   Возьми --base из 60Гц-линии (стек 8) либо поменяй N_STACK на {base_k}.")
    pol = m.policy
    ppo_to_bcnet(net, pol, dev)
    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(256):
        s = rng.integers(0, 256, size=(120, 160, n_stack), dtype=np.uint8)
        x = torch.from_numpy(s).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(dev)
        with torch.no_grad():
            lm, lb, _, _ = net(x)
        a, _ = m.predict(s, deterministic=True)
        ok += int(a[0] == int(lm.argmax())) + int(a[1] == int(lb.argmax()))
    acc = ok / 512.0
    print(f"[reverse-check] совпадение: {acc:.3f} (гейт 0.99)")
    if acc < 0.99:
        raise SystemExit("!! перенос не сошёлся — перезапусти iter1 БЕЗ --base (from-scratch)")


def bcnet_to_ppo(net, maps, n_stack, dev, out):
    """Перенос весов BCNet в свежую PPO-обёртку и сохранение (для eval/play)."""
    venv = VecTransposeImage(VecFrameStack(
        DummyVecEnv([lambda: Monitor(SimOsuEnv(maps))]),
        n_stack=n_stack, channels_order="last"))
    model = PPO("CnnPolicy", venv, n_steps=512, batch_size=512,
                learning_rate=5e-5, ent_coef=0.0, verbose=0, device=dev)
    pol = model.policy
    sd = net.state_dict()

    assert len(pol.mlp_extractor.policy_net) == 0, "net_arch не пуст"
    assert pol.action_net.weight.shape == (32, 512)
    assert pol.value_net.weight.shape == (1, 512)

    pol.features_extractor.cnn.load_state_dict(
        {k[len("cnn."):]: v for k, v in sd.items() if k.startswith("cnn.")})
    pol.features_extractor.linear.load_state_dict(
        {"0.weight": sd["linear.weight"], "0.bias": sd["linear.bias"]})
    with torch.no_grad():
        pol.action_net.weight[:27].copy_(sd["move_head.weight"].to(dev))
        pol.action_net.bias[:27].copy_(sd["move_head.bias"].to(dev))
        pol.action_net.weight[27:].copy_(sd["btn_head.weight"].to(dev))
        pol.action_net.bias[27:].copy_(sd["btn_head.bias"].to(dev))
        pol.value_net.weight.copy_(sd["value_head.weight"].to(dev))
        pol.value_net.bias.copy_(sd["value_head.bias"].to(dev))

    model.save(out)
