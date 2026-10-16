# Проверяет ТОЛЬКО корректность копирования весов (случайные кадры):
# кривая копия даст ~1/27, правильная — 1.000.
import numpy as np, torch
from stable_baselines3 import PPO
from bc_train import BCNet

CK, N_STACK = "best_stage2b/stage2b_800000_steps.zip", 4

model = PPO.load(CK, device="cpu")
net = BCNet(n_stack=N_STACK); net.eval()
pol = model.policy

net.cnn.load_state_dict(pol.features_extractor.cnn.state_dict())
net.linear.load_state_dict(pol.features_extractor.linear[0].state_dict())
with torch.no_grad():
    net.move_head.weight.copy_(pol.action_net.weight[:27]); net.move_head.bias.copy_(pol.action_net.bias[:27])
    net.btn_head.weight.copy_(pol.action_net.weight[27:]);  net.btn_head.bias.copy_(pol.action_net.bias[27:])
    net.value_head.weight.copy_(pol.value_net.weight);      net.value_head.bias.copy_(pol.value_net.bias)

rng = np.random.default_rng(0)
mm = mb = 0
N = 512
for _ in range(N):
    s = rng.integers(0, 256, size=(120, 160, N_STACK), dtype=np.uint8)
    x = torch.from_numpy(s).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        lm, lb, _, _ = net(x)
    a, _ = model.predict(s, deterministic=True)
    mm += int(a[0] == int(lm.argmax())); mb += int(a[1] == int(lb.argmax()))
print(f"move: {mm/N:.3f} | btn: {mb/N:.3f}   (гейт >= 0.99; иначе — DAgger from-scratch)")
maxd = 0.0
for _ in range(64):
    s = rng.integers(0, 256, size=(120, 160, N_STACK), dtype=np.uint8)
    x = torch.from_numpy(s).permute(2, 0, 1).unsqueeze(0).float()
    with torch.no_grad():
        d = model.policy.get_distribution(x)
        bl, _, _, _ = net(x / 255.0)
    maxd = max(maxd, (d.distribution[0].logits - bl).abs().max().item())
print(f"max |Δlogits|: {maxd:.5f}   (гейт < 1e-2)")