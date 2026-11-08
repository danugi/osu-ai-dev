#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_maps.py — синтетические .osu карты для локального воспроизведения феномена BC.
Кружки + спиннеры (без слайдеров — для задачи прицела не критично), AR/CS/OD
из диапазона реальных карт, потоки и прыжки."""
import os
import sys
import math
import random

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/z/my-project/scripts/mini_bc_run/sim_maps"
N_MAPS = int(sys.argv[2]) if len(sys.argv) > 2 else 8

TPL = """osu file format v14

[General]
AudioFilename: audio.mp3
StackLeniency: 0.7
Mode: 0

[Metadata]
Title: Synth{sid}
TitleUnicode: Synth{sid}
Artist: glm-repro
Creator: gen_maps
Version: normal

[Difficulty]
HPDrainRate: 5
CircleSize: {cs}
OverallDifficulty: {od}
ApproachRate: {ar}
SliderMultiplier: 1.8
SliderTickRate: 1

[TimingPoints]
0,{bl},4,1,0,50,1,0

[HitObjects]
{objects}
"""


def gen_map(rng, sid):
    ar = rng.choice([8.0, 8.4, 8.8, 9.0, 9.3])
    cs = rng.choice([3.5, 4.0, 4.4, 5.0])
    od = rng.choice([5.0, 6.0, 7.0, 8.0])
    bl = rng.choice([300, 360, 420, 480])       # beat length, мс
    objs = []
    t = 1200.0
    x, y = 256.0, 192.0
    n_total = rng.randint(48, 64)
    while len(objs) < n_total:
        mode = rng.random()
        if mode < 0.22:                          # поток: 4-7 кружков близко и часто
            for _ in range(rng.randint(4, 7)):
                if len(objs) >= n_total:
                    break
                ang = rng.uniform(0, 2 * math.pi)
                r = rng.uniform(35, 85)
                x = min(500, max(12, x + math.cos(ang) * r))
                y = min(372, max(12, y + math.sin(ang) * r))
                objs.append(f"{int(x)},{int(y)},{int(t)},1,0")
                t += rng.uniform(135, 175)
        elif mode < 0.30:                        # спиннер
            dur = rng.randint(1200, 2000)
            objs.append(f"256,192,{int(t)},12,8,{int(t + dur)}")
            t += dur + rng.uniform(300, 450)
            x, y = 256.0, 192.0
        else:                                    # обычный прыжок
            ang = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(60, 330)
            x = min(500, max(12, x + math.cos(ang) * r))
            y = min(372, max(12, y + math.sin(ang) * r))
            objs.append(f"{int(x)},{int(y)},{int(t)},1,0")
            t += rng.uniform(230, 430)
    return TPL.format(sid=sid, cs=cs, od=od, ar=ar, bl=bl, objects="\n".join(objs))


os.makedirs(OUT, exist_ok=True)
for sid in range(N_MAPS):
    rng = random.Random(7000 + sid)
    path = os.path.join(OUT, f"synth{sid:02d}.osu")
    with open(path, "w", encoding="utf-8") as f:
        f.write(gen_map(rng, sid))
    print("карта:", path)
print("готово:", N_MAPS, "карт")
