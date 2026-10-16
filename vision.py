import requests
import cv2
import numpy as np
import ctypes
import math
import os
import time
import copy
from bisect import bisect_left, bisect_right


# ============================================================
# WINDOWS API
# ============================================================
user32 = ctypes.windll.user32


# ============================================================
# ENVIRONMENT SETTINGS
# ============================================================
# osu! standard playfield is 512x384 (4:3).
OSU_W, OSU_H = 512.0, 384.0
IMG_W, IMG_H = 320, 240

# Small visual margin around the osu! playfield so edge objects
# and reverse markers are not clipped.
PLAYFIELD_MARGIN_X = 10
PLAYFIELD_MARGIN_Y = 8
PLAY_W = IMG_W - PLAYFIELD_MARGIN_X * 2
PLAY_H = IMG_H - PLAYFIELD_MARGIN_Y * 2

TOSU_URL = "http://127.0.0.1:24050/json/v2"

# How much of the past stays visible.
PAST_MS = 40.0

# Fade timings.
HIT_FADE_MS = 150.0
FADE_IN_MS = 100.0

# Grayscale semantics:
# 255 = hit objects & slider ball, 180 = spinner,
# 120 = slider body, 100 = approach ring.
OBJ_VALUE = 255
APPROACH_VALUE = 100
SLIDER_BODY_VALUE = 120
SPINNER_VALUE = 180
CURSOR_VALUE  = 255   # ядро
CURSOR_RADIUS = 6     # радиус ядра (~3px на входе сети)
CURSOR_HALO   = 3     # толщина чёрной обводки (~1.5px на входе)


# ============================================================
# MODS (биты из tosu: play.mods.number)
# ============================================================
MOD_EZ = 1 << 1   # 2
MOD_HD = 1 << 3   # 8
MOD_HR = 1 << 4   # 16
MOD_DT = 1 << 6   # 64
MOD_HT = 1 << 8   # 256
MOD_NC = 1 << 9   # 512

# Активные моды — влияет на рендер (HD).
# Ставится снаружи: vision.ACTIVE_MODS = mods
ACTIVE_MODS = 0

# Hidden: за какую долю preempt объект гаснет до удара.
HD_FADE_FRAC = 0.45


# ============================================================
# COORDINATES
# ============================================================
def to_canvas(x, y):
    """osu! playfield -> 320x240 vision canvas with a small safety margin."""
    cx = int(round(PLAYFIELD_MARGIN_X + x * PLAY_W / OSU_W))
    cy = int(round(PLAYFIELD_MARGIN_Y + y * PLAY_H / OSU_H))
    return (
        min(IMG_W - 1, max(0, cx)),
        min(IMG_H - 1, max(0, cy)),
    )


# ============================================================
# CURVE APPROXIMATION
# ============================================================
class PathApproximator:
    @staticmethod
    def approximate_linear(control_points):
        return list(control_points)

    @staticmethod
    def approximate_bezier(control_points):
        if not control_points:
            return []
        if len(control_points) == 1:
            return list(control_points)

        output, segments, current_seg = [], [], [control_points[0]]

        for p in control_points[1:]:
            if p == current_seg[-1] and len(current_seg) > 1:
                segments.append(current_seg)
                current_seg = [p]
            else:
                current_seg.append(p)

        segments.append(current_seg)

        for seg in segments:
            if len(seg) == 1:
                continue
            PathApproximator._bezier_recursive(seg, output)
            if len(output) > 1 and output[-1] == output[-2]:
                output.pop()

        return output

    @staticmethod
    def _bezier_recursive(points, output):
        if PathApproximator._is_flat_enough(points):
            if not output:
                output.append(points[0])
            output.extend(points[1:])
            return

        left, right = PathApproximator._subdivide(points)
        PathApproximator._bezier_recursive(left, output)
        PathApproximator._bezier_recursive(right, output)

    @staticmethod
    def _is_flat_enough(points):
        for i in range(1, len(points) - 1):
            p1 = points[i - 1]
            p2 = points[i]
            p3 = points[i + 1]
            vec_x = p1[0] - 2 * p2[0] + p3[0]
            vec_y = p1[1] - 2 * p2[1] + p3[1]
            if vec_x * vec_x + vec_y * vec_y > 0.25 ** 2:
                return False
        return True

    @staticmethod
    def _subdivide(points):
        n = len(points)
        midpoints = list(points)
        left = [None] * n
        right = [None] * n

        for i in range(n):
            left[i] = midpoints[0]
            right[n - i - 1] = midpoints[n - i - 1]
            for j in range(n - i - 1):
                midpoints[j] = (
                    (midpoints[j][0] + midpoints[j + 1][0]) / 2.0,
                    (midpoints[j][1] + midpoints[j + 1][1]) / 2.0,
                )

        return left, right

    @staticmethod
    def approximate_catmull(control_points):
        if not control_points:
            return []
        if len(control_points) == 1:
            return list(control_points)

        result = []
        catmull_detail = 50

        for i in range(len(control_points) - 1):
            v1 = control_points[i - 1] if i > 0 else control_points[i]
            v2 = control_points[i]
            v3 = (
                control_points[i + 1]
                if i < len(control_points) - 1
                else (v2[0] * 2 - v1[0], v2[1] * 2 - v1[1])
            )
            v4 = (
                control_points[i + 2]
                if i < len(control_points) - 2
                else (v3[0] * 2 - v2[0], v3[1] * 2 - v2[1])
            )

            for c in range(catmull_detail + 1):
                p = PathApproximator._catmull_find_point(v1, v2, v3, v4, c / catmull_detail)
                if not result or p != result[-1]:
                    result.append(p)

        return result

    @staticmethod
    def _catmull_find_point(vec1, vec2, vec3, vec4, t):
        t2 = t * t
        t3 = t2 * t

        x = 0.5 * (
            2 * vec2[0]
            + (-vec1[0] + vec3[0]) * t
            + (2 * vec1[0] - 5 * vec2[0] + 4 * vec3[0] - vec4[0]) * t2
            + (-vec1[0] + 3 * vec2[0] - 3 * vec3[0] + vec4[0]) * t3
        )
        y = 0.5 * (
            2 * vec2[1]
            + (-vec1[1] + vec3[1]) * t
            + (2 * vec1[1] - 5 * vec2[1] + 4 * vec3[1] - vec4[1]) * t2
            + (-vec1[1] + 3 * vec2[1] - 3 * vec3[1] + vec4[1]) * t3
        )

        return x, y

    @staticmethod
    def approximate_circular_arc(control_points):
        if len(control_points) < 3:
            return list(control_points)

        p1, p2, p3 = control_points[:3]
        cross = (
            (p2[0] - p1[0]) * (p3[1] - p1[1])
            - (p2[1] - p1[1]) * (p3[0] - p1[0])
        )
        if abs(cross) < 1e-6:
            return PathApproximator.approximate_bezier(control_points)

        ax, ay = p1
        bx, by = p2
        cx, cy = p3
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-6:
            return PathApproximator.approximate_bezier(control_points)

        ux = (
            (ax * ax + ay * ay) * (by - cy)
            + (bx * bx + by * by) * (cy - ay)
            + (cx * cx + cy * cy) * (ay - by)
        ) / d
        uy = (
            (ax * ax + ay * ay) * (cx - bx)
            + (bx * bx + by * by) * (ax - cx)
            + (cx * cx + cy * cy) * (bx - ax)
        ) / d

        center = (ux, uy)
        radius = math.hypot(ax - ux, ay - uy)

        theta_start = math.atan2(p1[1] - center[1], p1[0] - center[0])
        theta_end = math.atan2(p3[1] - center[1], p3[0] - center[0])
        theta_mid = math.atan2(p2[1] - center[1], p2[0] - center[0])

        while theta_mid < theta_start:
            theta_mid += 2 * math.pi
        while theta_end < theta_start:
            theta_end += 2 * math.pi

        if theta_mid > theta_end:
            theta_start, theta_end = theta_end, theta_start + 2 * math.pi

        theta_range = theta_end - theta_start
        circular_arc_tolerance = 0.1

        if abs(theta_range) < 1e-6 or 2 * radius <= circular_arc_tolerance:
            amount_points = 2
        else:
            acos_arg = max(-1.0, min(1.0, 1 - circular_arc_tolerance / radius))
            divisor = 2 * math.acos(acos_arg)
            amount_points = (
                max(2, math.ceil(abs(theta_range) / divisor))
                if divisor > 0
                else 2
            )

        output = []
        for i in range(amount_points):
            fract = i / (amount_points - 1)
            theta = theta_start + fract * theta_range
            output.append(
                (
                    math.cos(theta) * radius + center[0],
                    math.sin(theta) * radius + center[1],
                )
            )

        return output


def flatten_curve(curve_type, points):
    if not points:
        return []
    if len(points) == 1:
        return [points[0]]

    if curve_type == "L":
        result = PathApproximator.approximate_linear(points)
    elif curve_type == "P":
        result = (
            PathApproximator.approximate_circular_arc(points)
            if len(points) == 3
            else PathApproximator.approximate_bezier(points)
        )
    elif curve_type == "C":
        result = PathApproximator.approximate_catmull(points)
    else:
        result = PathApproximator.approximate_bezier(points)

    if len(result) < 2 and len(points) > 1:
        result = [points[0], points[-1]]

    return result if result else list(points)


def get_path_length(points):
    if len(points) < 2:
        return 0.0
    return sum(
        math.hypot(
            points[i + 1][0] - points[i][0],
            points[i + 1][1] - points[i][1],
        )
        for i in range(len(points) - 1)
    )


def get_point_on_curve(points, progress):
    if not points:
        return 0.0, 0.0
    if len(points) == 1:
        return points[0]

    total_dist = get_path_length(points)
    if total_dist <= 0:
        return points[0]

    target_dist = total_dist * max(0.0, min(1.0, progress))
    current_dist = 0.0

    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        seg_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        if seg_dist <= 0:
            continue

        if current_dist + seg_dist >= target_dist:
            seg_progress = (target_dist - current_dist) / seg_dist
            return (
                p1[0] + (p2[0] - p1[0]) * seg_progress,
                p1[1] + (p2[1] - p1[1]) * seg_progress,
            )

        current_dist += seg_dist

    return points[-1]


# ============================================================
# .OSU PARSER
# ============================================================
def parse_osu_file(filepath):
    """Parse only what the renderer needs."""
    hitobjects = []
    approach_rate = 9.0
    slider_multiplier = 1.4
    circle_size = 4.0
    overall_difficulty = 9.0
    section = None

    timing_uninherited = []
    timing_inherited = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue

        if section == "Difficulty":
            if line.startswith("ApproachRate:"):
                approach_rate = float(line.split(":", 1)[1])
            elif line.startswith("OverallDifficulty:"):
                # FIX: do not overwrite ApproachRate here.
                overall_difficulty = float(line.split(":", 1)[1])
            elif line.startswith("SliderMultiplier:"):
                slider_multiplier = float(line.split(":", 1)[1])
            elif line.startswith("CircleSize:"):
                circle_size = float(line.split(":", 1)[1])

        elif section == "TimingPoints":
            parts = line.split(",")
            if len(parts) < 2:
                continue

            try:
                timing = int(float(parts[0]))
                beat_length = float(parts[1])
            except ValueError:
                continue

            if beat_length > 0:
                timing_uninherited.append({"time": timing, "bl": beat_length})
            elif beat_length < 0:
                timing_inherited.append({"time": timing, "sv": -100.0 / beat_length})

        elif section == "HitObjects":
            parts = line.split(",")
            if len(parts) < 4:
                continue

            try:
                x = int(parts[0])
                y = int(parts[1])
                start_time = int(parts[2])
                obj_type = int(parts[3])
            except ValueError:
                continue

            # 0 = circle, 1 = slider, 2 = spinner
            if obj_type & 1:
                object_type = 0
            elif obj_type & 2:
                object_type = 1
            elif obj_type & 8:
                object_type = 2
            else:
                continue

            obj = {
                "time": start_time,
                "x": x,
                "y": y,
                "type": object_type,
                "duration": 0.0,
                "end_x": x,
                "end_y": y,
                "curve_points": [(x, y)],
                "repeats": 1,
            }

            # Slider
            if object_type == 1 and len(parts) >= 8:
                try:
                    raw_curve = parts[5].split("|")
                    curve_type = raw_curve[0]
                    curve_points = [(x, y)]

                    for point in raw_curve[1:]:
                        px, py = point.split(":")[:2]
                        curve_points.append((int(px), int(py)))

                    repeats = max(1, int(parts[6]))
                    length = float(parts[7])

                    flattened = flatten_curve(curve_type, curve_points)

                    # Some curve approximators (especially circular arcs) can
                    # return the same geometric path in the opposite order.
                    # The osu! slider direction must ALWAYS begin at the
                    # hitobject start position and finish at the slider end.
                    if len(flattened) >= 2:
                        start_pt = (x, y)
                        end_pt = curve_points[-1]
                        normal_error = (
                            math.hypot(flattened[0][0] - start_pt[0], flattened[0][1] - start_pt[1])
                            + math.hypot(flattened[-1][0] - end_pt[0], flattened[-1][1] - end_pt[1])
                        )
                        reversed_error = (
                            math.hypot(flattened[-1][0] - start_pt[0], flattened[-1][1] - start_pt[1])
                            + math.hypot(flattened[0][0] - end_pt[0], flattened[0][1] - end_pt[1])
                        )
                        if reversed_error < normal_error:
                            flattened.reverse()

                    obj["curve_points"] = flattened
                    obj["end_x"], obj["end_y"] = flattened[-1] if flattened else (x, y)
                    obj["repeats"] = repeats

                    # Find timing point active at slider start.
                    beat_length = 600.0
                    for tp in reversed(timing_uninherited):
                        if tp["time"] <= start_time:
                            beat_length = tp["bl"]
                            break

                    sv_multiplier = 1.0
                    for tp in reversed(timing_inherited):
                        if tp["time"] <= start_time:
                            sv_multiplier = tp["sv"]
                            break

                    beat_length = max(1e-6, beat_length)
                    sm = max(1e-6, slider_multiplier)
                    sv_multiplier = max(1e-6, sv_multiplier)

                    # Duration of one traversal, multiplied by repeats.
                    one_repeat_duration = (
                        beat_length * (length / (100.0 * sm * sv_multiplier))
                    )
                    obj["duration"] = max(0.0, one_repeat_duration * repeats)
                except (ValueError, IndexError, ZeroDivisionError):
                    obj["duration"] = 0.0

            # Spinner
            elif object_type == 2 and len(parts) >= 6:
                try:
                    obj["duration"] = max(0.0, int(parts[5]) - start_time)
                except ValueError:
                    obj["duration"] = 0.0

            hitobjects.append(obj)

    # osu! preempt from AR.
    if approach_rate < 5:
        ar_ms = 1200.0 + 600.0 * (5.0 - approach_rate) / 5.0
    elif approach_rate == 5:
        ar_ms = 1200.0
    else:
        ar_ms = 1200.0 - 750.0 * (approach_rate - 5.0) / 5.0

    # 50 judgment window (not critical for rendering, returned for env).
    if overall_difficulty <= 5:
        hit_window_50 = 200.0 - 10.0 * overall_difficulty
    else:
        hit_window_50 = 190.0 - 8.0 * overall_difficulty

    # Failsafe for malformed/old slider durations.
    for i, obj in enumerate(hitobjects):
        if obj["type"] != 0 and obj["duration"] <= 0:
            if i + 1 < len(hitobjects):
                obj["duration"] = max(1.0, hitobjects[i + 1]["time"] - obj["time"])
            else:
                obj["duration"] = 500.0

    return hitobjects, ar_ms, circle_size, hit_window_50


# ============================================================
# RENDERER
# ============================================================
def object_visible(obj, current_time, ar_ms):
    appear_time = obj["time"] - ar_ms
    end_time = obj["time"] + obj["duration"]
    disappear_time = end_time + HIT_FADE_MS
    return appear_time <= current_time <= disappear_time


def _hd_fade(current_time, start_time, ar_ms, floor):
    """Hidden: объект гаснет до момента удара. floor — остаточная яркость."""
    fade_len = max(1.0, ar_ms * HD_FADE_FRAC)
    fade_start = start_time - fade_len
    if current_time <= fade_start:
        return 1.0
    k = max(0.0, min(1.0, (current_time - fade_start) / fade_len))
    return 1.0 - (1.0 - floor) * k


def object_alpha(obj, current_time, ar_ms):
    start_time = obj["time"]
    end_time = start_time + obj["duration"]
    appear_time = start_time - ar_ms
    disappear_time = end_time + HIT_FADE_MS

    if current_time < appear_time or current_time > disappear_time:
        return 0.0

    if current_time < appear_time + FADE_IN_MS:
        alpha = max(0.0, (current_time - appear_time) / FADE_IN_MS)
    elif current_time <= end_time:
        alpha = 1.0
    else:
        alpha = max(0.0, 1.0 - (current_time - end_time) / HIT_FADE_MS)

    if ACTIVE_MODS & MOD_HD:
        if obj["type"] == 1:
            # Слайдер: гаснет до удара, тело тает по ходу, слабый след остаётся
            if current_time > start_time:
                t = (current_time - start_time) / max(1.0, obj["duration"])
                alpha *= max(0.15, 1.0 - t)
            else:
                alpha *= _hd_fade(current_time, start_time, ar_ms, 0.15)
        elif obj["type"] == 2:
            # Спиннер: оставляем след 0.3 — иначе с HD сеть его вообще не увидит
            alpha *= _hd_fade(current_time, start_time, ar_ms, 0.3)
        else:
            # Кружок: к моменту удара полностью невидим (настоящий HD)
            alpha *= _hd_fade(current_time, start_time, ar_ms, 0.0)

    return max(0.0, min(1.0, alpha))


def draw_approach_circle(canvas, center, hit_radius, start_time, current_time, ar_ms, value):
    delta = start_time - current_time
    if delta <= 0:
        return

    progress = max(0.0, min(1.0, delta / max(1.0, ar_ms)))

    # Large -> small as the hit time approaches.
    radius = int(hit_radius * (1.0 + 3.0 * progress))
    radius = max(hit_radius + 1, radius)

    cv2.circle(canvas, center, radius, int(value), 1, cv2.LINE_AA)


def draw_slider_body(canvas, obj, value, thickness):
    points = [to_canvas(*p) for p in obj["curve_points"]]
    if len(points) >= 2:
        np_points = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [np_points], False, int(value), thickness, cv2.LINE_AA)


def draw_reverse_marker(canvas, endpoint, travel_direction, value, size):
    """Draw a white, black-outlined triangle whose tip points along reverse travel."""
    dx, dy = travel_direction
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return
    dx /= length
    dy /= length
    px, py = -dy, dx

    gap = size * 0.40
    tip = (endpoint[0] + dx * gap, endpoint[1] + dy * gap)
    base_center = (endpoint[0] - dx * size * 1.25, endpoint[1] - dy * size * 1.25)
    half_width = size * 0.85

    outer = np.asarray([
        (int(round(tip[0])), int(round(tip[1]))),
        (int(round(base_center[0] + px * half_width)), int(round(base_center[1] + py * half_width))),
        (int(round(base_center[0] - px * half_width)), int(round(base_center[1] - py * half_width))),
    ], dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(canvas, [outer], 0, cv2.LINE_AA)

    inner_tip = (endpoint[0] + dx * gap * 0.95, endpoint[1] + dy * gap * 0.95)
    inner_base = (endpoint[0] - dx * size * 1.05, endpoint[1] - dy * size * 1.05)
    inner_half = size * 0.62
    inner = np.asarray([
        (int(round(inner_tip[0])), int(round(inner_tip[1]))),
        (int(round(inner_base[0] + px * inner_half)), int(round(inner_base[1] + py * inner_half))),
        (int(round(inner_base[0] - px * inner_half)), int(round(inner_base[1] - py * inner_half))),
    ], dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(canvas, [inner], int(value), cv2.LINE_AA)

def draw_cursor(canvas, x_osu, y_osu):
    """Белое ядро + чёрное гало: видно и на пустом поле, и на белом кружке.
    Вызывается ПОСЛЕ отрисовки объектов — гало обязано перекрывать их."""
    cx, cy = to_canvas(x_osu, y_osu)
    cv2.circle(canvas, (cx, cy), CURSOR_RADIUS + CURSOR_HALO, 0, -1)   # чёрное гало
    cv2.circle(canvas, (cx, cy), CURSOR_RADIUS, CURSOR_VALUE, -1)      # белое ядро

def draw_slider_reverse_markers(canvas, obj, value, size):
    """Draw reversal arrows at every actual slider repeat boundary."""
    repeats = max(1, int(obj.get('repeats', 1)))
    curve = [to_canvas(*p) for p in obj.get('curve_points', [])]
    if repeats <= 1 or len(curve) < 2:
        return

    start = curve[0]
    end = curve[-1]

    forward = (end[0] - start[0], end[1] - start[1])
    length = math.hypot(*forward)
    if length <= 1e-6:
        return
    forward = (forward[0] / length, forward[1] / length)
    backward = (-forward[0], -forward[1])

    marker_size = max(5.0, float(size) * 1.15)

    # Repeat 1 reverses at END: END -> START. Repeat 2 reverses at START:
    # START -> END. Thus the arrow points in the exact post-reversal direction.
    for reversal_index in range(1, repeats):
        if reversal_index % 2 == 1:
            draw_reverse_marker(canvas, end, backward, value, marker_size)
        else:
            draw_reverse_marker(canvas, start, forward, value, marker_size)


def render_hitobject(canvas, obj, current_time, ar_ms, cs_radius_px):
    """Minimal vision renderer: circles, approach rings, slider path/ball, reverses."""
    alpha = object_alpha(obj, current_time, ar_ms)
    if alpha <= 0.0:
        return

    obj_value = int(OBJ_VALUE * alpha)
    approach_value = int(APPROACH_VALUE * alpha)
    slider_value = int(SLIDER_BODY_VALUE * alpha)
    hd = bool(ACTIVE_MODS & MOD_HD)   # при HD approach-кругов не существует

    start_time = obj["time"]
    end_time = start_time + obj["duration"]
    radius = max(2, int(cs_radius_px))

    if obj["type"] == 0:  # Circle
        center = to_canvas(obj["x"], obj["y"])
        cv2.circle(canvas, center, radius, obj_value, -1, cv2.LINE_AA)

        if current_time < start_time and not hd:
            draw_approach_circle(
                canvas, center, radius, start_time, current_time, ar_ms, approach_value
            )

    elif obj["type"] == 1:  # Slider
        thickness = max(2, int(round(radius * 0.18)))
        draw_slider_body(canvas, obj, slider_value, thickness)
        start_center = to_canvas(obj["x"], obj["y"])

        # Before the hit: ordinary hitcircle + approach circle.
        if current_time < start_time:
            cv2.circle(canvas, start_center, radius, obj_value, -1, cv2.LINE_AA)
            if not hd:
                draw_approach_circle(
                    canvas, start_center, radius, start_time,
                    current_time, ar_ms, approach_value,
                )

        # After the hit, the white hitcircle travels along the slider.
        if start_time <= current_time <= end_time:
            repeats = max(1, int(obj.get("repeats", 1)))
            total_duration = max(1.0, float(obj["duration"]))
            repeat_duration = total_duration / repeats
            elapsed = max(0.0, min(total_duration, current_time - start_time))

            # Explicit per-repeat time: first traversal is ALWAYS start -> end.
            cycle_index = min(repeats - 1, int(elapsed // repeat_duration))
            cycle_elapsed = elapsed - cycle_index * repeat_duration
            local_progress = max(0.0, min(1.0, cycle_elapsed / repeat_duration))

            if cycle_index % 2 == 0:
                curve_progress = local_progress       # START -> END
            else:
                curve_progress = 1.0 - local_progress # END -> START

            moving_x, moving_y = get_point_on_curve(obj["curve_points"], curve_progress)
            moving_center = to_canvas(moving_x, moving_y)
            cv2.circle(canvas, moving_center, radius, obj_value, -1, cv2.LINE_AA)

        # Reverse markers fade together with the slider (важно для HD).
        if obj.get("repeats", 1) > 1:
            draw_slider_reverse_markers(canvas, obj, obj_value, max(4.0, radius * 0.28))

    elif obj["type"] == 2:  # Spinner
        center = to_canvas(256.0, 192.0)
        spinner_radius = max(4, int(45 * PLAY_W / OSU_W))
        cv2.circle(
            canvas,
            center,
            spinner_radius,
            int(SPINNER_VALUE * alpha),
            2,
            cv2.LINE_AA,
        )


# ============================================================
# MOD HELPERS
# ============================================================
def ar_value_from_ms(ms):
    if ms > 1200.0:
        return 5.0 - (ms - 1200.0) * 5.0 / 600.0
    if ms < 1200.0:
        return 5.0 + (1200.0 - ms) * 5.0 / 750.0
    return 5.0


def ar_ms_from_value(ar):
    if ar < 5.0:
        return 1200.0 + 600.0 * (5.0 - ar) / 5.0
    if ar > 5.0:
        return 1200.0 - 750.0 * (ar - 5.0) / 5.0
    return 1200.0


def apply_mods(hitobjects, ar_ms, cs, mods):
    """HR/EZ меняют AR, CS и (HR) положение объектов.
    DT/HT/NC не трогаем: их эффект приходит сам через живое время из tosu.
    МУТИРУЕТ список — передавай copy.deepcopy(...)!"""
    ar = ar_value_from_ms(ar_ms)

    if mods & MOD_HR:
        ar = min(10.0, ar * 1.4)
        cs = min(10.0, cs * 1.3)
        for o in hitobjects:
            o["y"] = OSU_H - o["y"]
            o["end_y"] = OSU_H - o["end_y"]
            o["curve_points"] = [(x, OSU_H - y) for (x, y) in o["curve_points"]]
    elif mods & MOD_EZ:
        ar = max(0.0, ar * 0.5)
        cs = max(0.0, cs * 0.5)

    return ar_ms_from_value(ar), cs


# ============================================================
# TOSU HELPERS
# ============================================================
def get_current_time(state):
    beatmap = state.get("beatmap", {})
    value = beatmap.get("time", {}).get("live", 0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def get_mod_number(state):
    play = state.get("play", {})
    mods = play.get("mods", {})

    if isinstance(mods, dict):
        try:
            return int(mods.get("number", 0))
        except (TypeError, ValueError):
            return 0

    return 0


# ============================================================
# MAIN (standalone viewer) — запускается ТОЛЬКО как `python vision.py`
# При import vision этот блок не выполняется.
# ============================================================
if __name__ == "__main__":
    window_name = "Neuro Vision"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    hwnd = user32.FindWindowW(None, window_name)
    if hwnd:
        user32.SetWindowPos(
            hwnd, 0xFFFF, 0, 0, 0, 0,
            0x0001 | 0x0002 | 0x0010 | 0x00040000,
        )

    # Сырой разбор (один раз на карту) + текущая сборка под конкретные моды.
    raw_md5 = None
    raw_objects, raw_ar_ms, raw_cs, raw_od = [], 1200.0, 4.0, 9.0
    map_key = None
    hitobjects, object_times = [], []
    ar_ms, cs_radius_px = 1200.0, 18.0

    print("Ожидание tosu...")

    while True:
        try:
            response = requests.get(TOSU_URL, timeout=2)
            response.raise_for_status()
            state = response.json()

            beatmap = state.get("beatmap", {})
            md5 = beatmap.get("checksum", "")
            current_time = get_current_time(state)
            mods = get_mod_number(state)

            # Сырой парс — один раз на новую карту.
            if md5 and md5 != raw_md5:
                base_path = state.get("folders", {}).get("songs", "")
                rel_path = state.get("files", {}).get("beatmap", "")
                filepath = os.path.join(base_path, rel_path) if base_path and rel_path else None

                if filepath and os.path.exists(filepath):
                    raw_objects, raw_ar_ms, raw_cs, raw_od = parse_osu_file(filepath)
                    raw_md5 = md5
                    map_key = None  # форсируем пересборку
                else:
                    print("Файл .osu не найден по пути из tosu.      ", end="\r")

            # Пересборка при смене карты ИЛИ МОДОВ (md5 от модов не меняется!).
            if raw_md5 and map_key != (md5, mods):
                objs = copy.deepcopy(raw_objects)
                ar_ms, cs = apply_mods(objs, raw_ar_ms, raw_cs, mods)
                ACTIVE_MODS = mods  # включает HD в рендере

                hitobjects = objs
                object_times = [o["time"] for o in objs]
                cs_radius_px = max(2.0, (54.4 - 4.48 * cs) * PLAY_W / OSU_W)
                map_key = (md5, mods)

                names = [n for b, n in (
                    (MOD_EZ, "EZ"), (MOD_HD, "HD"), (MOD_HR, "HR"),
                    (MOD_DT, "DT"), (MOD_HT, "HT"), (MOD_NC, "NC"),
                ) if mods & b]
                print(
                    f"\nКарта: {len(objs)} об. | AR={ar_ms:.0f} ms | "
                    f"CS={cs:.2f} | OD={raw_od:.2f} | "
                    f"моды: {'+'.join(names) if names else 'nomod'}"
                )

            if hitobjects:
                # Pure grayscale observation. No osu! UI and no cursor.
                canvas = np.zeros((IMG_H, IMG_W), dtype=np.uint8)

                visible_from = current_time - PAST_MS
                visible_to = current_time + ar_ms

                left = max(0, bisect_left(object_times, visible_from) - 1)
                right = min(len(hitobjects), bisect_right(object_times, visible_to) + 1)

                for obj in hitobjects[left:right]:
                    if not object_visible(obj, current_time, ar_ms):
                        continue
                    try:
                        render_hitobject(
                            canvas,
                            obj,
                            current_time,
                            ar_ms,
                            cs_radius_px,
                        )
                    except Exception as render_error:
                        print(
                            f"Render error for object at {obj.get('time', '?')}: "
                            f"{render_error}"
                        )

                cv2.imshow(window_name, canvas)

                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            print(
                f"Time: {current_time:8.1f} ms | "
                f"Objects: {len(hitobjects):5d}   ",
                end="\r",
            )

        except requests.exceptions.RequestException:
            print("Нет связи с tosu. Запущен ли tosu.exe?   ", end="\r")
            time.sleep(0.5)
        except ValueError as error:
            print(f"Ошибка JSON: {error}                      ", end="\r")
            time.sleep(0.2)
        except Exception as error:
            print(f"Ошибка: {error}                      ", end="\r")
            time.sleep(0.2)

    cv2.destroyAllWindows()