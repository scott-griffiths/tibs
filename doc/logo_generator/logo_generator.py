from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(".")
OUT = ROOT / "outputs"
ANIMATED_PREVIEW = "tibs-transition-preview.png"
DOC_LOGO = Path(__file__).resolve().parents[1] / "tibs.png"

# Edit this block, then run this script with no command-line arguments.
CONFIG = {
    "width": 550,
    "height": 235,
    "frame_count": 28,
    "trace_frame_count": 28,
    "start_hold_ms": 600,
    "trace_duration_ms": 1000,
    "duration_ms": 500,
    "initial_delay_ms": 0,
    "final_hold_ms": 3600000,
    "stroke_width": 40,
    "corner_radius": 8,
    "box_size": 112,
    "box_gap": 53,
    "origin_x": 63,
    "origin_y": 86,
    "ascender_height": 52,
    "t_ascender_height": 35,
    "t_bar_left": 28,
    "t_bar_right": 35,
    "t_foot_right": 56,
    "i_top_left": 24,
    "color": "#1e86de",
    "background": "transparent",
    "border_size": 0,
    "border_color": "#0f5f9a",
}

DEFAULTS = CONFIG


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def is_transparent(value: object) -> bool:
    return str(value).strip().lower() in {"", "none", "transparent"}


def border_size(config: dict = DEFAULTS) -> float:
    return max(0.0, float(config.get("border_size", 0)))


def border_enabled(config: dict = DEFAULTS) -> bool:
    return border_size(config) > 0 and not is_transparent(config.get("border_color", ""))


def start_hold_millis(config: dict = DEFAULTS) -> float:
    return float(config.get("start_hold_ms", 0))


def trace_duration_millis(config: dict = DEFAULTS) -> float:
    return float(config.get("trace_duration_ms", 0))


def pause_duration_millis(config: dict = DEFAULTS) -> float:
    return float(config.get("initial_delay_ms", 0))


def transition_duration_millis(config: dict = DEFAULTS) -> float:
    return float(config["duration_ms"])


def total_duration_millis(config: dict = DEFAULTS) -> float:
    return (
        start_hold_millis(config)
        + trace_duration_millis(config)
        + pause_duration_millis(config)
        + transition_duration_millis(config)
    )


def trace_frame_count(config: dict = DEFAULTS) -> int:
    return max(2, int(config.get("trace_frame_count", config["frame_count"])))


def transition_frame_count(config: dict = DEFAULTS) -> int:
    return max(2, int(config["frame_count"]))


def total_frame_count(config: dict = DEFAULTS) -> int:
    return trace_frame_count(config) + transition_frame_count(config)


def frame_times_millis(config: dict = DEFAULTS) -> list[float]:
    trace_count = trace_frame_count(config)
    transition_count = transition_frame_count(config)
    start_hold = start_hold_millis(config)
    trace_duration = trace_duration_millis(config)
    pause_duration = pause_duration_millis(config)
    transition_duration = transition_duration_millis(config)

    trace_times = [0.0] + [
        start_hold + i * trace_duration / (trace_count - 1) for i in range(1, trace_count)
    ]
    transition_start = start_hold + trace_duration + pause_duration
    if pause_duration <= 0:
        transition_times = [
            transition_start + (i + 1) * transition_duration / transition_count for i in range(transition_count)
        ]
    else:
        transition_times = [
            transition_start + i * transition_duration / (transition_count - 1) for i in range(transition_count)
        ]
    return trace_times + transition_times


def frame_raw_t(index: int, config: dict = DEFAULTS) -> float:
    total = total_duration_millis(config)
    if total <= 0:
        return 1.0
    return clamp(frame_times_millis(config)[index] / total)


def timeline_millis(raw_t: float, config: dict = DEFAULTS) -> int:
    if raw_t <= 0:
        return 0
    return round(raw_t * total_duration_millis(config))


def ease_out_cubic(t: float) -> float:
    return 1 - (1 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    t = clamp(t)
    if t < 0.5:
        return 4 * t * t * t
    return 1 - ((-2 * t + 2) ** 3) / 2


def ease_in_out_sine(t: float) -> float:
    return -(math.cos(math.pi * clamp(t)) - 1) / 2


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 1.0 if x >= edge1 else 0.0
    x = clamp((x - edge0) / (edge1 - edge0))
    return x * x * (3 - 2 * x)


def mix(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def mix_point(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
    return (mix(a[0], b[0], t), mix(a[1], b[1], t))


def lerp_points(
    start: list[tuple[float, float]],
    end: list[tuple[float, float]],
    t: float,
) -> list[tuple[float, float]]:
    return [mix_point(a, b, t) for a, b in zip(start, end)]


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_toward(
    src: tuple[float, float],
    dst: tuple[float, float],
    amount: float,
) -> tuple[float, float]:
    d = dist(src, dst)
    if d == 0:
        return src
    return (src[0] + (dst[0] - src[0]) * amount / d, src[1] + (dst[1] - src[1]) * amount / d)


def rounded_path(points: list[tuple[float, float]], radius: float) -> str:
    if len(points) < 2:
        return ""

    def fmt(p: tuple[float, float]) -> str:
        return f"{p[0]:.2f},{p[1]:.2f}"

    if len(points) == 2 or radius <= 0:
        return "M " + " L ".join(fmt(p) for p in points)

    pieces = [f"M {fmt(points[0])}"]
    for i in range(1, len(points) - 1):
        prev_pt = points[i - 1]
        pt = points[i]
        next_pt = points[i + 1]
        before = dist(prev_pt, pt)
        after = dist(pt, next_pt)
        r = min(radius, before * 0.42, after * 0.42)
        if r <= 0.01:
            pieces.append(f"L {fmt(pt)}")
            continue
        p1 = point_toward(pt, prev_pt, r)
        p2 = point_toward(pt, next_pt, r)
        pieces.append(f"L {fmt(p1)}")
        pieces.append(f"Q {fmt(pt)} {fmt(p2)}")
    pieces.append(f"L {fmt(points[-1])}")
    return " ".join(pieces)


def path_spec(
    name: str,
    start: list[tuple[float, float]],
    end: list[tuple[float, float]],
    progress: float,
    opacity: float = 1.0,
    roundable: bool = True,
    extension: float | None = None,
) -> dict:
    return {
        "name": name,
        "points": lerp_points(start, end, progress),
        "opacity": opacity,
        "roundable": roundable,
        "extension": extension,
    }


def static_spec(
    name: str,
    points: list[tuple[float, float]],
    roundable: bool = True,
    extension: float | None = None,
) -> dict:
    return {
        "name": name,
        "points": points,
        "opacity": 1.0,
        "roundable": roundable,
        "extension": extension,
    }


def rect_spec(
    name: str,
    cx: float,
    cy: float,
    size: float,
    radius: float,
    opacity: float = 1.0,
) -> dict:
    return {
        "name": name,
        "x": cx - size / 2,
        "y": cy - size / 2,
        "width": size,
        "height": size,
        "radius": min(radius, size / 2),
        "opacity": opacity,
    }


def rotate_endpoint(
    pivot: tuple[float, float],
    length: float,
    start_angle: float,
    end_angle: float,
    progress: float,
) -> tuple[float, float]:
    angle = mix(start_angle, end_angle, progress)
    return (pivot[0] + math.cos(angle) * length, pivot[1] + math.sin(angle) * length)


def trace_square_specs(
    name: str,
    x0: float,
    y_top: float,
    box: float,
    progress: float,
    config: dict = DEFAULTS,
    reverse: bool = False,
) -> tuple[list[dict], list[dict]]:
    y_bot = y_top + box
    x1 = x0 + box
    block = float(config["stroke_width"])
    offset = max(0.0, block - border_size(config) * 2)
    if reverse:
        points = [
            (x0 + offset, y_top),
            (x1, y_top),
            (x1, y_bot),
            (x0, y_bot),
            (x0, y_top),
        ]
    else:
        points = [
            (x1 - offset, y_top),
            (x0, y_top),
            (x0, y_bot),
            (x1, y_bot),
            (x1, y_top),
        ]
    total_travel = sum(dist(start, end) for start, end in zip(points, points[1:]))
    d = clamp(progress) * total_travel
    paths: list[dict] = []

    def add_segment(segment_name: str, start: tuple[float, float], end: tuple[float, float]) -> None:
        if dist(start, end) > 0.01:
            paths.append(static_spec(f"{name}-{segment_name}", [start, end], roundable=False))

    head = points[0]
    remaining = d
    for index, (start, end) in enumerate(zip(points, points[1:])):
        segment_length = dist(start, end)
        if remaining <= 0:
            break
        head = end if remaining >= segment_length else point_toward(start, end, remaining)
        add_segment(f"trace-{index}", start, head)
        remaining -= segment_length

    rects = [
        rect_spec(
            f"{name}-top-tracer",
            head[0],
            head[1],
            block,
            float(config["corner_radius"]),
        )
    ]
    return paths, rects


def trace_shapes(progress: float, config: dict = DEFAULTS) -> dict:
    progress = clamp(progress)
    if progress >= 0.999:
        return transition_shapes(0.0, config)

    box = float(config["box_size"])
    gap = float(config["box_gap"])
    y_top = float(config["origin_y"])
    lx0 = float(config["origin_x"])
    bx0 = lx0 + box + gap
    sx0 = bx0 + box + gap

    paths: list[dict] = []
    rects: list[dict] = []
    for name, x0, reverse in [("left-box", lx0, False), ("middle-box", bx0, True), ("right-box", sx0, False)]:
        box_paths, box_rects = trace_square_specs(name, x0, y_top, box, progress, config, reverse)
        paths.extend(box_paths)
        rects.extend(box_rects)

    return {
        "paths": paths,
        "rects": rects,
        "dot": {"name": "i-dot", "cx": lx0 + box, "cy": y_top, "r": 0, "opacity": 0},
    }


def transition_shapes(raw_t: float, config: dict = DEFAULTS) -> dict:
    p = ease_out_cubic(raw_t)

    box = float(config["box_size"])
    gap = float(config["box_gap"])
    y_top = float(config["origin_y"])
    y_bot = y_top + box
    y_mid = (y_top + y_bot) / 2
    lx0 = float(config["origin_x"])
    lx1 = lx0 + box
    bx0 = lx1 + gap
    bx1 = bx0 + box
    sx0 = bx1 + gap
    sx1 = sx0 + box
    asc = float(config["ascender_height"])
    t_asc = float(config.get("t_ascender_height", config["ascender_height"]))
    i_top_left = float(config["i_top_left"])
    s_half = (y_bot - y_top) / 2
    s_fold_len = mix(s_half, (sx1 - sx0) / 2, p)
    s_middle_overlap = float(config["corner_radius"]) * smoothstep(0.55, 1.0, raw_t)

    paths = [
        # First box: its four sides become a three-stroke t plus the i stem.
        path_spec("t-stem", [(lx0, y_top), (lx0, y_bot)], [(lx0, y_top - t_asc), (lx0, y_bot)], p, roundable=False),
        path_spec(
            "t-bar",
            [(lx0, y_top), (lx1 - i_top_left, y_top)],
            [(lx0 - float(config["t_bar_left"]), y_top), (lx0 + float(config["t_bar_right"]), y_top)],
            p,
            roundable=False,
        ),
        static_spec("i-top", [(lx1 - i_top_left, y_top), (lx1, y_top)], roundable=False),
        path_spec(
            "t-foot",
            [(lx0, y_bot), (lx1, y_bot)],
            [(lx0, y_bot), (lx0 + float(config["t_foot_right"]), y_bot)],
            p,
            roundable=False,
        ),
        path_spec("i-stem", [(lx1, y_top), (lx1, y_bot)], [(lx1, y_top), (lx1, y_bot)], p, roundable=False),
        # Second box: stays in place as the b loop, with the left side extending up.
        path_spec("b-stem", [(bx0, y_top), (bx0, y_bot)], [(bx0, y_top - asc), (bx0, y_bot)], p, roundable=False),
        static_spec("b-top", [(bx0, y_top), (bx1, y_top)], roundable=False),
        static_spec("b-right", [(bx1, y_top), (bx1, y_bot)], roundable=False),
        static_spec("b-bottom", [(bx1, y_bot), (bx0, y_bot)], roundable=False),
        # Third box: top and bottom stay put; vertical halves fold into the s.
        static_spec("s-top", [(sx0, y_top), (sx1, y_top)], roundable=False),
        static_spec("s-bottom", [(sx0, y_bot), (sx1, y_bot)], roundable=False),
        static_spec("s-left-upper", [(sx0, y_top), (sx0, y_mid)], roundable=False),
        static_spec("s-right-lower", [(sx1, y_mid), (sx1, y_bot)], roundable=False),
        static_spec(
            "s-left-fold",
            [(sx0, y_mid), rotate_endpoint((sx0, y_mid), s_fold_len, math.pi / 2, 0, p)],
            roundable=False,
            extension=(0, s_middle_overlap),
        ),
        static_spec(
            "s-right-fold",
            [(sx1, y_mid), rotate_endpoint((sx1, y_mid), s_fold_len, -math.pi / 2, -math.pi, p)],
            roundable=False,
            extension=(0, s_middle_overlap),
        ),
    ]
    return {
        "paths": paths,
        "rects": [],
        "dot": {
            "name": "i-dot",
            "cx": lx1,
            "cy": mix(y_top, y_top - asc, p),
            "r": config["stroke_width"] * 0.50,
            "opacity": 1.0,
        },
    }


def frame_shapes(raw_t: float, config: dict = DEFAULTS) -> dict:
    elapsed = clamp(raw_t) * total_duration_millis(config)
    start_hold = start_hold_millis(config)
    trace_duration = trace_duration_millis(config)
    pause_duration = pause_duration_millis(config)

    if elapsed < start_hold:
        return trace_shapes(0.0, config)

    elapsed -= start_hold
    if trace_duration > 0 and elapsed < trace_duration:
        return trace_shapes(elapsed / trace_duration, config)
    if elapsed < trace_duration + pause_duration:
        return transition_shapes(0.0, config)

    transition_duration = transition_duration_millis(config)
    if transition_duration <= 0:
        return transition_shapes(1.0, config)
    return transition_shapes((elapsed - trace_duration - pause_duration) / transition_duration, config)


def svg_frame(raw_t: float, config: dict = DEFAULTS, include_background: bool = True) -> str:
    width = config["width"]
    height = config["height"]
    stroke = config["stroke_width"]
    radius = config["corner_radius"]
    color = config["color"]
    bg = config["background"]
    shapes = frame_shapes(raw_t, config)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
    ]
    if include_background and not is_transparent(bg):
        parts.append(f'<rect id="background" width="{width}" height="{height}" fill="{bg}"/>')
    if border_enabled(config):
        parts.append(svg_shapes_group(shapes, stroke, radius, str(config["border_color"]), 0, "tibs-border"))
    parts.append(svg_shapes_group(shapes, stroke, radius, color, border_size(config) if border_enabled(config) else 0, "tibs"))
    parts.append("</svg>")
    return "".join(parts)


def svg_shapes_group(
    shapes: dict,
    stroke: float,
    radius: float,
    fill: str,
    inset: float,
    group_id: str,
) -> str:
    parts = [f'<g id="{group_id}" fill="{fill}">']
    render_stroke = max(0.0, stroke - inset * 2)
    render_radius = max(0.0, radius - inset)
    for item in shapes["paths"]:
        op = item["opacity"]
        if op <= 0.001 or render_stroke <= 0.001:
            continue
        parts.append(
            svg_segment_rect(
                item["name"],
                item["points"],
                render_stroke,
                render_radius,
                op,
                inset_extension(item.get("extension"), inset),
            )
        )
    for item in shapes.get("rects", []):
        op = item["opacity"]
        if op <= 0.001:
            continue
        x = item["x"] + inset
        y = item["y"] + inset
        width = max(0.0, item["width"] - inset * 2)
        height = max(0.0, item["height"] - inset * 2)
        if width <= 0.001 or height <= 0.001:
            continue
        item_radius = max(0.0, item["radius"] - inset)
        parts.append(
            f'<rect id="{item["name"]}" x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" rx="{item_radius:.2f}" opacity="{op:.3f}"/>'
        )
    dot = shapes["dot"]
    if dot["opacity"] > 0.001 and dot["r"] > 0.001:
        size = max(0.0, dot["r"] * 2 - inset * 2)
        if size <= 0.001:
            parts.append("</g>")
            return "".join(parts)
        dot_radius = min(render_radius, size / 2)
        parts.append(
            f'<rect id="{dot["name"]}" x="{dot["cx"] - size / 2:.2f}" y="{dot["cy"] - size / 2:.2f}" width="{size:.2f}" height="{size:.2f}" rx="{dot_radius:.2f}" opacity="{dot["opacity"]:.3f}"/>'
        )
    parts.append("</g>")
    return "".join(parts)


def inset_extension(extension: object, inset: float) -> float | tuple[float, float] | None:
    if extension is None:
        return None
    if isinstance(extension, (list, tuple)):
        return (max(0.0, float(extension[0]) - inset), max(0.0, float(extension[1]) - inset))
    return max(0.0, float(extension) - inset)


def svg_segment_rect(
    name: str,
    points: list[tuple[float, float]],
    stroke: float,
    radius: float,
    opacity: float,
    extension: float | None = None,
) -> str:
    if len(points) < 2:
        return ""
    x1, y1 = points[0]
    x2, y2 = points[-1]
    length = dist((x1, y1), (x2, y2))
    if length <= 0.01:
        return ""
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    start_ext, end_ext = segment_extensions(extension, stroke / 2)
    visual_length = length + start_ext + end_ext
    r = min(radius, stroke / 2, visual_length / 2)
    return (
        f'<rect id="{name}" x="{-start_ext:.2f}" y="{-stroke / 2:.2f}" width="{visual_length:.2f}" height="{stroke:.2f}" '
        f'rx="{r:.2f}" transform="translate({x1:.2f},{y1:.2f}) rotate({angle:.3f})" opacity="{opacity:.3f}"/>'
    )


def segment_extensions(extension: object, default: float) -> tuple[float, float]:
    if extension is None:
        return default, default
    if isinstance(extension, (list, tuple)):
        return float(extension[0]), float(extension[1])
    value = float(extension)
    return value, value


PREVIEW_GENERATOR_JS = r"""
const DEFAULTS = {
  width: 800,
  height: 360,
  frameCount: 28,
  traceFrameCount: 28,
  startHoldMs: 400,
  traceDurationMs: 650,
  durationMs: 1100,
  initialDelayMs: 0,
  strokeWidth: 30,
  cornerRadius: 4,
  boxSize: 112,
  boxGap: 38,
  originX: 155,
  originY: 128,
  ascenderHeight: 42,
  tAscenderHeight: 42,
  tBarLeft: 30,
  tBarRight: 50,
  tFootRight: 70,
  iTopLeft: 24,
  color: '#281DF6',
  background: '#000000',
  borderSize: 3,
  borderColor: '#0f5f9a'
};

function clamp(value, lo = 0, hi = 1) {
  return Math.max(lo, Math.min(hi, value));
}

function isTransparent(value) {
  return ['', 'none', 'transparent'].includes(String(value || '').trim().toLowerCase());
}

function numberOrDefault(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function borderSize(config) {
  return Math.max(0, Number(config.borderSize || 0));
}

function borderEnabled(config) {
  return borderSize(config) > 0 && !isTransparent(config.borderColor);
}

function startHoldMillis(config) {
  return Number(config.startHoldMs || 0);
}

function traceDurationMillis(config) {
  return Number(config.traceDurationMs || 0);
}

function pauseDurationMillis(config) {
  return Number(config.initialDelayMs || 0);
}

function transitionDurationMillis(config) {
  return Number(config.durationMs || 0);
}

function totalDurationMillis(config) {
  return startHoldMillis(config) + traceDurationMillis(config) + pauseDurationMillis(config) + transitionDurationMillis(config);
}

function traceFrameCount(config) {
  return Math.max(2, Math.round(Number(config.traceFrameCount) || Number(config.frameCount) || 2));
}

function transitionFrameCount(config) {
  return Math.max(2, Math.round(Number(config.frameCount) || 2));
}

function totalFrameCount(config) {
  return traceFrameCount(config) + transitionFrameCount(config);
}

function frameTimesMillis(config) {
  const traceCount = traceFrameCount(config);
  const transitionCount = transitionFrameCount(config);
  const startHold = startHoldMillis(config);
  const traceDuration = traceDurationMillis(config);
  const pauseDuration = pauseDurationMillis(config);
  const transitionDuration = transitionDurationMillis(config);
  const times = [0];
  for (let i = 1; i < traceCount; i++) {
    times.push(startHold + (i * traceDuration) / (traceCount - 1));
  }
  const transitionStart = startHold + traceDuration + pauseDuration;
  if (pauseDuration <= 0) {
    for (let i = 0; i < transitionCount; i++) {
      times.push(transitionStart + ((i + 1) * transitionDuration) / transitionCount);
    }
  } else {
    for (let i = 0; i < transitionCount; i++) {
      times.push(transitionStart + (i * transitionDuration) / (transitionCount - 1));
    }
  }
  return times;
}

function frameRawT(index, config) {
  const total = totalDurationMillis(config);
  if (total <= 0) return 1;
  return clamp(frameTimesMillis(config)[index] / total);
}

function timelineMillis(rawT, config) {
  if (rawT <= 0) return 0;
  return Math.round(rawT * totalDurationMillis(config));
}

function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

function easeInOutCubic(t) {
  t = clamp(t);
  if (t < 0.5) return 4 * t * t * t;
  return 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function easeInOutSine(t) {
  return -(Math.cos(Math.PI * clamp(t)) - 1) / 2;
}

function smoothstep(edge0, edge1, x) {
  if (edge0 === edge1) return x >= edge1 ? 1 : 0;
  x = clamp((x - edge0) / (edge1 - edge0));
  return x * x * (3 - 2 * x);
}

function mix(a, b, t) {
  return a + (b - a) * t;
}

function mixPoint(a, b, t) {
  return [mix(a[0], b[0], t), mix(a[1], b[1], t)];
}

function lerpPoints(start, end, t) {
  return start.map((point, index) => mixPoint(point, end[index], t));
}

function dist(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function pointToward(src, dst, amount) {
  const d = dist(src, dst);
  if (d === 0) return src;
  return [
    src[0] + ((dst[0] - src[0]) * amount) / d,
    src[1] + ((dst[1] - src[1]) * amount) / d
  ];
}

function fmt(point) {
  return `${point[0].toFixed(2)},${point[1].toFixed(2)}`;
}

function roundedPath(points, radius) {
  if (points.length < 2) return '';
  if (points.length === 2 || radius <= 0) return `M ${points.map(fmt).join(' L ')}`;

  const pieces = [`M ${fmt(points[0])}`];
  for (let i = 1; i < points.length - 1; i++) {
    const prev = points[i - 1];
    const pt = points[i];
    const next = points[i + 1];
    const before = dist(prev, pt);
    const after = dist(pt, next);
    const r = Math.min(radius, before * 0.42, after * 0.42);
    if (r <= 0.01) {
      pieces.push(`L ${fmt(pt)}`);
      continue;
    }
    const p1 = pointToward(pt, prev, r);
    const p2 = pointToward(pt, next, r);
    pieces.push(`L ${fmt(p1)}`);
    pieces.push(`Q ${fmt(pt)} ${fmt(p2)}`);
  }
  pieces.push(`L ${fmt(points[points.length - 1])}`);
  return pieces.join(' ');
}

function pathSpec(name, start, end, progress, opacity = 1, roundable = true, extension = null) {
  return {
    name,
    points: lerpPoints(start, end, progress),
    opacity,
    roundable,
    extension
  };
}

function staticSpec(name, points, roundable = true, extension = null) {
  return {
    name,
    points,
    opacity: 1,
    roundable,
    extension
  };
}

function rectSpec(name, cx, cy, size, radius, opacity = 1) {
  return {
    name,
    x: cx - size / 2,
    y: cy - size / 2,
    width: size,
    height: size,
    radius: Math.min(radius, size / 2),
    opacity
  };
}

function rotateEndpoint(pivot, length, startAngle, endAngle, progress) {
  const angle = mix(startAngle, endAngle, progress);
  return [
    pivot[0] + Math.cos(angle) * length,
    pivot[1] + Math.sin(angle) * length
  ];
}

function transitionShapes(rawT, config) {
  const p = easeOutCubic(rawT);
  const box = Number(config.boxSize);
  const gap = Number(config.boxGap);
  const yTop = Number(config.originY);
  const yBot = yTop + box;
  const yMid = (yTop + yBot) / 2;
  const lx0 = Number(config.originX);
  const lx1 = lx0 + box;
  const bx0 = lx1 + gap;
  const bx1 = bx0 + box;
  const sx0 = bx1 + gap;
  const sx1 = sx0 + box;
  const asc = Number(config.ascenderHeight);
  const tAsc = Number(config.tAscenderHeight ?? config.ascenderHeight);
  const iTopLeft = Number(config.iTopLeft);
  const sHalf = (yBot - yTop) / 2;
  const sFoldLen = mix(sHalf, (sx1 - sx0) / 2, p);
  const sMiddleOverlap = Number(config.cornerRadius) * smoothstep(0.55, 1, rawT);
  const paths = [
    pathSpec('t-stem', [[lx0, yTop], [lx0, yBot]], [[lx0, yTop - tAsc], [lx0, yBot]], p, 1, false),
    pathSpec('t-bar', [[lx0, yTop], [lx1 - iTopLeft, yTop]], [[lx0 - Number(config.tBarLeft), yTop], [lx0 + Number(config.tBarRight), yTop]], p, 1, false),
    staticSpec('i-top', [[lx1 - iTopLeft, yTop], [lx1, yTop]], false),
    pathSpec('t-foot', [[lx0, yBot], [lx1, yBot]], [[lx0, yBot], [lx0 + Number(config.tFootRight), yBot]], p, 1, false),
    pathSpec('i-stem', [[lx1, yTop], [lx1, yBot]], [[lx1, yTop], [lx1, yBot]], p, 1, false),
    pathSpec('b-stem', [[bx0, yTop], [bx0, yBot]], [[bx0, yTop - asc], [bx0, yBot]], p, 1, false),
    staticSpec('b-top', [[bx0, yTop], [bx1, yTop]], false),
    staticSpec('b-right', [[bx1, yTop], [bx1, yBot]], false),
    staticSpec('b-bottom', [[bx1, yBot], [bx0, yBot]], false),
    staticSpec('s-top', [[sx0, yTop], [sx1, yTop]], false),
    staticSpec('s-bottom', [[sx0, yBot], [sx1, yBot]], false),
    staticSpec('s-left-upper', [[sx0, yTop], [sx0, yMid]], false),
    staticSpec('s-right-lower', [[sx1, yMid], [sx1, yBot]], false),
    staticSpec('s-left-fold', [[sx0, yMid], rotateEndpoint([sx0, yMid], sFoldLen, Math.PI / 2, 0, p)], false, [0, sMiddleOverlap]),
    staticSpec('s-right-fold', [[sx1, yMid], rotateEndpoint([sx1, yMid], sFoldLen, -Math.PI / 2, -Math.PI, p)], false, [0, sMiddleOverlap])
  ];
  return {
    paths,
    rects: [],
    dot: {
      name: 'i-dot',
      cx: lx1,
      cy: mix(yTop, yTop - asc, p),
      r: config.strokeWidth * 0.50,
      opacity: 1
    }
  };
}

function traceSquareSpecs(name, x0, yTop, box, progress, config, reverse = false) {
  const yBot = yTop + box;
  const x1 = x0 + box;
  const block = Number(config.strokeWidth);
  const offset = Math.max(0, block - borderSize(config) * 2);
  const points = reverse
    ? [[x0 + offset, yTop], [x1, yTop], [x1, yBot], [x0, yBot], [x0, yTop]]
    : [[x1 - offset, yTop], [x0, yTop], [x0, yBot], [x1, yBot], [x1, yTop]];
  const totalTravel = points.slice(0, -1).reduce((total, point, index) => total + dist(point, points[index + 1]), 0);
  let remaining = clamp(progress) * totalTravel;
  let head = points[0];
  const paths = [];

  function addSegment(segmentName, start, end) {
    if (dist(start, end) > 0.01) {
      paths.push(staticSpec(`${name}-${segmentName}`, [start, end], false));
    }
  }

  for (let index = 0; index < points.length - 1; index++) {
    const start = points[index];
    const end = points[index + 1];
    const segmentLength = dist(start, end);
    if (remaining <= 0) break;
    head = remaining >= segmentLength ? end : pointToward(start, end, remaining);
    addSegment(`trace-${index}`, start, head);
    remaining -= segmentLength;
  }

  return {
    paths,
    rects: [
      rectSpec(`${name}-top-tracer`, head[0], head[1], block, Number(config.cornerRadius))
    ]
  };
}

function traceShapes(progress, config) {
  progress = clamp(progress);
  if (progress >= 0.999) return transitionShapes(0, config);

  const box = Number(config.boxSize);
  const gap = Number(config.boxGap);
  const yTop = Number(config.originY);
  const lx0 = Number(config.originX);
  const bx0 = lx0 + box + gap;
  const sx0 = bx0 + box + gap;
  const paths = [];
  const rects = [];
  for (const [name, x0, reverse] of [['left-box', lx0, false], ['middle-box', bx0, true], ['right-box', sx0, false]]) {
    const specs = traceSquareSpecs(name, x0, yTop, box, progress, config, reverse);
    paths.push(...specs.paths);
    rects.push(...specs.rects);
  }
  return {
    paths,
    rects,
    dot: { name: 'i-dot', cx: lx0 + box, cy: yTop, r: 0, opacity: 0 }
  };
}

function frameShapes(rawT, config) {
  const elapsed = clamp(rawT) * totalDurationMillis(config);
  const startHold = startHoldMillis(config);
  const traceDuration = traceDurationMillis(config);
  const pauseDuration = pauseDurationMillis(config);
  if (elapsed < startHold) {
    return traceShapes(0, config);
  }
  const activeElapsed = elapsed - startHold;
  if (traceDuration > 0 && activeElapsed < traceDuration) {
    return traceShapes(activeElapsed / traceDuration, config);
  }
  if (activeElapsed < traceDuration + pauseDuration) {
    return transitionShapes(0, config);
  }
  const transitionDuration = transitionDurationMillis(config);
  if (transitionDuration <= 0) return transitionShapes(1, config);
  return transitionShapes((activeElapsed - traceDuration - pauseDuration) / transitionDuration, config);
}

function svgFrame(rawT, config, includeBackground = true) {
  const shapes = frameShapes(rawT, config);
  const parts = [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${config.width} ${config.height}" width="${config.width}" height="${config.height}">`
  ];
  if (includeBackground && !isTransparent(config.background)) {
    parts.push(`<rect id="background" width="${config.width}" height="${config.height}" fill="${config.background}"/>`);
  }
  if (borderEnabled(config)) {
    parts.push(shapesGroupSvg(shapes, config.strokeWidth, config.cornerRadius, config.borderColor, 0, 'tibs-border'));
  }
  parts.push(shapesGroupSvg(shapes, config.strokeWidth, config.cornerRadius, config.color, borderEnabled(config) ? borderSize(config) : 0, 'tibs'));
  parts.push('</svg>');
  return parts.join('');
}

function shapesGroupSvg(shapes, stroke, radius, fill, inset, groupId) {
  const parts = [`<g id="${groupId}" fill="${fill}">`];
  const renderStroke = Math.max(0, Number(stroke) - inset * 2);
  const renderRadius = Math.max(0, Number(radius) - inset);
  for (const item of shapes.paths) {
    if (item.opacity <= 0.001 || renderStroke <= 0.001) continue;
    parts.push(segmentRectSvg(item.name, item.points, renderStroke, renderRadius, item.opacity, insetExtension(item.extension, inset)));
  }
  for (const item of shapes.rects || []) {
    if (item.opacity <= 0.001) continue;
    const x = item.x + inset;
    const y = item.y + inset;
    const width = Math.max(0, item.width - inset * 2);
    const height = Math.max(0, item.height - inset * 2);
    if (width <= 0.001 || height <= 0.001) continue;
    const itemRadius = Math.max(0, item.radius - inset);
    parts.push(`<rect id="${item.name}" x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${width.toFixed(2)}" height="${height.toFixed(2)}" rx="${itemRadius.toFixed(2)}" opacity="${item.opacity.toFixed(3)}"/>`);
  }
  if (shapes.dot.opacity > 0.001 && shapes.dot.r > 0.001) {
    const size = Math.max(0, shapes.dot.r * 2 - inset * 2);
    if (size <= 0.001) {
      parts.push('</g>');
      return parts.join('');
    }
    const dotRadius = Math.min(renderRadius, size / 2);
    parts.push(`<rect id="${shapes.dot.name}" x="${(shapes.dot.cx - size / 2).toFixed(2)}" y="${(shapes.dot.cy - size / 2).toFixed(2)}" width="${size.toFixed(2)}" height="${size.toFixed(2)}" rx="${dotRadius.toFixed(2)}" opacity="${shapes.dot.opacity.toFixed(3)}"/>`);
  }
  parts.push('</g>');
  return parts.join('');
}

function insetExtension(extension, inset) {
  if (extension === null || extension === undefined) return null;
  if (Array.isArray(extension)) return [Math.max(0, Number(extension[0]) - inset), Math.max(0, Number(extension[1]) - inset)];
  return Math.max(0, Number(extension) - inset);
}

function segmentRectSvg(name, points, stroke, radius, opacity, extension = null) {
  if (points.length < 2) return '';
  const start = points[0];
  const end = points[points.length - 1];
  const length = dist(start, end);
  if (length <= 0.01) return '';
  const angle = Math.atan2(end[1] - start[1], end[0] - start[0]) * 180 / Math.PI;
  const [startExt, endExt] = segmentExtensions(extension, stroke / 2);
  const visualLength = length + startExt + endExt;
  const r = Math.min(radius, stroke / 2, visualLength / 2);
  return `<rect id="${name}" x="${(-startExt).toFixed(2)}" y="${(-stroke / 2).toFixed(2)}" width="${visualLength.toFixed(2)}" height="${stroke.toFixed(2)}" rx="${r.toFixed(2)}" transform="translate(${start[0].toFixed(2)},${start[1].toFixed(2)}) rotate(${angle.toFixed(3)})" opacity="${opacity.toFixed(3)}"/>`;
}

function segmentExtensions(extension, defaultValue) {
  if (extension === null || extension === undefined) return [defaultValue, defaultValue];
  if (Array.isArray(extension)) return [Number(extension[0]), Number(extension[1])];
  const value = Number(extension);
  return [value, value];
}
"""


PREVIEW_HTML = (
    r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>tibs transition preview</title>
  <style>
    :root {
      --panel: #181818;
      --border: #303030;
      --text: #f1f1f1;
      --muted: #a8a8a8;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101010;
      color: var(--text);
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    main {
      display: grid;
      place-items: center;
      padding: 24px;
    }
    .stage {
      width: min(100%, 960px);
      aspect-ratio: 800 / 360;
      background: #000;
      border: 1px solid var(--border);
      overflow: hidden;
    }
    .stage svg {
      display: block;
      width: 100%;
      height: 100%;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(6, minmax(110px, 1fr));
      gap: 12px;
      align-items: end;
      padding: 14px;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
    }
    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    input, button {
      height: 34px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #0f0f0f;
      color: var(--text);
      padding: 0 10px;
      font: inherit;
    }
    button {
      cursor: pointer;
      background: #242424;
      font-weight: 700;
    }
    @media (max-width: 820px) {
      .controls {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
  </style>
</head>
<body>
  <section class="controls">
    <label>Stroke<input id="strokeWidth" type="number" min="1" step="1" value="30"></label>
    <label>Radius<input id="cornerRadius" type="number" min="0" step="1" value="4"></label>
    <label>Trace frames<input id="traceFrameCount" type="number" min="2" max="80" step="1" value="28"></label>
    <label>Start hold<input id="startHoldMs" type="number" min="0" step="50" value="400"></label>
    <label>Trace duration<input id="traceDurationMs" type="number" min="100" step="50" value="650"></label>
    <label>Morph duration<input id="durationMs" type="number" min="100" step="50" value="1100"></label>
    <label>Pause<input id="initialDelayMs" type="number" min="0" step="50" value="0"></label>
    <label>Box size<input id="boxSize" type="number" min="20" step="1" value="112"></label>
    <label>Spacing<input id="boxGap" type="number" min="0" step="1" value="38"></label>
    <label>X<input id="originX" type="number" step="1" value="155"></label>
    <label>Y<input id="originY" type="number" step="1" value="128"></label>
    <label>T top left<input id="tBarLeft" type="number" min="0" step="1" value="30"></label>
    <label>T top right<input id="tBarRight" type="number" min="0" step="1" value="50"></label>
    <label>T foot<input id="tFootRight" type="number" min="0" step="1" value="70"></label>
    <label>Ascender<input id="ascenderHeight" type="number" min="0" step="1" value="42"></label>
    <label>T ascender<input id="tAscenderHeight" type="number" min="0" step="1" value="42"></label>
    <label>I top<input id="iTopLeft" type="number" min="0" step="1" value="24"></label>
    <label>Border size<input id="borderSize" type="number" min="0" step="0.5" value="3"></label>
    <label>Border color<input id="borderColor" value="#0f5f9a"></label>
    <label>Fill color<input id="color" value="#281DF6"></label>
    <label>Background<input id="background" value="#000000"></label>
    <button id="replay">Replay</button>
  </section>
  <main>
    <div class="stage" id="stage"></div>
  </main>
  <script>
"""
    + PREVIEW_GENERATOR_JS
    + r"""
    const stage = document.getElementById('stage');
    let start = performance.now();

    function readConfig() {
      return {
        ...DEFAULTS,
        strokeWidth: Number(document.getElementById('strokeWidth').value) || DEFAULTS.strokeWidth,
        cornerRadius: Number(document.getElementById('cornerRadius').value) || DEFAULTS.cornerRadius,
        traceFrameCount: Math.max(2, Math.round(numberOrDefault(document.getElementById('traceFrameCount').value, DEFAULTS.traceFrameCount))),
        startHoldMs: Math.max(0, numberOrDefault(document.getElementById('startHoldMs').value, DEFAULTS.startHoldMs)),
        traceDurationMs: Number(document.getElementById('traceDurationMs').value) || DEFAULTS.traceDurationMs,
        durationMs: Number(document.getElementById('durationMs').value) || DEFAULTS.durationMs,
        initialDelayMs: Math.max(0, numberOrDefault(document.getElementById('initialDelayMs').value, DEFAULTS.initialDelayMs)),
        boxSize: Number(document.getElementById('boxSize').value) || DEFAULTS.boxSize,
        boxGap: Number(document.getElementById('boxGap').value) || DEFAULTS.boxGap,
        originX: Number(document.getElementById('originX').value) || DEFAULTS.originX,
        originY: Number(document.getElementById('originY').value) || DEFAULTS.originY,
        tBarLeft: Number(document.getElementById('tBarLeft').value) || DEFAULTS.tBarLeft,
        tBarRight: Number(document.getElementById('tBarRight').value) || DEFAULTS.tBarRight,
        tFootRight: Number(document.getElementById('tFootRight').value) || DEFAULTS.tFootRight,
        ascenderHeight: Number(document.getElementById('ascenderHeight').value) || DEFAULTS.ascenderHeight,
        tAscenderHeight: Number(document.getElementById('tAscenderHeight').value) || DEFAULTS.tAscenderHeight,
        iTopLeft: Number(document.getElementById('iTopLeft').value) || DEFAULTS.iTopLeft,
        borderSize: Math.max(0, numberOrDefault(document.getElementById('borderSize').value, DEFAULTS.borderSize)),
        borderColor: document.getElementById('borderColor').value || DEFAULTS.borderColor,
        color: document.getElementById('color').value || DEFAULTS.color,
        background: document.getElementById('background').value || DEFAULTS.background
      };
    }

    function tick(now) {
      const config = readConfig();
      const elapsed = now - start;
      const rawT = clamp(elapsed / totalDurationMillis(config));
      stage.innerHTML = svgFrame(rawT, config, true);
      if (elapsed < totalDurationMillis(config)) requestAnimationFrame(tick);
    }

    function replay() {
      start = performance.now();
      requestAnimationFrame(tick);
    }

    document.getElementById('replay').addEventListener('click', replay);
    for (const id of ['strokeWidth', 'cornerRadius', 'traceFrameCount', 'startHoldMs', 'traceDurationMs', 'durationMs', 'initialDelayMs', 'boxSize', 'boxGap', 'originX', 'originY', 'tBarLeft', 'tBarRight', 'tFootRight', 'ascenderHeight', 'tAscenderHeight', 'iTopLeft', 'borderSize', 'borderColor', 'color', 'background']) {
      document.getElementById(id).addEventListener('change', replay);
      document.getElementById(id).addEventListener('input', () => {
        if (!['startHoldMs', 'traceDurationMs', 'durationMs', 'initialDelayMs'].includes(id)) stage.innerHTML = svgFrame(1, readConfig(), true);
      });
    }
    replay();
  </script>
</body>
</html>
"""
)


def draw_line_antialias(
    image: Image.Image,
    points: list[tuple[float, float]],
    fill: tuple[int, int, int, int],
    width: int,
    radius: float,
    extension: float | None = None,
):
    if len(points) < 2 or fill[3] <= 0:
        return
    start = points[0]
    end = points[-1]
    length = dist(start, end)
    if length <= 0.01:
        return
    polygon = rounded_segment_polygon(start, end, width, radius, extension)
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).polygon(polygon, fill=fill)
    image.alpha_composite(layer)


def rounded_segment_polygon(
    start: tuple[float, float],
    end: tuple[float, float],
    width: float,
    radius: float,
    extension: float | None = None,
) -> list[tuple[float, float]]:
    length = dist(start, end)
    ux = (end[0] - start[0]) / length
    uy = (end[1] - start[1]) / length
    nx = -uy
    ny = ux
    half = width / 2
    start_ext, end_ext = segment_extensions(extension, half)
    left = -start_ext
    right = length + end_ext
    r = min(radius, half, (right - left) / 2)
    if r <= 0:
        local = [(left, -half), (right, -half), (right, half), (left, half)]
    else:
        local: list[tuple[float, float]] = []
        arc_defs = [
            ((right - r, -half + r), -90, 0),
            ((right - r, half - r), 0, 90),
            ((left + r, half - r), 90, 180),
            ((left + r, -half + r), 180, 270),
        ]
        for (cx, cy), start_angle, end_angle in arc_defs:
            for step in range(6):
                angle = math.radians(start_angle + (end_angle - start_angle) * step / 5)
                local.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))
    return [(start[0] + ux * x + nx * y, start[1] + uy * x + ny * y) for x, y in local]


def hex_to_rgba(hex_color: str, opacity: float = 1.0) -> tuple[int, int, int, int]:
    if is_transparent(hex_color):
        return (0, 0, 0, 0)
    h = str(hex_color).replace("#", "").strip()
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    return (
        int(h[0:2], 16),
        int(h[2:4], 16),
        int(h[4:6], 16),
        round(255 * clamp(opacity)),
    )


def scaled_extension(extension: object, scale: int, inset: float = 0.0) -> float | tuple[float, float] | None:
    adjusted = inset_extension(extension, inset)
    if isinstance(adjusted, tuple):
        return (adjusted[0] * scale, adjusted[1] * scale)
    return None if adjusted is None else adjusted * scale


def draw_shapes_layer(
    layer: Image.Image,
    shapes: dict,
    config: dict,
    fill_color: str,
    scale: int,
    inset: float = 0.0,
) -> None:
    stroke = int(round(max(0.0, config["stroke_width"] - inset * 2) * scale))
    radius = max(0.0, config["corner_radius"] - inset) * scale
    for item in shapes["paths"]:
        if item["opacity"] <= 0.001 or stroke <= 0:
            continue
        pts = [(x * scale, y * scale) for x, y in item["points"]]
        draw_line_antialias(
            layer,
            pts,
            hex_to_rgba(fill_color, item["opacity"]),
            stroke,
            radius,
            scaled_extension(item.get("extension"), scale, inset),
        )
    for item in shapes.get("rects", []):
        if item["opacity"] <= 0.001:
            continue
        width = max(0.0, item["width"] - inset * 2)
        height = max(0.0, item["height"] - inset * 2)
        if width <= 0.001 or height <= 0.001:
            continue
        item_layer = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(item_layer)
        draw.rounded_rectangle(
            (
                (item["x"] + inset) * scale,
                (item["y"] + inset) * scale,
                (item["x"] + inset + width) * scale,
                (item["y"] + inset + height) * scale,
            ),
            radius=max(0.0, item["radius"] - inset) * scale,
            fill=hex_to_rgba(fill_color, item["opacity"]),
        )
        layer.alpha_composite(item_layer)
    dot = shapes["dot"]
    if dot["opacity"] > 0.001 and dot["r"] > 0.001:
        dot_layer = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(dot_layer)
        cx, cy = dot["cx"] * scale, dot["cy"] * scale
        size = max(0.0, dot["r"] * 2 - inset * 2) * scale
        if size <= 0.001:
            return
        r = min(radius, size / 2)
        draw.rounded_rectangle(
            (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2),
            radius=r,
            fill=hex_to_rgba(fill_color, dot["opacity"]),
        )
        layer.alpha_composite(dot_layer)


def render_frame_png(raw_t: float, config: dict = DEFAULTS, scale: int = 2) -> Image.Image:
    width, height = config["width"], config["height"]
    img = Image.new("RGBA", (width * scale, height * scale), hex_to_rgba(config["background"], 1))
    shapes = frame_shapes(raw_t, config)
    if border_enabled(config):
        border_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw_shapes_layer(border_layer, shapes, config, str(config["border_color"]), scale)
        img.alpha_composite(border_layer)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_shapes_layer(layer, shapes, config, str(config["color"]), scale, border_size(config) if border_enabled(config) else 0)
    img.alpha_composite(layer)
    return img.resize((width, height), Image.Resampling.LANCZOS)


def js_config(config: dict) -> dict:
    return {
        "width": config["width"],
        "height": config["height"],
        "frameCount": config["frame_count"],
        "traceFrameCount": config["trace_frame_count"],
        "startHoldMs": config["start_hold_ms"],
        "traceDurationMs": config["trace_duration_ms"],
        "durationMs": config["duration_ms"],
        "initialDelayMs": config["initial_delay_ms"],
        "strokeWidth": config["stroke_width"],
        "cornerRadius": config["corner_radius"],
        "boxSize": config["box_size"],
        "boxGap": config["box_gap"],
        "originX": config["origin_x"],
        "originY": config["origin_y"],
        "ascenderHeight": config["ascender_height"],
        "tAscenderHeight": config["t_ascender_height"],
        "tBarLeft": config["t_bar_left"],
        "tBarRight": config["t_bar_right"],
        "tFootRight": config["t_foot_right"],
        "iTopLeft": config["i_top_left"],
        "color": config["color"],
        "background": config["background"],
        "borderSize": config["border_size"],
        "borderColor": config["border_color"],
    }


def with_js_defaults(text: str, config: dict) -> str:
    defaults = "const DEFAULTS = " + json.dumps(js_config(config), indent=2) + ";"
    return re.sub(r"const DEFAULTS = \{.*?\};", defaults, text, count=1, flags=re.S)


def with_input_defaults(html: str, config: dict) -> str:
    replacements = {
        "strokeWidth": config["stroke_width"],
        "cornerRadius": config["corner_radius"],
        "frameCount": config["frame_count"],
        "traceFrameCount": config["trace_frame_count"],
        "startHoldMs": config["start_hold_ms"],
        "traceDurationMs": config["trace_duration_ms"],
        "durationMs": config["duration_ms"],
        "initialDelayMs": config["initial_delay_ms"],
        "boxSize": config["box_size"],
        "boxGap": config["box_gap"],
        "originX": config["origin_x"],
        "originY": config["origin_y"],
        "ascenderHeight": config["ascender_height"],
        "tAscenderHeight": config["t_ascender_height"],
        "tBarLeft": config["t_bar_left"],
        "tBarRight": config["t_bar_right"],
        "tFootRight": config["t_foot_right"],
        "iTopLeft": config["i_top_left"],
        "color": config["color"],
        "background": config["background"],
        "borderSize": config["border_size"],
        "borderColor": config["border_color"],
    }
    for element_id, value in replacements.items():
        html = re.sub(
            rf'(id="{element_id}"[^>]*value=")[^"]*(")',
            rf"\g<1>{value}\2",
            html,
            count=1,
        )
    return html


def write_outputs(config: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preview_html = with_input_defaults(with_js_defaults(PREVIEW_HTML, config), config)
    preview_html = preview_html.replace("aspect-ratio: 800 / 360;", f"aspect-ratio: {config['width']} / {config['height']};")

    (OUT / "tibs-transition-preview.html").write_text(preview_html, encoding="utf-8")
    frames = [render_frame_png(frame_raw_t(i, config), config) for i in range(total_frame_count(config))]
    frame_times = frame_times_millis(config)
    durations = [max(1, round(frame_times[i + 1] - frame_times[i])) for i in range(len(frame_times) - 1)]
    final_hold_ms = int(config.get("final_hold_ms", 0))
    durations.append(final_hold_ms if final_hold_ms else max(1, round(transition_duration_millis(config) / transition_frame_count(config))))
    animated_preview = OUT / ANIMATED_PREVIEW
    frames[0].save(
        animated_preview,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=1,
    )
    shutil.copy2(animated_preview, DOC_LOGO)



if __name__ == "__main__":
    write_outputs(CONFIG)
    print(f"Wrote assets to {OUT}")
