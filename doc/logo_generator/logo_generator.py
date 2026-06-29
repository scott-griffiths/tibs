from __future__ import annotations

import json
import math
import re
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(".")
OUT = ROOT / "outputs"
PLUGIN_DIR = OUT / "tibs-transition-figma-plugin"
ANIMATED_PREVIEW = "tibs-transition-preview.png"
DOC_LOGO = Path(__file__).resolve().parents[1] / "tibs.png"

# Edit this block, then run this script with no command-line arguments.
CONFIG = {
    "width": 550,
    "height": 235,
    "frame_count": 28,
    "duration_ms": 500,
    "initial_delay_ms": 1000,
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
    "t_bar_right": 33,
    "t_foot_right": 56,
    "i_top_left": 24,
    "color": "#1e86de",
    "background": "transparent",
}

DEFAULTS = CONFIG


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def is_transparent(value: object) -> bool:
    return str(value).strip().lower() in {"", "none", "transparent"}


def timeline_millis(raw_t: float, config: dict = DEFAULTS) -> int:
    if raw_t <= 0:
        return 0
    return round(float(config.get("initial_delay_ms", 0)) + raw_t * float(config["duration_ms"]))


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


def rotate_endpoint(
    pivot: tuple[float, float],
    length: float,
    start_angle: float,
    end_angle: float,
    progress: float,
) -> tuple[float, float]:
    angle = mix(start_angle, end_angle, progress)
    return (pivot[0] + math.cos(angle) * length, pivot[1] + math.sin(angle) * length)


def frame_shapes(raw_t: float, config: dict = DEFAULTS) -> dict:
    p = ease_in_out_sine(raw_t)

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
        "dot": {
            "name": "i-dot",
            "cx": lx1,
            "cy": mix(y_top, y_top - asc, p),
            "r": config["stroke_width"] * 0.50,
            "opacity": 1.0,
        },
    }


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
    parts.append(f'<g id="tibs" fill="{color}">')
    for item in shapes["paths"]:
        op = item["opacity"]
        if op <= 0.001:
            continue
        parts.append(svg_segment_rect(item["name"], item["points"], stroke, radius, op, item.get("extension")))
    dot = shapes["dot"]
    if dot["opacity"] > 0.001 and dot["r"] > 0.001:
        size = dot["r"] * 2
        dot_radius = min(radius, size / 2)
        parts.append(
            f'<rect id="{dot["name"]}" x="{dot["cx"] - size / 2:.2f}" y="{dot["cy"] - size / 2:.2f}" width="{size:.2f}" height="{size:.2f}" rx="{dot_radius:.2f}" opacity="{dot["opacity"]:.3f}"/>'
        )
    parts.append("</g></svg>")
    return "".join(parts)


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
    r = min(radius, stroke / 2, length / 2)
    start_ext, end_ext = segment_extensions(extension, stroke / 2)
    return (
        f'<rect id="{name}" x="{-start_ext:.2f}" y="{-stroke / 2:.2f}" width="{length + start_ext + end_ext:.2f}" height="{stroke:.2f}" '
        f'rx="{r:.2f}" transform="translate({x1:.2f},{y1:.2f}) rotate({angle:.3f})" opacity="{opacity:.3f}"/>'
    )


def segment_extensions(extension: object, default: float) -> tuple[float, float]:
    if extension is None:
        return default, default
    if isinstance(extension, (list, tuple)):
        return float(extension[0]), float(extension[1])
    value = float(extension)
    return value, value


def svg_frames_sheet(config: dict = DEFAULTS) -> str:
    fw, fh = config["width"], config["height"]
    frame_count = config["frame_count"]
    cols = 4
    gap = 36
    label_h = 24
    rows = math.ceil(frame_count / cols)
    width = cols * fw + (cols - 1) * gap
    height = rows * (fh + label_h) + (rows - 1) * gap
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#111111"/>',
    ]
    for i in range(frame_count):
        x = (i % cols) * (fw + gap)
        y = (i // cols) * (fh + label_h + gap)
        raw_t = i / (frame_count - 1)
        parts.append(f'<g id="frame-{i:02d}" transform="translate({x},{y})">')
        parts.append(svg_frame(raw_t, config, include_background=True))
        millis = timeline_millis(raw_t, config)
        parts.append(
            f'<text x="8" y="{fh + 17}" fill="#BBBBBB" font-family="Inter, Arial, sans-serif" font-size="14">frame {i + 1:02d} / {millis}ms</text>'
        )
        parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)


PLUGIN_GENERATOR_JS = r"""
const DEFAULTS = {
  width: 800,
  height: 360,
  frameCount: 28,
  durationMs: 1100,
  initialDelayMs: 1500,
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
  background: '#000000'
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

function rotateEndpoint(pivot, length, startAngle, endAngle, progress) {
  const angle = mix(startAngle, endAngle, progress);
  return [
    pivot[0] + Math.cos(angle) * length,
    pivot[1] + Math.sin(angle) * length
  ];
}

function frameShapes(rawT, config) {
  const p = easeInOutSine(rawT);
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
    dot: {
      name: 'i-dot',
      cx: lx1,
      cy: mix(yTop, yTop - asc, p),
      r: config.strokeWidth * 0.50,
      opacity: 1
    }
  };
}

function svgFrame(rawT, config, includeBackground = true) {
  const shapes = frameShapes(rawT, config);
  const parts = [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${config.width} ${config.height}" width="${config.width}" height="${config.height}">`
  ];
  if (includeBackground && !isTransparent(config.background)) {
    parts.push(`<rect id="background" width="${config.width}" height="${config.height}" fill="${config.background}"/>`);
  }
  parts.push(`<g id="tibs" fill="${config.color}">`);
  for (const item of shapes.paths) {
    if (item.opacity <= 0.001) continue;
    parts.push(segmentRectSvg(item.name, item.points, config.strokeWidth, config.cornerRadius, item.opacity, item.extension));
  }
  if (shapes.dot.opacity > 0.001 && shapes.dot.r > 0.001) {
    const size = shapes.dot.r * 2;
    const radius = Math.min(config.cornerRadius, size / 2);
    parts.push(`<rect id="${shapes.dot.name}" x="${(shapes.dot.cx - size / 2).toFixed(2)}" y="${(shapes.dot.cy - size / 2).toFixed(2)}" width="${size.toFixed(2)}" height="${size.toFixed(2)}" rx="${radius.toFixed(2)}" opacity="${shapes.dot.opacity.toFixed(3)}"/>`);
  }
  parts.push('</g></svg>');
  return parts.join('');
}

function segmentRectSvg(name, points, stroke, radius, opacity, extension = null) {
  if (points.length < 2) return '';
  const start = points[0];
  const end = points[points.length - 1];
  const length = dist(start, end);
  if (length <= 0.01) return '';
  const angle = Math.atan2(end[1] - start[1], end[0] - start[0]) * 180 / Math.PI;
  const r = Math.min(radius, stroke / 2, length / 2);
  const [startExt, endExt] = segmentExtensions(extension, stroke / 2);
  return `<rect id="${name}" x="${(-startExt).toFixed(2)}" y="${(-stroke / 2).toFixed(2)}" width="${(length + startExt + endExt).toFixed(2)}" height="${stroke.toFixed(2)}" rx="${r.toFixed(2)}" transform="translate(${start[0].toFixed(2)},${start[1].toFixed(2)}) rotate(${angle.toFixed(3)})" opacity="${opacity.toFixed(3)}"/>`;
}

function segmentExtensions(extension, defaultValue) {
  if (extension === null || extension === undefined) return [defaultValue, defaultValue];
  if (Array.isArray(extension)) return [Number(extension[0]), Number(extension[1])];
  const value = Number(extension);
  return [value, value];
}
"""


PLUGIN_CODE_JS = (
    PLUGIN_GENERATOR_JS
    + r"""

figma.showUI(__html__, { width: 320, height: 730, themeColors: true });

figma.ui.onmessage = async (message) => {
  if (message.type !== 'generate') return;
  const initialDelay = numberOrDefault(message.config.initialDelayMs, DEFAULTS.initialDelayMs);
  const config = {
    ...DEFAULTS,
    ...message.config,
    frameCount: Math.max(2, Math.min(80, Math.round(Number(message.config.frameCount) || DEFAULTS.frameCount))),
    durationMs: Math.max(100, Math.round(Number(message.config.durationMs) || DEFAULTS.durationMs)),
    initialDelayMs: Math.max(0, Math.round(initialDelay)),
    strokeWidth: Math.max(1, Number(message.config.strokeWidth) || DEFAULTS.strokeWidth),
    cornerRadius: Math.max(0, Number(message.config.cornerRadius) || DEFAULTS.cornerRadius)
  };

  const page = figma.createPage();
  page.name = `tibs transition ${config.frameCount}f`;
  figma.currentPage = page;

  const gap = 56;
  const cols = 4;
  const created = [];
  for (let i = 0; i < config.frameCount; i++) {
    const rawT = i / (config.frameCount - 1);
    const millis = i === 0 ? 0 : Math.round(config.initialDelayMs + rawT * config.durationMs);
    const frame = figma.createFrame();
    frame.name = `tibs ${String(i + 1).padStart(2, '0')} - ${millis}ms`;
    frame.resize(config.width, config.height);
    frame.x = (i % cols) * (config.width + gap);
    frame.y = Math.floor(i / cols) * (config.height + gap);
    frame.fills = isTransparent(config.background) ? [] : [{ type: 'SOLID', color: hexToRgb(config.background) }];

    const node = figma.createNodeFromSvg(svgFrame(rawT, config, false));
    node.name = `editable vectors ${String(i + 1).padStart(2, '0')}`;
    node.x = 0;
    node.y = 0;
    frame.appendChild(node);
    page.appendChild(frame);
    created.push(frame);
  }

  figma.currentPage.selection = created;
  figma.viewport.scrollAndZoomIntoView(created);
  figma.closePlugin(`Created ${created.length} editable vector frames with shared ease-in-out sine timing.`);
};

function hexToRgb(hex) {
  const cleaned = String(hex || '#000000').replace('#', '').trim();
  const full = cleaned.length === 3
    ? cleaned.split('').map((ch) => ch + ch).join('')
    : cleaned.padEnd(6, '0').slice(0, 6);
  return {
    r: parseInt(full.slice(0, 2), 16) / 255,
    g: parseInt(full.slice(2, 4), 16) / 255,
    b: parseInt(full.slice(4, 6), 16) / 255
  };
}
"""
)


PLUGIN_UI_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    :root {
      color-scheme: light dark;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 12px;
    }
    body {
      margin: 0;
      padding: 16px;
      background: var(--figma-color-bg);
      color: var(--figma-color-text);
    }
    label {
      display: grid;
      gap: 6px;
      margin-bottom: 12px;
      font-weight: 600;
    }
    input {
      height: 32px;
      border: 1px solid var(--figma-color-border);
      border-radius: 6px;
      padding: 0 9px;
      background: var(--figma-color-bg-secondary);
      color: var(--figma-color-text);
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    button {
      width: 100%;
      height: 36px;
      border: 0;
      border-radius: 6px;
      background: var(--figma-color-bg-brand);
      color: var(--figma-color-text-onbrand);
      font-weight: 700;
      cursor: pointer;
    }
    p {
      color: var(--figma-color-text-secondary);
      line-height: 1.45;
      margin: 0 0 14px;
    }
  </style>
</head>
<body>
  <p>Generates editable vector frames from boxes to tibs. Re-run with new values to iterate.</p>
  <div class="row">
    <label>Stroke<input id="strokeWidth" type="number" min="1" step="1" value="30"></label>
    <label>Radius<input id="cornerRadius" type="number" min="0" step="1" value="4"></label>
  </div>
  <div class="row">
    <label>Frames<input id="frameCount" type="number" min="2" max="80" step="1" value="28"></label>
    <label>Duration<input id="durationMs" type="number" min="100" step="50" value="1100"></label>
  </div>
  <div class="row">
    <label>Initial delay<input id="initialDelayMs" type="number" min="0" step="50" value="1500"></label>
  </div>
  <div class="row">
    <label>Box size<input id="boxSize" type="number" min="20" step="1" value="112"></label>
    <label>Spacing<input id="boxGap" type="number" min="0" step="1" value="38"></label>
  </div>
  <div class="row">
    <label>X<input id="originX" type="number" step="1" value="155"></label>
    <label>Y<input id="originY" type="number" step="1" value="128"></label>
  </div>
  <div class="row">
    <label>T top left<input id="tBarLeft" type="number" min="0" step="1" value="30"></label>
    <label>T top right<input id="tBarRight" type="number" min="0" step="1" value="50"></label>
  </div>
  <div class="row">
    <label>T foot<input id="tFootRight" type="number" min="0" step="1" value="70"></label>
    <label>Ascender<input id="ascenderHeight" type="number" min="0" step="1" value="42"></label>
  </div>
  <div class="row">
    <label>T ascender<input id="tAscenderHeight" type="number" min="0" step="1" value="42"></label>
    <label>I top<input id="iTopLeft" type="number" min="0" step="1" value="24"></label>
  </div>
  <label>Color<input id="color" value="#281DF6"></label>
  <label>Background<input id="background" value="#000000"></label>
  <button id="generate">Generate Frames</button>
  <script>
    document.getElementById('generate').onclick = () => {
      const ids = ['strokeWidth', 'cornerRadius', 'frameCount', 'durationMs', 'initialDelayMs', 'boxSize', 'boxGap', 'originX', 'originY', 'tBarLeft', 'tBarRight', 'tFootRight', 'ascenderHeight', 'tAscenderHeight', 'iTopLeft', 'color', 'background'];
      const config = Object.fromEntries(ids.map((id) => [id, document.getElementById(id).value]));
      parent.postMessage({ pluginMessage: { type: 'generate', config } }, '*');
    };
  </script>
</body>
</html>
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
    <label>Duration<input id="durationMs" type="number" min="100" step="50" value="1100"></label>
    <label>Initial delay<input id="initialDelayMs" type="number" min="0" step="50" value="1500"></label>
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
    <label>Color<input id="color" value="#281DF6"></label>
    <label>Background<input id="background" value="#000000"></label>
    <button id="replay">Replay</button>
  </section>
  <main>
    <div class="stage" id="stage"></div>
  </main>
  <script>
"""
    + PLUGIN_GENERATOR_JS
    + r"""
    const stage = document.getElementById('stage');
    let start = performance.now();

    function readConfig() {
      return {
        ...DEFAULTS,
        strokeWidth: Number(document.getElementById('strokeWidth').value) || DEFAULTS.strokeWidth,
        cornerRadius: Number(document.getElementById('cornerRadius').value) || DEFAULTS.cornerRadius,
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
        color: document.getElementById('color').value || DEFAULTS.color,
        background: document.getElementById('background').value || DEFAULTS.background
      };
    }

    function tick(now) {
      const config = readConfig();
      const elapsed = now - start;
      const rawT = clamp((elapsed - config.initialDelayMs) / config.durationMs);
      stage.innerHTML = svgFrame(rawT, config, true);
      if (elapsed < config.initialDelayMs + config.durationMs) requestAnimationFrame(tick);
    }

    function replay() {
      start = performance.now();
      requestAnimationFrame(tick);
    }

    document.getElementById('replay').addEventListener('click', replay);
    for (const id of ['strokeWidth', 'cornerRadius', 'durationMs', 'initialDelayMs', 'boxSize', 'boxGap', 'originX', 'originY', 'tBarLeft', 'tBarRight', 'tFootRight', 'ascenderHeight', 'tAscenderHeight', 'iTopLeft', 'color', 'background']) {
      document.getElementById(id).addEventListener('change', replay);
      document.getElementById(id).addEventListener('input', () => {
        if (!['durationMs', 'initialDelayMs'].includes(id)) stage.innerHTML = svgFrame(1, readConfig(), true);
      });
    }
    replay();
  </script>
</body>
</html>
"""
)


README = """# tibs transition

This package contains an editable vector rebuild of the supplied animated PNG, simplified into a one-way transition from three boxes to `tibs`.

The animation uses persistent stroke segments rather than fades: the first box becomes a three-stroke `t` plus the `i`, the `i` dot rises out of the top of its stem, the second box mostly holds position as the `b`, and the third box splits its vertical sides so the folded halves form the middle bar of the `s`.

## Best Figma path

1. In Figma, choose `Plugins > Development > Import plugin from manifest...`.
2. Select `tibs-transition-figma-plugin/manifest.json`.
3. Run `Plugins > Development > tibs transition generator`.
4. Adjust stroke, corner radius, color, spacing, size, duration, and frame count in the plugin UI, then generate.

The plugin creates editable vector frames on a new Figma page. The frames use one shared ease-in-out sine timing curve sampled into 28 frames by default. The animated PNG and browser preview hold on the first frame for `initial_delay_ms` before the transition starts. The animated PNG plays once and then holds on the final frame.

## Other files

- `tibs-transition-frames.svg`: all generated frames in a grid, importable into Figma as vectors.
- `tibs-transition-preview.html`: browser preview with live controls for stroke, radius, color, spacing, size, and duration.
- `tibs_transition_generator.py`: configurable Python generator for regenerating the package.
- `tibs-transition-preview.png`: non-editable animated PNG preview of the generated animation. It plays once and then holds on the final frame. Use this file in web pages with the normal `image/png` MIME type. The generator also copies this file to `doc/tibs.png`.
"""


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
    r = min(radius, width / 2, length / 2)
    polygon = rounded_segment_polygon(start, end, width, r, extension)
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
    r = min(radius, half, length / 2)
    start_ext, end_ext = segment_extensions(extension, half)
    left = -start_ext
    right = length + end_ext
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


def render_frame_png(raw_t: float, config: dict = DEFAULTS, scale: int = 2) -> Image.Image:
    width, height = config["width"], config["height"]
    img = Image.new("RGBA", (width * scale, height * scale), hex_to_rgba(config["background"], 1))
    shapes = frame_shapes(raw_t, config)
    stroke = int(round(config["stroke_width"] * scale))
    radius = config["corner_radius"] * scale
    for item in shapes["paths"]:
        if item["opacity"] <= 0.001:
            continue
        pts = [(x * scale, y * scale) for x, y in item["points"]]
        raw_extension = item.get("extension")
        if isinstance(raw_extension, (list, tuple)):
            extension = (raw_extension[0] * scale, raw_extension[1] * scale)
        else:
            extension = None if raw_extension is None else raw_extension * scale
        draw_line_antialias(img, pts, hex_to_rgba(config["color"], item["opacity"]), stroke, radius, extension)
    dot = shapes["dot"]
    if dot["opacity"] > 0.001 and dot["r"] > 0.001:
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        cx, cy = dot["cx"] * scale, dot["cy"] * scale
        size = dot["r"] * 2 * scale
        r = min(radius, size / 2)
        draw.rounded_rectangle(
            (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2),
            radius=r,
            fill=hex_to_rgba(config["color"], dot["opacity"]),
        )
        img.alpha_composite(layer)
    return img.resize((width, height), Image.Resampling.LANCZOS)


def render_contact_png(config: dict = DEFAULTS) -> Image.Image:
    cols = 4
    thumb_scale = 0.5
    fw = int(config["width"] * thumb_scale)
    fh = int(config["height"] * thumb_scale)
    label_h = 24
    gap = 18
    rows = math.ceil(config["frame_count"] / cols)
    sheet = Image.new(
        "RGBA",
        (cols * fw + (cols - 1) * gap, rows * (fh + label_h) + (rows - 1) * gap),
        (17, 17, 17, 255),
    )
    draw = ImageDraw.Draw(sheet)
    for i in range(config["frame_count"]):
        raw_t = i / (config["frame_count"] - 1)
        frame = render_frame_png(raw_t, config).resize((fw, fh), Image.Resampling.LANCZOS)
        x = (i % cols) * (fw + gap)
        y = (i // cols) * (fh + label_h + gap)
        sheet.alpha_composite(frame, (x, y))
        draw.text((x + 6, y + fh + 5), f"{i + 1:02d}  {timeline_millis(raw_t, config)}ms", fill=(190, 190, 190, 255))
    return sheet


def js_config(config: dict) -> dict:
    return {
        "width": config["width"],
        "height": config["height"],
        "frameCount": config["frame_count"],
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
    }


def with_js_defaults(text: str, config: dict) -> str:
    defaults = "const DEFAULTS = " + json.dumps(js_config(config), indent=2) + ";"
    return re.sub(r"const DEFAULTS = \{.*?\};", defaults, text, count=1, flags=re.S)


def with_input_defaults(html: str, config: dict) -> str:
    replacements = {
        "strokeWidth": config["stroke_width"],
        "cornerRadius": config["corner_radius"],
        "frameCount": config["frame_count"],
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
    }
    for element_id, value in replacements.items():
        html = re.sub(
            rf'(id="{element_id}"[^>]*value=")[^"]*(")',
            rf"\g<1>{value}\2",
            html,
            count=1,
        )
    return html


def build_readme(config: dict) -> str:
    return README + """

## Regenerate

Edit the hard-coded `CONFIG` block at the top of `tibs_transition_generator.py`, then run the generator with no command-line options:

```bash
python3 tibs_transition_generator.py
```
"""


def write_outputs(config: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PLUGIN_DIR.mkdir(parents=True, exist_ok=True)

    plugin_code = with_js_defaults(PLUGIN_CODE_JS, config)
    plugin_ui = with_input_defaults(PLUGIN_UI_HTML, config)
    preview_html = with_input_defaults(with_js_defaults(PREVIEW_HTML, config), config)
    preview_html = preview_html.replace("aspect-ratio: 800 / 360;", f"aspect-ratio: {config['width']} / {config['height']};")

    (OUT / "tibs-transition-frames.svg").write_text(svg_frames_sheet(config), encoding="utf-8")
    (OUT / "tibs-transition-preview.html").write_text(preview_html, encoding="utf-8")
    (OUT / "README-tibs-transition.md").write_text(build_readme(config), encoding="utf-8")

    (PLUGIN_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "name": "tibs transition generator",
                "id": "tibs-transition-generator-local",
                "api": "1.0.0",
                "main": "code.js",
                "ui": "ui.html",
                "editorType": ["figma"],
                "documentAccess": "dynamic-page",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (PLUGIN_DIR / "code.js").write_text(plugin_code, encoding="utf-8")
    (PLUGIN_DIR / "ui.html").write_text(plugin_ui, encoding="utf-8")

    frames = [render_frame_png(i / (config["frame_count"] - 1), config) for i in range(config["frame_count"])]
    frame_duration = round(config["duration_ms"] / config["frame_count"])
    durations = [frame_duration] * config["frame_count"]
    durations[0] += int(config.get("initial_delay_ms", 0))
    final_hold_ms = int(config.get("final_hold_ms", 0))
    if final_hold_ms:
        durations[-1] = final_hold_ms
    animated_preview = OUT / ANIMATED_PREVIEW
    frames[0].save(
        animated_preview,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=1,
    )
    shutil.copy2(animated_preview, DOC_LOGO)
    render_contact_png(config).save(OUT / "tibs-transition-contact.png")

    script_copy = OUT / "tibs_transition_generator.py"
    source = Path(__file__).resolve()
    if source != script_copy.resolve():
        shutil.copy2(source, script_copy)

    zip_path = OUT / "tibs-transition-package.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in [
            OUT / "README-tibs-transition.md",
            OUT / "tibs-transition-frames.svg",
            OUT / "tibs-transition-preview.html",
            OUT / ANIMATED_PREVIEW,
            OUT / "tibs_transition_generator.py",
            PLUGIN_DIR / "manifest.json",
            PLUGIN_DIR / "code.js",
            PLUGIN_DIR / "ui.html",
        ]:
            zf.write(path, path.relative_to(OUT))


if __name__ == "__main__":
    write_outputs(CONFIG)
    print(f"Wrote assets to {OUT}")
