#!/usr/bin/env python3
"""Create a local HTML tool for drawing manual trajectories on an XODR BEV image."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bev-image", required=True, type=Path)
    parser.add_argument("--bev-metadata", type=Path, help="Projection metadata JSON written next to the BEV image.")
    parser.add_argument("--output-html", required=True, type=Path)
    parser.add_argument("--default-class-name", default="vehicle.car")
    return parser.parse_args()


def relpath(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve()).replace(os.sep, "/")


def main() -> int:
    args = parse_args()
    metadata_path = args.bev_metadata or args.bev_image.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    projection = metadata.get("bev_metadata", metadata)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    image_src = relpath(args.bev_image, args.output_html.parent)
    payload = {
        "imageSrc": image_src,
        "projection": projection,
        "defaultClassName": args.default_class_name,
    }
    html = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    args.output_html.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output_html}")
    return 0


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>XODR BEV Trajectory Editor</title>
<style>
  :root {
    color-scheme: light;
    --bg: #f6f7f9;
    --panel: #ffffff;
    --text: #1d2430;
    --muted: #657085;
    --line: #d8dee8;
    --accent: #c51f5d;
    --blue: #2864d8;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    height: 100vh;
    display: grid;
    grid-template-rows: auto 1fr;
    background: var(--bg);
    color: var(--text);
    font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    border-bottom: 1px solid var(--line);
    background: var(--panel);
  }
  button, input {
    height: 32px;
    border: 1px solid var(--line);
    background: #fff;
    color: var(--text);
    border-radius: 6px;
    padding: 0 10px;
    font: inherit;
  }
  button.primary { background: var(--blue); color: #fff; border-color: var(--blue); }
  button.warn { color: #a31238; }
  label { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); }
  #status { margin-left: auto; color: var(--muted); white-space: nowrap; }
  main {
    min-height: 0;
    display: grid;
    grid-template-columns: 1fr 300px;
  }
  #stageWrap {
    overflow: auto;
    min-width: 0;
    min-height: 0;
    background: #20242c;
  }
  #stage {
    position: relative;
    width: max-content;
    height: max-content;
    cursor: crosshair;
  }
  #bev, #draw {
    display: block;
  }
  #draw {
    position: absolute;
    left: 0;
    top: 0;
  }
  aside {
    overflow: auto;
    border-left: 1px solid var(--line);
    background: var(--panel);
    padding: 12px;
  }
  .track {
    display: grid;
    grid-template-columns: 24px 1fr auto;
    align-items: center;
    gap: 8px;
    padding: 8px 0;
    border-bottom: 1px solid var(--line);
  }
  .swatch {
    width: 16px;
    height: 16px;
    border-radius: 50%;
  }
  .small { color: var(--muted); font-size: 12px; }
  @media (max-width: 900px) {
    main { grid-template-columns: 1fr; }
    aside { max-height: 220px; border-left: 0; border-top: 1px solid var(--line); }
  }
</style>
</head>
<body>
<header>
  <button id="newTrack" class="primary">New Track</button>
  <button id="undoPoint">Undo Point</button>
  <button id="deleteTrack" class="warn">Delete Track</button>
  <button id="saveJson">Save JSON</button>
  <label>Class <input id="className" value=""></label>
  <span id="status"></span>
</header>
<main>
  <div id="stageWrap">
    <div id="stage">
      <img id="bev" alt="">
      <canvas id="draw"></canvas>
    </div>
  </div>
  <aside>
    <div class="small">Click on the BEV image to add points. Use New Track for each vehicle trajectory.</div>
    <div id="tracks"></div>
  </aside>
</main>
<script>
const config = __PAYLOAD__;
const colors = ["#d81b60", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa", "#00897b", "#e53935", "#3949ab"];
const img = document.getElementById("bev");
const canvas = document.getElementById("draw");
const ctx = canvas.getContext("2d");
const tracksEl = document.getElementById("tracks");
const statusEl = document.getElementById("status");
const classNameEl = document.getElementById("className");
let tracks = [];
let active = -1;

classNameEl.value = config.defaultClassName || "vehicle.car";
img.src = config.imageSrc;
img.onload = () => {
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  render();
};

function newTrack() {
  const id = `manual_car_${tracks.length}`;
  tracks.push({ id, class_name: classNameEl.value || "vehicle.car", points: [] });
  active = tracks.length - 1;
  render();
}

function canvasPoint(evt) {
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width;
  const sy = canvas.height / rect.height;
  return { x: Math.round((evt.clientX - rect.left) * sx), y: Math.round((evt.clientY - rect.top) * sy) };
}

canvas.addEventListener("click", (evt) => {
  if (active < 0) newTrack();
  tracks[active].class_name = classNameEl.value || "vehicle.car";
  tracks[active].points.push(canvasPoint(evt));
  render();
});

document.getElementById("newTrack").onclick = newTrack;
document.getElementById("undoPoint").onclick = () => {
  if (active >= 0) tracks[active].points.pop();
  render();
};
document.getElementById("deleteTrack").onclick = () => {
  if (active >= 0) {
    tracks.splice(active, 1);
    active = Math.min(active, tracks.length - 1);
  }
  render();
};
document.getElementById("saveJson").onclick = () => {
  const payload = {
    coordinate_frame: "pixel",
    bev_metadata: config.projection,
    tracks: tracks.filter(t => t.points.length >= 2)
  };
  const blob = new Blob([JSON.stringify(payload, null, 2) + "\\n"], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "manual_tracks.json";
  a.click();
  URL.revokeObjectURL(a.href);
};

function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  tracks.forEach((track, i) => {
    const color = colors[i % colors.length];
    ctx.lineWidth = i === active ? 5 : 3;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.beginPath();
    track.points.forEach((p, j) => {
      if (j === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
    track.points.forEach((p, j) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, j === 0 ? 6 : 4, 0, Math.PI * 2);
      ctx.fill();
    });
  });
  renderList();
}

function renderList() {
  tracksEl.innerHTML = "";
  tracks.forEach((track, i) => {
    const row = document.createElement("div");
    row.className = "track";
    row.onclick = () => { active = i; render(); };
    row.innerHTML = `
      <span class="swatch" style="background:${colors[i % colors.length]}"></span>
      <span>${track.id}<br><span class="small">${track.points.length} points</span></span>
      <button>${i === active ? "Active" : "Select"}</button>
    `;
    tracksEl.appendChild(row);
  });
  const valid = tracks.filter(t => t.points.length >= 2).length;
  statusEl.textContent = `${valid} valid tracks`;
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
