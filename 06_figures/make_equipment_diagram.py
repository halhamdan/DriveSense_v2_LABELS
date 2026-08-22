"""
Equipment installation diagram: a labeled side-profile cutaway of the study
vehicle showing where each sensor/device was physically mounted, matching
the genre of Model_1 (UL-DD)'s Fig. 2 (camera position diagram) and Model_3
(MPD-DF)'s Fig. 4 (sensing setup body diagram), rendered as a clean 2D
technical schematic rather than a 3D render (which isn't reproducible here).

Usage:
  python make_equipment_diagram.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyBboxPatch, FancyArrowPatch, Wedge
from matplotlib.lines import Line2D

BASE = Path(__file__).parent
OUT_DIR = BASE / "paper_figures"
OUT_DIR.mkdir(exist_ok=True)

BODY = "#2b3440"
GLASS = "#cfe3ee"
GLASS_EDGE = "#8fb4c9"
SEAT = "#8a7355"
PERSON = "#d9a066"
ACCENT = "#c44e52"
ACCENT2 = "#1e6e5c"
WHEEL = "#1a1a1a"
LABEL_BG = "#f7f6f3"

fig, ax = plt.subplots(figsize=(12, 6.5))

# ---- car body silhouette (side profile, front to the right) ----
body_pts = [
    (0.5, 0.85), (0.5, 1.55), (0.75, 2.55), (1.75, 4.25), (2.95, 4.65),
    (6.55, 4.65), (7.85, 4.15), (8.85, 2.55), (9.35, 1.75), (9.35, 0.85),
]
ax.add_patch(Polygon(body_pts, closed=True, facecolor=BODY, edgecolor="none", zorder=1))

# rocker panel / underbody line
ax.plot([0.5, 9.35], [0.85, 0.85], color=BODY, linewidth=6, zorder=1, solid_capstyle="round")

# ---- glass / window cutaway ----
glass_pts = [
    (1.95, 2.75), (2.15, 3.95), (3.15, 4.35), (6.35, 4.35), (7.45, 3.95),
    (8.35, 2.75), (7.5, 2.75), (2.7, 2.75),
]
ax.add_patch(Polygon(glass_pts, closed=True, facecolor=GLASS, edgecolor=GLASS_EDGE, linewidth=1.2, zorder=2))

# wheels
for wx in (2.25, 7.65):
    ax.add_patch(Circle((wx, 0.85), 0.52, facecolor=WHEEL, edgecolor="none", zorder=3))
    ax.add_patch(Circle((wx, 0.85), 0.22, facecolor="#555", edgecolor="none", zorder=3))

# dashboard
ax.add_patch(Polygon([(7.5, 2.75), (8.7, 2.75), (8.7, 2.55), (7.5, 3.1)],
                      closed=True, facecolor="#4a4a4a", edgecolor="none", zorder=3))

# seat
ax.add_patch(FancyBboxPatch((5.55, 2.75), 0.55, 0.95, boxstyle="round,pad=0.02,rounding_size=0.08",
                             facecolor=SEAT, edgecolor="none", zorder=3))

# ---- seated driver (simplified figure) ----
ax.add_patch(FancyBboxPatch((6.15, 2.95), 0.55, 0.85, boxstyle="round,pad=0.02,rounding_size=0.15",
                             facecolor=PERSON, edgecolor="none", zorder=4))  # torso
ax.add_patch(Circle((6.55, 4.0), 0.26, facecolor=PERSON, edgecolor="none", zorder=4))  # head
# arm to steering wheel
ax.plot([6.7, 7.35], [3.35, 3.05], color=PERSON, linewidth=9, solid_capstyle="round", zorder=4)
# wrist marker (EmotiBit location)
wrist_xy = (7.1, 2.92)

# steering wheel
ax.add_patch(Circle((7.55, 3.05), 0.32, facecolor="none", edgecolor="#3a3a3a", linewidth=4, zorder=5))
ax.add_patch(Circle((7.55, 3.05), 0.06, facecolor="#3a3a3a", edgecolor="none", zorder=5))

# ---- device markers ----
gnss_xy = (4.6, 4.66)
front_cam_xy = (7.6, 4.02)
side_cam_xy = (8.15, 3.55)
vbox_xy = (8.55, 2.65)

ax.add_patch(Circle(gnss_xy, 0.07, facecolor=ACCENT2, edgecolor="white", linewidth=1, zorder=6))
ax.add_patch(Circle(front_cam_xy, 0.065, facecolor=ACCENT, edgecolor="white", linewidth=1, zorder=6))
ax.add_patch(Circle(side_cam_xy, 0.065, facecolor=ACCENT, edgecolor="white", linewidth=1, zorder=6))
ax.add_patch(Circle(vbox_xy, 0.07, facecolor="#5b3a8e", edgecolor="white", linewidth=1, zorder=6))
ax.add_patch(Circle(wrist_xy, 0.065, facecolor="#c98a1e", edgecolor="white", linewidth=1, zorder=7))

# ---- camera fields of view ----
def fov_wedge(cam_xy, target_xy, half_angle_deg, radius, color):
    import numpy as np
    dx, dy = target_xy[0] - cam_xy[0], target_xy[1] - cam_xy[1]
    ang = np.degrees(np.arctan2(dy, dx))
    w = Wedge(cam_xy, radius, ang - half_angle_deg, ang + half_angle_deg,
              facecolor=color, edgecolor="none", alpha=0.18, zorder=2)
    ax.add_patch(w)

FOV_FRONT = "#c44e52"
FOV_SIDE = "#4c72b0"
fov_wedge(front_cam_xy, (6.55, 4.0), 20, 1.25, FOV_FRONT)
fov_wedge(side_cam_xy, (6.35, 3.25), 24, 1.5, FOV_SIDE)


def callout(xy, text, label_xy, color, ha="left"):
    ax.annotate(text, xy=xy, xytext=label_xy, fontsize=9.5, color="#1a1a1a",
                ha=ha, va="center", zorder=8,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=LABEL_BG, edgecolor=color, linewidth=1.3),
                arrowprops=dict(arrowstyle="-", color=color, linewidth=1.2,
                                 shrinkA=2, shrinkB=6, connectionstyle="arc3,rad=0.08"))


callout(gnss_xy, "GNSS antenna\n(roof-mounted)", (4.3, 5.75), ACCENT2)
callout(front_cam_xy, "Front camera\n(driver-facing, facial data)", (9.9, 4.55), FOV_FRONT)
callout(side_cam_xy, "Side camera\n(in-cabin behaviour)", (10.4, 3.4), FOV_SIDE, ha="left")
callout(vbox_xy, "VBOX HD2 logger\n+ OBD-II / CAN-bus link", (10.4, 1.9), "#5b3a8e", ha="left")
callout(wrist_xy, "EmotiBit\n(EDA, PPG, HR, temp., IMU)", (9.9, 0.55), "#c98a1e", ha="left")

ax.set_xlim(-0.3, 12.4)
ax.set_ylim(0.1, 6.3)
ax.set_aspect("equal")
ax.axis("off")

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=ACCENT2, markersize=9, label='Vehicle dynamics / GNSS'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=ACCENT, markersize=9, label='Video (facial / in-cabin)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor="#5b3a8e", markersize=9, label='Data logger'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor="#c98a1e", markersize=9, label='Wearable biometric'),
]
ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.08),
          ncol=4, frameon=False, fontsize=9.5)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_equipment_installation.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "fig_equipment_installation.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved to {OUT_DIR}/fig_equipment_installation.{{pdf,png}}")
