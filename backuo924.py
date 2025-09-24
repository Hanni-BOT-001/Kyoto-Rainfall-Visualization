import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
"""Matplotlib ripple animation driven by rainfall CSV.

This script looks for `kyotov03 copy.csv` i        col = color_from_rain(rain)
        # use the computed max_r so ripples never exceed ~1/220 canvas area
        r = Ripple(x, y, col, r0=0.02, max_r=max_r)
        # ensure RGBA tuple; use matplotlib helper
        from matplotlib.colors import to_rgba
        rgba = to_rgba(r.color, r.alpha)
        # create two concentric ellipses with random shapes and very thin lines
        width1, height1 = r.r * 2, r.r * 2 * r.aspect_ratio
        width2, height2 = r.r * 2 * 0.7, r.r * 2 * 0.7 * r.aspect_ratio
        c1 = Ellipse((r.x, r.y), width1, height1, angle=r.angle, fill=False, edgecolor=rgba, linewidth=0.625, zorder=2)
        c2 = Ellipse((r.x, r.y), width2, height2, angle=r.angle, fill=False, edgecolor=rgba, linewidth=0.625, zorder=2)
        ax.add_patch(c1)
        ax.add_patch(c2)
        ripples.append(r)
        artists.append([c1, c2]) directory and uses
columns containing 'rain', 'wind', 'humidity', 'temperature' (case-insensitive).
If the CSV is missing or columns are not found, the script falls back to
deterministic synthetic data so the animation still runs.

Run with:
  python main.py
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Ellipse
import argparse


CSV_CANDIDATE = os.path.join(os.path.dirname(__file__), 'kyotov03 copy.csv')


def load_data(path):
    # if file missing, deterministic synthetic fallback
    if not path or not os.path.exists(path):
        rng = np.random.RandomState(12345)
        n = 1000
        return {
            'rain': rng.uniform(0, 30, n),
            'wind_dir': rng.uniform(0, 360, n),
            'rh': rng.uniform(30, 100, n),
            'temp': rng.uniform(0, 30, n)
        }

    # Try reading directly; if pandas ParserError occurs (multi-section CSV),
    # attempt to detect a proper header line that starts with 'time,' and read from there.
    try:
        df = pd.read_csv(path)
    except Exception as e:
        # attempt header-detection fallback
        header_idx = None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = [ln.rstrip('\n') for ln in f.readlines()]
            time_header_indices = []
            for i, ln in enumerate(lines[:200]):
                if ln.lower().startswith('time,'):
                    cols = [p.strip() for p in ln.split(',')]
                    time_header_indices.append((i, len(cols)))
            if time_header_indices:
                header_idx = max(time_header_indices, key=lambda x: x[1])[0]
                df = pd.read_csv(path, skiprows=header_idx)
            else:
                # give up and use synthetic fallback
                raise
        except Exception:
            # final fallback: deterministic synthetic
            rng = np.random.RandomState(12345)
            n = 1000
            return {
                'rain': rng.uniform(0, 30, n),
                'wind_dir': rng.uniform(0, 360, n),
                'rh': rng.uniform(30, 100, n),
                'temp': rng.uniform(0, 30, n)
            }

    cols = {c.lower(): c for c in df.columns}

    def find(candidates):
        for cand in candidates:
            for k, orig in cols.items():
                if cand in k:
                    # coerce to numeric, replace non-numeric with NaN then fill with 0
                    try:
                        series = pd.to_numeric(df[orig], errors='coerce').fillna(0)
                        return series.values
                    except Exception:
                        try:
                            return df[orig].astype(float).fillna(0).values
                        except Exception:
                            return df[orig].values
        return None

    rain = find(['rain', 'precip', 'precipitation'])
    wind = find(['wind', 'wind_direction', 'winddir'])
    rh = find(['relative_humidity', 'humidity', 'rh'])
    temp = find(['temperature', 'temp', 'air_temperature'])

    n = len(df)
    if rain is None: rain = np.zeros(n)
    if wind is None: wind = np.zeros(n)
    if rh is None: rh = np.zeros(n)
    if temp is None: temp = np.zeros(n)

    return {'rain': rain, 'wind_dir': wind, 'rh': rh, 'temp': temp}


def color_from_rain(r):
    # map rainfall 0..30mm to muted Morandi color palette with slightly increased saturation
    v = max(0.0, min(1.0, r / 30.0))
    low = np.array([0.75, 0.85, 0.78])   # soft sage green with more green
    mid = np.array([0.7, 0.78, 0.88])    # muted blue-gray with more blue
    high = np.array([0.88, 0.78, 0.68])  # warm beige with more warmth
    if v < 0.5:
        t = v / 0.5
        col = low * (1 - t) + mid * t
    else:
        t = (v - 0.5) / 0.5
        col = mid * (1 - t) + high * t
    return tuple(col.tolist())


class Ripple:
    def __init__(self, x, y, color, r0=0.02, max_r=0.6):
        # r is in axis fraction units (0..1)
        self.x = x
        self.y = y
        self.r = float(r0)
        self.max_r = float(max_r)
        self.alpha = 0.95
        self.color = color
        # random ellipse parameters
        self.aspect_ratio = np.random.uniform(0.6, 1.0)  # ratio of height to width (1.0 = circle)
        self.angle = np.random.uniform(0, 360)  # rotation angle in degrees

    def step(self):
        # grow and fade slowly so ripples remain visible (3 times slower)
        self.r += 0.0027
        self.alpha -= 0.002
        return (self.alpha > 0.02) and (self.r < self.max_r)


def main():
    parser = argparse.ArgumentParser(description='Rain ripple animation (interactive or save mode)')
    parser.add_argument('--save', action='store_true', help='Render and save the animation to file (non-interactive)')
    args = parser.parse_args()

    data = load_data(CSV_CANDIDATE)
    n = len(data['rain'])

    fig, ax = plt.subplots(figsize=(10, 7))
    # load the Kyoto map image as the background (preferred)
    try:
        kyoto_img_path = os.path.join(os.path.dirname(__file__), 'Kyoto_map.jpg')
        if os.path.exists(kyoto_img_path):
            img = plt.imread(kyoto_img_path)
            # draw the image to fill the axes (axes coordinates 0..1), slightly dimmed so ripples show
            ax.imshow(img, extent=(0, 1, 0, 1), aspect='auto', interpolation='bilinear', zorder=0, alpha=0.75)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            # if the image is missing, use a black background (do not use the previous blue)
            ax.set_facecolor('#000000')
    except Exception:
        ax.set_facecolor('#000000')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    ripples = []
    artists = []

    info_text = ax.text(0.98, 0.02, '', ha='right', va='bottom', color='white', fontsize=10, transform=ax.transAxes,
                        bbox=dict(facecolor=(0,0,0,0.45), edgecolor='none', boxstyle='round'), zorder=4)

    # limit ripple area to ~1/220 of canvas area (about 10 times smaller than original)
    # circle area = pi * r^2 ; solve for r where area = (1/220) * canvas_area (canvas area = 1 for axes coords)
    max_area_frac = 1.0 / 220.0
    max_r = float(np.sqrt(max_area_frac / math.pi))  # in axis fraction units

    # use random positioning instead of grid-based positioning
    MARGIN = 0.06

    def spawn(idx):
        # use completely random positioning across the canvas
        rain = float(data['rain'][idx % n])
        wind = float(data['wind_dir'][idx % n])
        ang = math.radians(wind)

        # generate random position within margins
        x = MARGIN + np.random.rand() * (1.0 - 2 * MARGIN)
        y = MARGIN + np.random.rand() * (1.0 - 2 * MARGIN)

        # gentle wind nudge so ripples feel like they're affected by wind
        offx = 0.02 * math.cos(ang)
        offy = 0.02 * math.sin(ang)
        x += offx
        y += offy

        # clamp to axes bounds
        x = float(np.clip(x, 0.02, 0.98))
        y = float(np.clip(y, 0.02, 0.98))

        col = color_from_rain(rain)
        # use the computed max_r so ripples never exceed ~1/220 canvas area
        r = Ripple(x, y, col, r0=0.02, max_r=max_r)
        # ensure RGBA tuple; use matplotlib helper
        from matplotlib.colors import to_rgba
        rgba = to_rgba(r.color, r.alpha)
        # create two concentric ellipses with random shapes and very thin lines
        width1, height1 = r.r * 2, r.r * 2 * r.aspect_ratio
        width2, height2 = r.r * 2 * 0.7, r.r * 2 * 0.7 * r.aspect_ratio
        c1 = Ellipse((r.x, r.y), width1, height1, angle=r.angle, fill=False, edgecolor=rgba, linewidth=0.625, zorder=2)
        c2 = Ellipse((r.x, r.y), width2, height2, angle=r.angle, fill=False, edgecolor=rgba, linewidth=0.625, zorder=2)
        ax.add_patch(c1)
        ax.add_patch(c2)
        ripples.append(r)
        artists.append([c1, c2])

    frame_idx = {'i': 0}

    def update(frame):
        # spawn a few ripples
        if frame % 5 == 0:
            for _ in range(6):
                spawn(frame_idx['i'])
                frame_idx['i'] += 1

        # step ripples
        to_remove = []
        for rp, art_item in zip(ripples[:], artists[:]):
            alive = rp.step()
            from matplotlib.colors import to_rgba
            rgba = to_rgba(rp.color, max(0.0, rp.alpha))
            # check if art_item is a list (pair of ellipses) or single ellipse
            if isinstance(art_item, list):
                # update both concentric ellipses
                width1, height1 = rp.r * 2, rp.r * 2 * rp.aspect_ratio
                width2, height2 = rp.r * 2 * 0.7, rp.r * 2 * 0.7 * rp.aspect_ratio
                art_item[0].set_width(width1)
                art_item[0].set_height(height1)
                art_item[0].set_edgecolor(rgba)
                art_item[1].set_width(width2)
                art_item[1].set_height(height2)
                art_item[1].set_edgecolor(rgba)
            else:
                # handle legacy single ellipse (shouldn't happen with new code)
                width, height = rp.r * 2, rp.r * 2 * rp.aspect_ratio
                art_item.set_width(width)
                art_item.set_height(height)
                art_item.set_edgecolor(rgba)
            if not alive:
                to_remove.append((rp, art_item))

        for rp, art_item in to_remove:
            try:
                ripples.remove(rp)
                artists.remove(art_item)
                if isinstance(art_item, list):
                    art_item[0].remove()
                    art_item[1].remove()
                else:
                    art_item.remove()
            except ValueError:
                pass

        # update overlay
        idx = frame_idx['i'] % n
        info_text.set_text(f'Kyoto: 35.0116, 135.7681\nRain: {data["rain"][idx]:.2f} mm\nRH: {data["rh"][idx]:.1f}%\nTemp: {data["temp"][idx]:.1f} °C')
        # flatten artist pairs for return, handling both single circles and pairs
        all_artists = []
        for item in artists:
            if isinstance(item, list):
                all_artists.extend(item)
            else:
                all_artists.append(item)
        return all_artists + [info_text]

    anim = FuncAnimation(fig, update, frames=2000, interval=50, blit=False)

    if args.save:
        # Try to save as mp4 using ffmpeg; if that fails, fall back to exporting frames
        out_mp4 = os.path.join(os.path.dirname(__file__), 'main_animation.mp4')
        try:
            print('Attempting to save animation to', out_mp4)
            anim.save(out_mp4, fps=20, dpi=150, codec='h264', writer='ffmpeg')
            print('Saved animation to', out_mp4)
        except Exception as e:
            print('MP4 save failed:', e)
            print('Falling back to exporting frames using export_frames.py')
            try:
                # ensure script directory is on sys.path and import by module name
                import sys
                script_dir = os.path.dirname(__file__)
                if script_dir not in sys.path:
                    sys.path.insert(0, script_dir)
                try:
                    from export_frames import save_frames
                    save_frames()
                except Exception:
                    # final fallback: execute the file directly
                    import runpy
                    runpy.run_path(os.path.join(script_dir, 'export_frames.py'), run_name='__main__')
                print('Frames exported to frames/ directory')
            except Exception as e2:
                print('Fallback frame export failed:', e2)
        return

    # interactive/show mode
    plt.show()


if __name__ == '__main__':
    main()

