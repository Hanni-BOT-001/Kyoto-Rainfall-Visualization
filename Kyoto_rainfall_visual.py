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
        # create three concentric circles: blue halo + two main ripples
        # bright blue halo (outermost)
        halo_color = (0.3, 0.7, 1.0, r.alpha * 0.6)  # bright blue with reduced alpha
        c_halo = Circle((r.x, r.y), r.r * 1.3, fill=False, edgecolor=halo_color, linewidth=1.0, zorder=2)
        # main ripples with original colors
        c1 = Circle((r.x, r.y), r.r, fill=False, edgecolor=rgba, linewidth=0.625, zorder=2)
        c2 = Circle((r.x, r.y), r.r * 0.7, fill=False, edgecolor=rgba, linewidth=0.625, zorder=2)
        ax.add_patch(c_halo)
        ax.add_patch(c1)
        ax.add_patch(c2)
        ripples.append(r)
        artists.append([c_halo, c1, c2]) directory and uses
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
from matplotlib.patches import Circle
import argparse


CSV_CANDIDATE = os.path.join(os.path.dirname(__file__), 'kyotov9.24.csv')


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
    # map rainfall 0..30mm to brighter Morandi color palette for better visibility against dark clouds
    v = max(0.0, min(1.0, r / 30.0))
    low = np.array([0.85, 0.95, 0.88])   # brighter sage green
    mid = np.array([0.8, 0.88, 0.98])    # brighter blue-gray
    high = np.array([0.98, 0.88, 0.78])  # brighter warm beige
    if v < 0.5:
        t = v / 0.5
        col = low * (1 - t) + mid * t
    else:
        t = (v - 0.5) / 0.5
        col = mid * (1 - t) + high * t
    return tuple(col.tolist())


class Cloud:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = np.random.uniform(0.15, 0.3)
        self.height = np.random.uniform(0.08, 0.2)
        self.alpha = 0.0
        self.target_alpha = np.random.uniform(0.4, 0.6)
        self.fade_in = True
        self.lifetime = np.random.randint(300, 500)
        self.age = 0

    def step(self):
        self.age += 1
        
        # Fade in phase
        if self.fade_in:
            self.alpha += 0.005
            if self.alpha >= self.target_alpha:
                self.alpha = self.target_alpha
                self.fade_in = False
        
        # Fade out phase (start fading when 70% through lifetime)
        elif self.age > self.lifetime * 0.7:
            self.alpha -= 0.003
        
        return self.alpha > 0.01 and self.age < self.lifetime


class Ripple:
    def __init__(self, x, y, color, r0=0.02, max_r=0.6):
        # r is in axis fraction units (0..1)
        self.x = x
        self.y = y
        self.r = float(r0)
        self.max_r = float(max_r)
        self.alpha = 0.95
        self.color = color

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
    clouds = []
    cloud_artists = []

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
        # create three concentric circles: blue halo + two main ripples
        # bright blue halo (outermost)
        halo_color = (0.3, 0.7, 1.0, r.alpha * 0.6)  # bright blue with reduced alpha
        c_halo = Circle((r.x, r.y), r.r * 1.3, fill=False, edgecolor=halo_color, linewidth=1.0, zorder=2)
        # main ripples with original colors
        c1 = Circle((r.x, r.y), r.r, fill=False, edgecolor=rgba, linewidth=0.625, zorder=2)
        c2 = Circle((r.x, r.y), r.r * 0.7, fill=False, edgecolor=rgba, linewidth=0.625, zorder=2)
        ax.add_patch(c_halo)
        ax.add_patch(c1)
        ax.add_patch(c2)
        ripples.append(r)
        artists.append([c_halo, c1, c2])

    def spawn_cloud():
        # spawn blocky clouds in random locations
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        cloud = Cloud(x, y)
        
        # create blocky cloud patch with blurred edges using multiple overlapping rectangles
        from matplotlib.patches import Rectangle
        cloud_patches = []
        
        # Main cloud body (darker center)
        main_patch = Rectangle((cloud.x - cloud.width/2, cloud.y - cloud.height/2), 
                              cloud.width, cloud.height,
                              fill=True, facecolor='#1a1a1a', alpha=cloud.alpha, 
                              edgecolor='none', zorder=1)
        ax.add_patch(main_patch)
        cloud_patches.append(main_patch)
        
        # Blur effect using multiple lighter, larger rectangles
        for i in range(3):
            blur_scale = 1.2 + i * 0.3  # Progressively larger
            blur_alpha = cloud.alpha * (0.3 - i * 0.08)  # Progressively lighter
            blur_patch = Rectangle((cloud.x - cloud.width * blur_scale/2, 
                                  cloud.y - cloud.height * blur_scale/2), 
                                  cloud.width * blur_scale, 
                                  cloud.height * blur_scale,
                                  fill=True, facecolor='#2a2a2a', alpha=blur_alpha, 
                                  edgecolor='none', zorder=1)
            ax.add_patch(blur_patch)
            cloud_patches.append(blur_patch)
        
        clouds.append(cloud)
        cloud_artists.append(cloud_patches)

    frame_idx = {'i': 0}

    def update(frame):
        # spawn clouds occasionally
        if frame % 80 == 0 and np.random.rand() < 0.7:  # 70% chance every 80 frames
            spawn_cloud()
        
        # spawn a few ripples
        if frame % 5 == 0:
            for _ in range(6):
                spawn(frame_idx['i'])
                frame_idx['i'] += 1

        # step clouds
        clouds_to_remove = []
        for cloud, cloud_patches in zip(clouds[:], cloud_artists[:]):
            alive = cloud.step()
            # Update alpha for all patches in this cloud
            for i, patch in enumerate(cloud_patches):
                if i == 0:  # Main patch
                    patch.set_alpha(cloud.alpha)
                else:  # Blur patches
                    blur_alpha = cloud.alpha * (0.3 - (i-1) * 0.08)
                    patch.set_alpha(max(0, blur_alpha))
            if not alive:
                clouds_to_remove.append((cloud, cloud_patches))
        
        for cloud, cloud_patches in clouds_to_remove:
            try:
                clouds.remove(cloud)
                cloud_artists.remove(cloud_patches)
                for patch in cloud_patches:
                    patch.remove()
            except ValueError:
                pass

        # step ripples
        to_remove = []
        for rp, art_item in zip(ripples[:], artists[:]):
            alive = rp.step()
            from matplotlib.colors import to_rgba
            rgba = to_rgba(rp.color, max(0.0, rp.alpha))
            # check if art_item is a list (trio of circles with halo) or single circle
            if isinstance(art_item, list):
                if len(art_item) == 3:  # New format with halo
                    # update blue halo (outermost)
                    halo_color = (0.3, 0.7, 1.0, rp.alpha * 0.6)
                    art_item[0].set_radius(rp.r * 1.3)
                    art_item[0].set_edgecolor(halo_color)
                    # update main concentric circles
                    art_item[1].set_radius(rp.r)
                    art_item[1].set_edgecolor(rgba)
                    art_item[2].set_radius(rp.r * 0.7)
                    art_item[2].set_edgecolor(rgba)
                elif len(art_item) == 2:  # Legacy format with two circles
                    art_item[0].set_radius(rp.r)
                    art_item[0].set_edgecolor(rgba)
                    art_item[1].set_radius(rp.r * 0.7)
                    art_item[1].set_edgecolor(rgba)
            else:
                # handle legacy single circle (shouldn't happen with new code)
                art_item.set_radius(rp.r)
                art_item.set_edgecolor(rgba)
            if not alive:
                to_remove.append((rp, art_item))

        for rp, art_item in to_remove:
            try:
                ripples.remove(rp)
                artists.remove(art_item)
                if isinstance(art_item, list):
                    for circle in art_item:
                        circle.remove()
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
        
        # flatten cloud artists (each cloud has multiple patches)
        all_cloud_artists = []
        for cloud_patches in cloud_artists:
            all_cloud_artists.extend(cloud_patches)
        
        return all_artists + all_cloud_artists + [info_text]

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
            print('Animation could not be saved. Make sure ffmpeg is installed for MP4 output.')
            print('You can still run the script without --save to view the interactive animation.')
        return

    # interactive/show mode
    plt.show()


if __name__ == '__main__':
    main()

