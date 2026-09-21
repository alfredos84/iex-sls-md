"""Genera un GIF animado (corte transversal) de la indentacion a partir de
indent.lammpstrj, sin depender de OVITO/VESTA."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

HERE = Path(__file__).parent
TRAJ = HERE / "indent.lammpstrj"

TYPE_COLOR = {1: "#c9b8a0", 2: "#d94f4f", 3: "#4caf6e", 4: "#7a5cc9", 5: "#e08c2a", 6: "#222222"}
TYPE_SIZE  = {1: 3, 2: 2, 3: 4, 4: 4, 5: 4, 6: 6}

SLICE_HALF = 8.0   # A, medio-espesor del corte en y
YMID = 58.1        # A, centro de la caja en y

def read_frames(path):
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(f.readline())
            f.readline()  # ITEM: NUMBER OF ATOMS
            natoms = int(f.readline())
            f.readline()  # ITEM: BOX BOUNDS...
            f.readline(); f.readline(); f.readline()
            f.readline()  # ITEM: ATOMS id type x y z
            types, xs, zs = [], [], []
            for _ in range(natoms):
                parts = f.readline().split()
                y = float(parts[3])
                if abs(y - YMID) > SLICE_HALF:
                    continue
                types.append(int(parts[1]))
                xs.append(float(parts[2]))
                zs.append(float(parts[4]))
            yield step, np.array(types), np.array(xs), np.array(zs)

print("Leyendo frames (esto puede tardar unos minutos)...")
frames = list(read_frames(TRAJ))
print(f"{len(frames)} frames leidos, {frames[0][1].size} atomos en el corte del primer frame")

fig, ax = plt.subplots(figsize=(6, 6))
scat = ax.scatter([], [], s=3)
ax.set_xlim(0, 116)
ax.set_zlim = None
ax.set_ylim(-5, 120)
ax.set_xlabel("x (Å)")
ax.set_ylabel("z (Å)")
title = ax.set_title("")

def update(i):
    step, types, xs, zs = frames[i]
    colors = [TYPE_COLOR.get(t, "#999999") for t in types]
    sizes = [TYPE_SIZE.get(t, 2) for t in types]
    scat.set_offsets(np.column_stack([xs, zs]))
    scat.set_color(colors)
    scat.set_sizes(sizes)
    phase = "carga" if step <= 120000 else ("hold" if step <= 220000 else "descarga")
    title.set_text(f"Indentación CaO x=0 r=1 — step {step} ({phase})")
    return scat, title

ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
out = HERE / "indentation_movie.gif"
ani.save(out, writer=animation.PillowWriter(fps=12))
print(f"Escrito: {out}")
