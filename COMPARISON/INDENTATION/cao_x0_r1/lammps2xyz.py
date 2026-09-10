import sys
from pathlib import Path

TYPE_ELEM = {1: 'Si', 2: 'O', 3: 'Ca', 4: 'Na', 5: 'K', 6: 'C'}

def convert(data_path):
    atoms = []
    in_atoms = False
    with open(data_path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if s.startswith('Atoms'):
                in_atoms = True
                continue
            if in_atoms:
                if s[0].isalpha():
                    break
                parts = s.split()
                # formato: id type charge x y z [ix iy iz]
                t = int(parts[1])
                x, y, z = parts[3], parts[4], parts[5]
                atoms.append((TYPE_ELEM.get(t, 'X'), x, y, z))

    out_path = Path(data_path).with_suffix('.xyz')
    with open(out_path, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"converted from {Path(data_path).name}\n")
        for elem, x, y, z in atoms:
            f.write(f"{elem}  {x}  {y}  {z}\n")
    print(f"Escrito: {out_path}  ({len(atoms)} átomos)")

convert(sys.argv[1] if len(sys.argv) > 1 else "slab2x2_tip.data")
