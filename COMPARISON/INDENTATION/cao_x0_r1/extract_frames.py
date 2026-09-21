import sys

targets = {20000, 120000, 220000, 320000}
fname = "indent.lammpstrj"

state = "seek_timestep"
cur_step = None
natoms = None
box = {}
box_lines_read = 0
atoms_read = 0
out = None
ymid = None

with open(fname) as f:
    for line in f:
        line = line.rstrip("\n")
        if line == "ITEM: TIMESTEP":
            state = "read_timestep"
            continue
        if state == "read_timestep":
            cur_step = int(line)
            state = "seek_natoms"
            continue
        if line == "ITEM: NUMBER OF ATOMS":
            state = "read_natoms"
            continue
        if state == "read_natoms":
            natoms = int(line)
            state = "seek_box"
            box_lines_read = 0
            box = {}
            continue
        if line.startswith("ITEM: BOX BOUNDS"):
            state = "read_box"
            box_lines_read = 0
            continue
        if state == "read_box":
            lo, hi = line.split()[:2]
            box[box_lines_read] = (float(lo), float(hi))
            box_lines_read += 1
            if box_lines_read == 3:
                ymid = 0.5*(box[1][0]+box[1][1])
                state = "seek_atoms_header"
            continue
        if line.startswith("ITEM: ATOMS"):
            state = "read_atoms" if cur_step in targets else "skip_atoms"
            atoms_read = 0
            if cur_step in targets:
                out = open(f"frame_{cur_step}.txt", "w")
                out.write(f"# step={cur_step} xlo={box[0][0]} xhi={box[0][1]} ylo={box[1][0]} yhi={box[1][1]} zlo={box[2][0]} zhi={box[2][1]}\n")
            continue
        if state in ("read_atoms", "skip_atoms"):
            atoms_read += 1
            if state == "read_atoms":
                p = line.split()
                aid, atype, x, y, z = p[0], p[1], float(p[2]), float(p[3]), float(p[4])
                if abs(y - ymid) < 3.5:
                    out.write(f"{aid} {atype} {x:.3f} {z:.3f}\n")
            if atoms_read == natoms:
                if state == "read_atoms":
                    out.close()
                    print(f"frame {cur_step} escrito")
                state = "seek_timestep"

print("listo")
