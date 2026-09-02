"""
extract_modulus.py — STAGE9_STRESS_STRAIN_300K
Extrae el módulo de Young E de las curvas tensión-deformación uniaxiales.

Protocolo:
  - Ajuste lineal de σ vs ε en el rango [0, strain_elastic] (régimen elástico)
  - E = pendiente del ajuste (GPa)
  - Promedio sobre las 3 direcciones (x, y, z) y las 3 réplicas

Archivos de entrada:
  results/ss_x_{tag}.dat, ss_y_{tag}.dat, ss_z_{tag}.dat
  Columnas: strain_main  strain_t1  strain_t2  poisson_t1  poisson_t2
            stress_x_GPa  stress_y_GPa  stress_z_GPa

Uso:
  python extract_modulus.py --glass_type am   [AM as-melted]
  python extract_modulus.py --glass_type iex1
  python extract_modulus.py --glass_type iex2
  python extract_modulus.py --glass_type all   [resume todos]
  python extract_modulus.py --strain_max 0.02  [rango elástico, default 0.02]
"""

import numpy as np
import argparse
from pathlib import Path
import warnings

HERE = Path(__file__).parent

# Composition lists
GLASS_CONFIGS = {
    'am':   {'label': 'As-melted',  'x_vals': [0,1,3,6,9,12,15], 'x_key': 'x'},
    'iex1': {'label': 'IEX Proto1', 'x_vals': [1,3,6,9,12,15],   'x_key': 'xt'},
    'iex2': {'label': 'IEX Proto2', 'x_vals': [0,1,3,6,9,12],    'x_key': 'xp'},
}

# Seeds por replica y dirección (mismos que en los inputs LAMMPS)
SEEDS = {
    'x': {1: 42345, 2: 22345, 3: 33345},
    'y': {1: 32345, 2: 52345, 3: 72345},
    'z': {1: 12345, 2: 22345, 3: 32345},
}


def build_tag(glass_type, x, r):
    cfg = GLASS_CONFIGS[glass_type]
    k   = cfg['x_key']
    if glass_type == 'am':
        return f"AM300K_x{x}_r{r}"
    elif glass_type == 'iex1':
        return f"IEX1_300K_xt{x}_r{r}"
    else:
        return f"IEX2_300K_xp{x}_r{r}"


def extract_E_from_file(dat_file, strain_max=0.02):
    """
    Ajuste lineal σ vs ε en [0, strain_max].
    Columna 0: strain_main, columna correspondiente: stress_main_GPa.

    Para x: stress_main = col 5 (stress_x)
    Para y: stress_main = col 6 (stress_y)
    Para z: stress_main = col 7 (stress_z) — pero necesitamos la dir correcta.

    En la práctica, col 5 = stress_x, col 6 = stress_y, col 7 = stress_z,
    y col 0 = strain_main. Para x → stress=col5, y → col6, z → col7.
    """
    try:
        data = np.loadtxt(dat_file, skiprows=1)
    except Exception as e:
        return None, f"Error leyendo {dat_file}: {e}"

    if data.ndim < 2 or data.shape[0] < 5:
        return None, f"Datos insuficientes en {dat_file}"

    strain = data[:, 0]
    # Determinar columna de stress según dirección en el nombre del archivo
    name = Path(dat_file).name
    if '_x_' in name:
        stress = data[:, 5]   # stress_x_GPa
    elif '_y_' in name:
        stress = data[:, 6]   # stress_y_GPa
    elif '_z_' in name:
        stress = data[:, 7]   # stress_z_GPa
    else:
        stress = data[:, 5]   # default

    mask = (strain >= 0.0) & (strain <= strain_max)
    if mask.sum() < 3:
        return None, f"Menos de 3 puntos en [0, {strain_max}] en {dat_file}"

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        slope, intercept = np.polyfit(strain[mask], stress[mask], 1)

    return slope, None   # E in GPa


def process_glass(glass_type, strain_max=0.02, verbose=False):
    cfg    = GLASS_CONFIGS[glass_type]
    x_vals = cfg['x_vals']
    label  = cfg['label']

    print(f"\n{'='*60}")
    print(f"{label} — módulo de Young a 300 K (erate=1e9 s⁻¹, strain_max={strain_max})")
    print(f"{'='*60}")
    print(f"{'x':>4}  {'E_x':>7}  {'E_y':>7}  {'E_z':>7}  {'E_mean':>8}  {'E_std':>7}  "
          f"[réplicas × 3 dirs = 9 valores]")
    print(f"{'':>4}  {'(GPa)':>7}  {'(GPa)':>7}  {'(GPa)':>7}  {'(GPa)':>8}  {'(GPa)':>7}")

    summary = []
    for x in x_vals:
        E_all = []
        E_by_dir = {d: [] for d in ['x','y','z']}
        for r in [1, 2, 3]:
            tag = build_tag(glass_type, x, r)
            for d in ['x', 'y', 'z']:
                f = HERE / 'results' / f'ss_{d}_{tag}.dat'
                if not f.exists():
                    if verbose:
                        print(f"  MISSING: {f}")
                    continue
                E, err = extract_E_from_file(f, strain_max=strain_max)
                if E is None:
                    if verbose:
                        print(f"  ERROR {f}: {err}")
                    continue
                E_all.append(E)
                E_by_dir[d].append(E)

        Ex = np.mean(E_by_dir['x']) if E_by_dir['x'] else np.nan
        Ey = np.mean(E_by_dir['y']) if E_by_dir['y'] else np.nan
        Ez = np.mean(E_by_dir['z']) if E_by_dir['z'] else np.nan
        Em = np.mean(E_all)         if E_all           else np.nan
        Es = np.std(E_all, ddof=1)  if len(E_all) > 1  else 0.0

        print(f"  {x:>2d}   {Ex:>7.2f}  {Ey:>7.2f}  {Ez:>7.2f}  {Em:>8.2f}  {Es:>7.2f}")
        summary.append({'x': x, 'E_x': Ex, 'E_y': Ey, 'E_z': Ez,
                        'E_mean': Em, 'E_std': Es, 'n': len(E_all)})

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--glass_type', type=str, default='all',
                        choices=['am', 'iex1', 'iex2', 'all'])
    parser.add_argument('--strain_max', type=float, default=0.02,
                        help='Límite superior de strain para ajuste lineal (default 0.02)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    types = ['am', 'iex1', 'iex2'] if args.glass_type == 'all' else [args.glass_type]
    for gt in types:
        process_glass(gt, strain_max=args.strain_max, verbose=args.verbose)


if __name__ == '__main__':
    main()
