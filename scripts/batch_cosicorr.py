#!/usr/bin/env python3
"""
Batch runner for cosicorr correlate.
Pairs 4-band images (base) with 8-band images (target) in a folder.
4-band files: use --base_band 4
8-band files: use --target_band 7

Usage:
  python scripts/batch_cosicorr.py --dir tests/Cropped_data/3-Lafayette_1

Options can be passed through to the correlate command via --extra "--window_size 128 128 128 128 --step 10 10 ..."
"""
import argparse
import subprocess
import os
from pathlib import Path
import rasterio


def find_tifs(folder):
    return sorted([p for p in Path(folder).glob('**/*.tif')])


def bands_count(path):
    try:
        with rasterio.open(path) as src:
            return src.count
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', '-d', required=True, help='directory with .tif files')
    p.add_argument('--method', default='wavelet_ncc')
    p.add_argument('--window_size', default='128 128 128 128')
    p.add_argument('--step', default='10 10')
    p.add_argument('--min_peak', default='0.1')
    p.add_argument('--levels', default='3')
    p.add_argument('--wavelet', default="'db2'")
    p.add_argument('--vmin', default='-2')
    p.add_argument('--vmax', default='2')
    p.add_argument('--show', action='store_true')
    p.add_argument('--time_pairs', action='store_true', help='Run correlations for ordered time pairs (base earlier -> target later)')
    p.add_argument('--order_by', choices=['mtime', 'name', 'name_date'], default='mtime', help='Sort key for time_pairs')
    p.add_argument('--extra', default='', help='Extra raw options to pass to correlate')
    args = p.parse_args()

    folder = Path(args.dir)
    tifs = find_tifs(folder)
    if not tifs:
        print('No tif found in', folder)
        return

    # group by band count
    tifs_by_bands = {}
    for t in tifs:
        bc = bands_count(t)
        if bc is None:
            continue
        tifs_by_bands.setdefault(bc, []).append(t)

    four_band = tifs_by_bands.get(4, [])
    eight_band = tifs_by_bands.get(8, [])

    if not four_band:
        print('No 4-band files found')
    if not eight_band:
        print('No 8-band files found')

    # create a results folder inside the input directory to store outputs
    results_folder = folder / 'results_0.05minpeak_NIR'
    results_folder.mkdir(parents=True, exist_ok=True)
    outdir = str(results_folder)
    print(f'Outputs will be written to: {outdir}')

    def pick_band(path_obj):
        name = os.path.basename(str(path_obj)).lower()
        return '7' if '8band' in name else '4'

    def merge_extra_args(cmd, extra_str, outdir):
        """Append extra args but strip any user-provided --output_path so outputs go to our results folder.

        Handles both `--output_path value` and `--output_path=value` styles.
        """
        if not extra_str:
            # ensure our output_path is present
            return cmd + ['--output_path', outdir]
        tokens = extra_str.split()
        filtered = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == '--output_path':
                # skip this and the next token (value)
                i += 2
                continue
            if t.startswith('--output_path='):
                i += 1
                continue
            filtered.append(t)
            i += 1
        # ensure our output path is appended (override any user-provided one)
        filtered += ['--output_path', outdir]
        return cmd + filtered

    if args.time_pairs:
        # sort all tifs by chosen key
        if args.order_by == 'mtime':
            tifs_sorted = sorted(tifs, key=lambda p: p.stat().st_mtime)
        elif args.order_by == 'name':
            tifs_sorted = sorted(tifs, key=lambda p: p.name)
        else:
            # try to parse date from filename like YYMMMDDHHMMSS (e.g., 21JUN07185245 where 21 is year)
            import re
            from datetime import datetime

            mon_map = {m.lower(): i for i, m in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], start=1)}

            def parse_dt_from_name(p):
                name = p.name
                # pattern: YYMMMDDHHMMSS (year, month, day, time) e.g. 21JUN07185245
                m = re.search(r'(\d{2})([A-Za-z]{3})(\d{2})(\d{6})', name)
                if not m:
                    # fallback: try pattern with 4-digit year YYYYMMMDDHHMMSS
                    m2 = re.search(r'(\d{4})([A-Za-z]{3})(\d{2})(\d{6})', name)
                    if not m2:
                        # fallback: try simpler pattern YYMMMDD
                        m3 = re.search(r'(\d{2})([A-Za-z]{3})(\d{2})', name)
                        if not m3:
                            return p.stat().st_mtime
                        year_s, mon_s, day = m3.group(1), m3.group(2), m3.group(3)
                        hhmmss = '000000'
                        year = int(year_s)
                        if year < 100:
                            year = 2000 + year if year <= 25 else 1900 + year
                    else:
                        year_s, mon_s, day, hhmmss = m2.group(1), m2.group(2), m2.group(3), m2.group(4)
                        year = int(year_s)
                else:
                    year_s, mon_s, day, hhmmss = m.group(1), m.group(2), m.group(3), m.group(4)
                    year = int(year_s)
                    if year < 100:
                        year = 2000 + year if year <= 25 else 1900 + year

                try:
                    mon = mon_map.get(mon_s.lower(), None)
                    if mon is None:
                        return p.stat().st_mtime
                    hh = int(hhmmss[0:2])
                    mi = int(hhmmss[2:4])
                    ss = int(hhmmss[4:6])
                    dt = datetime(year, mon, int(day), hh, mi, ss)
                    return dt.timestamp()
                except Exception:
                    return p.stat().st_mtime

            tifs_sorted = sorted(tifs, key=parse_dt_from_name)

        # run all ordered pairs base->target where target comes after base
        for i in range(len(tifs_sorted)):
            for j in range(i+1, len(tifs_sorted)):
                base = tifs_sorted[i]
                target = tifs_sorted[j]
                base_band = pick_band(base)
                target_band = pick_band(target)
                cmd = [
                    'python3', 'scripts/cosicorr.py', 'correlate', str(base), str(target),
                    '--method', args.method,
                    '--window_size'] + args.window_size.split() + [
                    '--step'] + args.step.split() + [
                    f'--min_peak={args.min_peak}',
                    '--base_band', base_band,
                    '--target_band', target_band,
                    '--levels', args.levels,
                    '--wavelet', args.wavelet,
                    '--output_path', outdir,
                    '--vmin', args.vmin,
                    '--vmax', args.vmax,
                ]
                # ensure correlate writes the PNG preview (correlate saves a .png when --show is provided)
                if '--show' not in cmd:
                    cmd.append('--show')
                # merge extra args but force output path to our results folder
                cmd = merge_extra_args(cmd, args.extra, outdir)
                print('\nRunning:')
                print(' '.join(cmd))
                subprocess.run(cmd, check=False)
    else:
        # fallback: pair every 4-band base with every 8-band target by default
        for base in four_band:
            for target in eight_band:
                base_band = pick_band(base)
                target_band = pick_band(target)
                cmd = [
                    'python3', 'scripts/cosicorr.py', 'correlate', str(base), str(target),
                    '--method', args.method,
                    '--window_size'] + args.window_size.split() + [
                    '--step'] + args.step.split() + [
                    f'--min_peak={args.min_peak}',
                    '--base_band', base_band,
                    '--target_band', target_band,
                    '--levels', args.levels,
                    '--wavelet', args.wavelet,
                    '--output_path', outdir,
                    '--vmin', args.vmin,
                    '--vmax', args.vmax,
                ]
                # ensure correlate writes the PNG preview (correlate saves a .png when --show is provided)
                if '--show' not in cmd:
                    cmd.append('--show')
                # merge extra args but force output path to our results folder
                cmd = merge_extra_args(cmd, args.extra, outdir)
                print('\nRunning:')
                print(' '.join(cmd))
                subprocess.run(cmd, check=False)


if __name__ == '__main__':
    main()
