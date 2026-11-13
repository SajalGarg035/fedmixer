"""Inspect yearly GSOD archives and extract selected station CSVs."""

import argparse
import csv
import os
import re
import tarfile
import time
from collections import defaultdict

from .stations import read_station_ids


STATION_CSV = re.compile(r'^(\d{11})\.csv$', re.IGNORECASE)
CHUNK_SIZE = 1024 * 1024


def archive_path(archive_dir, year):
    return os.path.join(archive_dir, '{}.tar.gz'.format(year))


def list_station_members(path):
    try:
        with tarfile.open(path, mode='r:gz') as archive:
            return sorted({
                match.group(1)
                for member in archive.getmembers()
                if member.isfile()
                for match in [STATION_CSV.fullmatch(os.path.basename(member.name))]
                if match
            })
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError('could not read archive {}: {}'.format(path, exc))


def write_inventory(archive_dir, years, output):
    station_years = defaultdict(list)
    for index, year in enumerate(years, 1):
        path = archive_path(archive_dir, year)
        if not os.path.isfile(path):
            print('[{}/{}] missing archive: {}'.format(index, len(years), path))
            continue
        stations = list_station_members(path)
        print('[{}/{}] {}: {:,} station CSVs'.format(
            index, len(years), year, len(stations)))
        for station_id in stations:
            station_years[station_id].append(year)

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['station_id', 'years_available', 'year_count'])
        for station_id in sorted(station_years):
            available = sorted(station_years[station_id])
            writer.writerow([station_id, ','.join(map(str, available)), len(available)])
    print('saved station inventory to {}'.format(output))


def extract_year(path, year, raw_dir, selected=None):
    destination_dir = os.path.join(raw_dir, str(year))
    os.makedirs(destination_dir, exist_ok=True)
    wanted = set(selected) if selected is not None else None
    totals = {'extracted': 0, 'existing': 0, 'unselected': 0}
    started = time.monotonic()
    try:
        with tarfile.open(path, mode='r:gz') as archive:
            members = archive.getmembers()
            candidates = []
            for member in members:
                match = STATION_CSV.fullmatch(os.path.basename(member.name))
                if not member.isfile() or not match:
                    continue
                station_id = match.group(1)
                if wanted is not None and station_id not in wanted:
                    totals['unselected'] += 1
                    continue
                candidates.append((member, station_id))

            todo = [(member, station_id) for member, station_id in candidates
                    if not os.path.isfile(os.path.join(destination_dir, station_id + '.csv'))]
            totals['existing'] = len(candidates) - len(todo)
            if not todo:
                print('{}: no new station files ({} already present)'.format(
                    year, totals['existing']))
                return totals

            for index, (member, station_id) in enumerate(todo, 1):
                output_path = os.path.join(destination_dir, station_id + '.csv')
                temporary = output_path + '.part'
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError('could not read {} in {}'.format(member.name, path))
                try:
                    with source, open(temporary, 'wb') as output:
                        while True:
                            chunk = source.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            output.write(chunk)
                    os.replace(temporary, output_path)
                except OSError:
                    if os.path.exists(temporary):
                        os.remove(temporary)
                    raise
                totals['extracted'] += 1
                if index == 1 or index % 100 == 0 or index == len(todo):
                    elapsed = max(time.monotonic() - started, 0.001)
                    rate = index / elapsed
                    eta = (len(todo) - index) / rate if rate else 0
                    print('\r{}: {}/{} station files | elapsed {:.0f}s | ETA {:.0f}s'.format(
                        year, index, len(todo), elapsed, eta), end='', flush=True)
            print()
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError('could not extract archive {}: {}'.format(path, exc))
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    inventory = subparsers.add_parser('inventory', help='list station IDs in local archives')
    inventory.add_argument('--archive-dir', required=True)
    inventory.add_argument('--start-year', type=int, required=True)
    inventory.add_argument('--end-year', type=int, required=True)
    inventory.add_argument('--output', default='./data/global_surface/station_inventory.csv')

    extract = subparsers.add_parser('extract', help='extract station CSVs into pipeline raw layout')
    extract.add_argument('--archive-dir', required=True)
    extract.add_argument('--start-year', type=int, required=True)
    extract.add_argument('--end-year', type=int, required=True)
    extract.add_argument('--stations-file')
    extract.add_argument('--stations', nargs='*')
    extract.add_argument('--raw-dir', default='./data/global_surface/raw')

    args = parser.parse_args()
    if args.end_year < args.start_year:
        parser.error('--end-year must be >= --start-year')
    years = range(args.start_year, args.end_year + 1)
    if args.command == 'inventory':
        write_inventory(args.archive_dir, years, args.output)
        return
    try:
        selected = None
        if args.stations_file or args.stations:
            selected = read_station_ids(args.stations_file, args.stations)
        totals = {'extracted': 0, 'existing': 0, 'unselected': 0, 'missing_archives': 0}
        for index, year in enumerate(years, 1):
            path = archive_path(args.archive_dir, year)
            if not os.path.isfile(path):
                print('[{}/{}] missing archive: {}'.format(index, len(years), path))
                totals['missing_archives'] += 1
                continue
            print('[{}/{}] processing {}'.format(index, len(years), path))
            result = extract_year(path, year, args.raw_dir, selected)
            for key, value in result.items():
                totals[key] += value
        print('archive extraction summary: {}'.format(totals))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()

