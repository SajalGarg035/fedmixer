"""Download selected NOAA GSOD station/year CSV files."""

import argparse
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .stations import read_station_ids


BASE_URL = 'https://www.ncei.noaa.gov/data/global-summary-of-the-day/access'


def download_one(station_id, year, raw_dir, overwrite=False):
    destination = os.path.join(raw_dir, str(year), station_id + '.csv')
    if os.path.isfile(destination) and not overwrite:
        print('already downloaded: {}'.format(destination))
        return 'existing'

    url = '{}/{}/{}.csv'.format(BASE_URL, year, station_id)
    request = Request(url, headers={'User-Agent': 'TimeMixer-GSOD-pipeline/1.0'})
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = destination + '.part'
    try:
        with urlopen(request, timeout=120) as response, open(temporary, 'wb') as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if os.path.getsize(temporary) == 0:
            os.remove(temporary)
            raise RuntimeError('NOAA returned an empty file for {}'.format(url))
        os.replace(temporary, destination)
        print('downloaded {}'.format(destination))
        return 'downloaded'
    except HTTPError as exc:
        if os.path.exists(temporary):
            os.remove(temporary)
        if exc.code == 404:
            print('not available: station {} in {}'.format(station_id, year))
            return 'unavailable'
        raise
    except (URLError, TimeoutError, OSError):
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def download(station_ids, years, raw_dir, overwrite=False):
    totals = {'downloaded': 0, 'existing': 0, 'unavailable': 0}
    count = len(station_ids) * len(years)
    for index, (station_id, year) in enumerate(
            ((station_id, year) for year in years for station_id in station_ids), 1):
        print('[{}/{}] station={} year={}'.format(index, count, station_id, year))
        result = download_one(station_id, year, raw_dir, overwrite)
        totals[result] += 1
    print('download summary: {}'.format(totals))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stations-file')
    parser.add_argument('--stations', nargs='*')
    parser.add_argument('--start-year', type=int, required=True)
    parser.add_argument('--end-year', type=int, required=True)
    parser.add_argument('--raw-dir', default='./data/global_surface/raw')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.end_year < args.start_year:
        parser.error('--end-year must be >= --start-year')
    try:
        station_ids = read_station_ids(args.stations_file, args.stations)
        download(station_ids, range(args.start_year, args.end_year + 1),
                 args.raw_dir, args.overwrite)
    except (OSError, ValueError, RuntimeError, HTTPError, URLError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()

