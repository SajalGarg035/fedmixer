"""Inspect which NOAA GSOD station files exist for selected years."""

import argparse
import csv
import os
import re
from collections import defaultdict
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


BASE_URL = 'https://www.ncei.noaa.gov/data/global-summary-of-the-day/access'


class CsvLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.station_ids = set()

    def handle_starttag(self, tag, attrs):
        if tag.lower() != 'a':
            return
        href = dict(attrs).get('href', '')
        filename = os.path.basename(urlparse(href).path)
        match = re.fullmatch(r'(\d{11})\.csv', filename, re.IGNORECASE)
        if match:
            self.station_ids.add(match.group(1))


def stations_for_year(year):
    url = '{}/{}/'.format(BASE_URL, year)
    request = Request(url, headers={'User-Agent': 'TimeMixer-GSOD-pipeline/1.0'})
    try:
        with urlopen(request, timeout=60) as response:
            page = response.read().decode('utf-8', errors='replace')
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError('could not inspect NOAA GSOD year {}: {}'.format(year, exc))
    parser = CsvLinks()
    parser.feed(page)
    return parser.station_ids


def inspect_years(years, output):
    station_years = defaultdict(list)
    for year in years:
        ids = stations_for_year(year)
        print('{}: {} station files'.format(year, len(ids)))
        for station_id in ids:
            station_years[station_id].append(year)

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['station_id', 'years_available', 'year_count'])
        for station_id in sorted(station_years):
            available = sorted(station_years[station_id])
            writer.writerow([
                station_id,
                ','.join(str(year) for year in available),
                len(available),
            ])
    print('saved station inventory to {}'.format(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-year', type=int, required=True)
    parser.add_argument('--end-year', type=int, required=True)
    parser.add_argument('--output', default='./data/global_surface/station_inventory.csv')
    args = parser.parse_args()
    if args.end_year < args.start_year:
        parser.error('--end-year must be >= --start-year')
    inspect_years(range(args.start_year, args.end_year + 1), args.output)


if __name__ == '__main__':
    main()

