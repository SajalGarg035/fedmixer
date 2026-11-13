"""Download NOAA's yearly GSOD archive files with progress and resume support."""

import argparse
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = 'https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive'
CHUNK_SIZE = 1024 * 1024


def download_one(year, archive_dir, overwrite=False):
    os.makedirs(archive_dir, exist_ok=True)
    destination = os.path.join(archive_dir, '{}.tar.gz'.format(year))
    partial = destination + '.part'
    if os.path.isfile(destination) and not overwrite:
        print('already downloaded: {}'.format(destination))
        return 'existing'
    if overwrite and os.path.exists(partial):
        os.remove(partial)

    offset = os.path.getsize(partial) if os.path.isfile(partial) else 0
    url = '{}/{}.tar.gz'.format(BASE_URL, year)
    headers = {'User-Agent': 'TimeMixer-GSOD-pipeline/1.0'}
    if offset:
        headers['Range'] = 'bytes={}-'.format(offset)
    request = Request(url, headers=headers)
    started = time.monotonic()
    last_report = started
    received = 0
    try:
        with urlopen(request, timeout=120) as response:
            append = bool(offset and getattr(response, 'status', None) == 206)
            if offset and not append:
                offset = 0
            content_length = response.headers.get('Content-Length')
            total = offset + int(content_length) if content_length else None
            mode = 'ab' if append else 'wb'
            with open(partial, mode) as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    received += len(chunk)
                    now = time.monotonic()
                    if now - last_report >= 2:
                        elapsed = max(now - started, 0.001)
                        rate = received / elapsed
                        current = offset + received
                        if total:
                            remaining = max(total - current, 0)
                            eta = remaining / rate if rate else 0
                            print('\rGSOD {}: {:.1f}% | {:.1f}/{:.1f} MB | {:.1f} MB/s | ETA {:.0f}s'.format(
                                year, current * 100 / total, current / 1e6,
                                total / 1e6, rate / 1e6, eta), end='', flush=True)
                        else:
                            print('\rGSOD {}: {:.1f} MB downloaded | {:.1f} MB/s'.format(
                                year, current / 1e6, rate / 1e6), end='', flush=True)
                        last_report = now
        if received == 0 and not (offset and os.path.getsize(partial) > 0):
            raise RuntimeError('NOAA returned an empty archive for {}'.format(url))
        os.replace(partial, destination)
        elapsed = max(time.monotonic() - started, 0.001)
        print('\rDownloaded GSOD {}: {:.1f} MB in {:.1f}s ({:.1f} MB/s)'.format(
            year, os.path.getsize(destination) / 1e6, elapsed,
            received / elapsed / 1e6))
        return 'downloaded'
    except (HTTPError, URLError, TimeoutError, OSError, RuntimeError):
        # Keep a partial file so the next run can issue a Range request.
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-year', type=int, required=True)
    parser.add_argument('--end-year', type=int, required=True)
    parser.add_argument('--archive-dir', required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.end_year < args.start_year:
        parser.error('--end-year must be >= --start-year')
    totals = {'downloaded': 0, 'existing': 0}
    years = list(range(args.start_year, args.end_year + 1))
    started = time.monotonic()
    for index, year in enumerate(years, 1):
        elapsed = time.monotonic() - started
        rate = (index - 1) / elapsed if elapsed else 0
        eta = (len(years) - index + 1) / rate if rate else 0
        print('[{}/{}] Starting archive {} | elapsed {:.0f}s | ETA {:.0f}s'.format(
            index, len(years), year, elapsed, eta))
        try:
            result = download_one(year, args.archive_dir, args.overwrite)
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
            parser.error('download failed for {}: {}'.format(year, exc))
        totals[result] += 1
    print('archive download summary: {}'.format(totals))


if __name__ == '__main__':
    main()

