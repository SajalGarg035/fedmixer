"""Clean station data and emit chronological, hemisphere-aware season tasks."""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd


BASE_FEATURES = ('TEMP', 'DEWP', 'SLP', 'STP', 'VISIB', 'WDSP', 'MXSPD',
                 'GUST', 'MAX', 'MIN', 'PRCP', 'SNDP')
MISSING_VALUES = {
    'TEMP': (9999.9,), 'DEWP': (9999.9,), 'SLP': (9999.9,),
    'STP': (9999.9,), 'VISIB': (999.9,), 'WDSP': (999.9,),
    'MXSPD': (999.9,), 'GUST': (999.9,), 'MAX': (9999.9,),
    'MIN': (9999.9,), 'PRCP': (99.99,), 'SNDP': (999.9,),
}
NUMBER_PREFIX = re.compile(r'^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))')


def parse_numeric(series, column):
    text = series.astype('string').str.extract(NUMBER_PREFIX, expand=False)
    values = pd.to_numeric(text, errors='coerce')
    return values.mask(values.isin(MISSING_VALUES.get(column, ())))


def read_station_files(raw_dir):
    paths = sorted(glob.glob(os.path.join(raw_dir, '*', '*.csv')))
    if not paths:
        raise ValueError('no raw station/year CSV files found under {}'.format(raw_dir))
    grouped_paths = {}
    for path in paths:
        station_id = os.path.splitext(os.path.basename(path))[0]
        if not re.fullmatch(r'\d{11}', station_id):
            continue
        grouped_paths.setdefault(station_id, []).append(path)

    # Load one station at a time. A multi-decade, all-station archive can be
    # several gigabytes; holding every CSV as a DataFrame at once can exhaust RAM.
    for station_id, station_paths in sorted(grouped_paths.items()):
        frames = []
        for path in station_paths:
            frame = pd.read_csv(path, dtype={'STATION': 'string'})
            frame.columns = [str(column).strip().upper() for column in frame.columns]
            if 'DATE' not in frame:
                print('skip {}: no DATE column'.format(path))
                continue
            frame['DATE'] = pd.to_datetime(frame['DATE'], errors='coerce').dt.normalize()
            frame = frame.dropna(subset=['DATE'])
            frame['STATION_ID'] = station_id
            for column in BASE_FEATURES:
                if column in frame:
                    frame[column] = parse_numeric(frame[column], column)
            frames.append(frame)
        if frames:
            yield station_id, pd.concat(frames, ignore_index=True).sort_values('DATE')


def season_for_date(date, hemisphere):
    month = date.month
    if hemisphere == 'north':
        names = {12: 'winter', 1: 'winter', 2: 'winter',
                 3: 'spring', 4: 'spring', 5: 'spring',
                 6: 'summer', 7: 'summer', 8: 'summer',
                 9: 'autumn', 10: 'autumn', 11: 'autumn'}
    else:
        names = {12: 'summer', 1: 'summer', 2: 'summer',
                 3: 'autumn', 4: 'autumn', 5: 'autumn',
                 6: 'winter', 7: 'winter', 8: 'winter',
                 9: 'spring', 10: 'spring', 11: 'spring'}
    name = names[month]
    start_month = {'winter': 12 if hemisphere == 'north' else 6,
                   'spring': 3 if hemisphere == 'north' else 9,
                   'summer': 6 if hemisphere == 'north' else 12,
                   'autumn': 9 if hemisphere == 'north' else 3}[name]
    start_year = date.year - 1 if start_month == 12 and month in (1, 2) else date.year
    return name, pd.Timestamp(start_year, start_month, 1)


def season_starts(first_date, last_date):
    starts = []
    for year in range(first_date.year - 1, last_date.year + 1):
        for month in (3, 6, 9, 12):
            start = pd.Timestamp(year, month, 1)
            if start <= last_date and start + pd.DateOffset(months=3) > first_date:
                starts.append(start)
    return starts


def metadata_value(frame, column, default=''):
    if column not in frame:
        return default
    values = frame[column].dropna()
    return values.iloc[0] if len(values) else default


def make_task_windows(calendar, features, input_days, forecast_days, cutoff):
    values = calendar[list(features)].to_numpy(dtype=np.float32, na_value=np.nan)
    dates = calendar.index.to_numpy(dtype='datetime64[D]')
    train_x, train_y, train_dates = [], [], []
    test_x, test_y, test_dates = [], [], []
    total = input_days + forecast_days
    for start in range(max(0, len(calendar) - total + 1)):
        end_input = start + input_days
        end_target = end_input + forecast_days
        input_values = values[start:end_input]
        target_values = values[end_input:end_target]
        if not np.isfinite(input_values).all() or not np.isfinite(target_values).all():
            continue
        target_start = pd.Timestamp(dates[end_input])
        target_end = pd.Timestamp(dates[end_target - 1])
        if target_end < cutoff:
            train_x.append(input_values)
            train_y.append(target_values)
            train_dates.append(str(target_start.date()))
        elif target_start >= cutoff:
            test_x.append(input_values)
            test_y.append(target_values)
            test_dates.append(str(target_start.date()))

    def arrays(inputs, targets):
        if not inputs:
            return (np.empty((0, input_days, len(features)), dtype=np.float32),
                    np.empty((0, forecast_days, len(features)), dtype=np.float32))
        return np.stack(inputs), np.stack(targets)

    train_x, train_y = arrays(train_x, train_y)
    test_x, test_y = arrays(test_x, test_y)
    return train_x, train_y, test_x, test_y, train_dates, test_dates


def build_tasks(raw_dir, output_dir, features, input_days=31, forecast_days=7,
                min_temp_coverage=0.90, edge_tolerance_days=7,
                interpolation_limit=2):
    grouped = read_station_files(raw_dir)
    if 'TEMP' not in features:
        raise ValueError('TEMP must be included in --features for season coverage checks')
    os.makedirs(output_dir, exist_ok=True)
    manifest = []

    for station_id, source in grouped:
        source = source.drop_duplicates(subset=['DATE'], keep='last').set_index('DATE').sort_index()
        first_date, last_date = source.index.min(), source.index.max()
        latitude = pd.to_numeric(metadata_value(source, 'LATITUDE', np.nan), errors='coerce')
        if pd.isna(latitude):
            manifest.append({'station_id': station_id, 'status': 'skipped',
                             'reason': 'missing station latitude'})
            continue
        hemisphere = 'north' if latitude >= 0 else 'south'
        tropical_station = bool(abs(latitude) < 23.5)
        station_name = metadata_value(source, 'NAME')
        longitude = metadata_value(source, 'LONGITUDE', np.nan)
        elevation = metadata_value(source, 'ELEVATION', np.nan)

        for start in season_starts(first_date, last_date):
            end = start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
            expected_dates = pd.date_range(start, end, freq='D')
            raw_rows = source.reindex(expected_dates)
            temp_observed = raw_rows['TEMP'].dropna() if 'TEMP' in raw_rows else pd.Series(dtype=float)
            if temp_observed.empty:
                reason = 'no observed temperature in season'
                coverage = 0.0
            else:
                coverage = float(raw_rows['TEMP'].notna().mean())
                first_observed, last_observed = temp_observed.index.min(), temp_observed.index.max()
                edge = pd.Timedelta(days=edge_tolerance_days)
                if first_observed > start + edge:
                    reason = 'station starts partway through season'
                elif last_observed < end - edge:
                    reason = 'station ends partway through season'
                elif coverage < min_temp_coverage:
                    reason = 'temperature coverage below threshold'
                else:
                    reason = ''

            season_name, _ = season_for_date(start, hemisphere)
            season_id = '{}_{}_{}'.format(station_id, start.year, season_name)
            if reason:
                manifest.append({
                    'station_id': station_id, 'station_name': station_name,
                    'latitude': latitude, 'longitude': longitude,
                    'hemisphere': hemisphere, 'tropical_station': tropical_station,
                    'season_id': season_id,
                    'season_start': str(start.date()), 'season_end': str(end.date()),
                    'temp_coverage': round(coverage, 4), 'status': 'skipped',
                    'reason': reason,
                })
                continue

            calendar = raw_rows.copy()
            numeric_features = [feature for feature in features if feature in calendar]
            missing_features = [feature for feature in features if feature not in calendar]
            if missing_features:
                manifest.append({
                    'station_id': station_id, 'station_name': station_name,
                    'latitude': latitude, 'longitude': longitude,
                    'hemisphere': hemisphere, 'tropical_station': tropical_station,
                    'season_id': season_id,
                    'season_start': str(start.date()), 'season_end': str(end.date()),
                    'temp_coverage': round(coverage, 4), 'status': 'skipped',
                    'reason': 'missing feature columns: {}'.format(','.join(missing_features)),
                })
                continue

            interpolated_features = [
                feature for feature in numeric_features
                if feature not in ('PRCP', 'SNDP')
            ]
            if interpolated_features and interpolation_limit:
                calendar[interpolated_features] = calendar[interpolated_features].interpolate(
                    method='time', limit=interpolation_limit, limit_area='inside')
            calendar.index.name = 'DATE'
            cutoff = start + pd.Timedelta(days=int(len(calendar) * 0.8))
            train_x, train_y, test_x, test_y, train_dates, test_dates = make_task_windows(
                calendar, features, input_days, forecast_days, cutoff)
            station_dir = os.path.join(output_dir, 'stations', station_id)
            windows_dir = os.path.join(output_dir, 'windows', station_id)
            os.makedirs(station_dir, exist_ok=True)
            os.makedirs(windows_dir, exist_ok=True)

            season_frame = calendar.reset_index()
            season_frame['STATION_ID'] = station_id
            season_frame['SEASON_ID'] = season_id
            season_frame['SEASON'] = season_name
            season_frame['HEMISPHERE'] = hemisphere
            season_frame['TROPICAL_STATION'] = tropical_station
            season_frame['STATION_NAME'] = station_name
            season_frame['LATITUDE'] = latitude
            season_frame['LONGITUDE'] = longitude
            season_frame['ELEVATION'] = elevation
            leading_columns = [
                'STATION_ID', 'SEASON_ID', 'SEASON', 'HEMISPHERE',
                'STATION_NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'DATE',
            ]
            season_frame = season_frame[
                leading_columns + [column for column in season_frame.columns
                                   if column not in leading_columns]
            ]
            season_frame.to_csv(os.path.join(station_dir, season_id + '.csv'), index=False)

            np.savez_compressed(
                os.path.join(windows_dir, season_id + '.npz'),
                train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y,
                train_target_start_dates=np.asarray(train_dates, dtype='U10'),
                test_target_start_dates=np.asarray(test_dates, dtype='U10'),
                features=np.asarray(features, dtype='U16'),
                station_id=station_id, season=season_name, hemisphere=hemisphere,
                season_start=str(start.date()), season_end=str(end.date()),
                input_days=input_days, forecast_days=forecast_days,
            )
            manifest.append({
                'station_id': station_id, 'station_name': station_name,
                'latitude': latitude, 'longitude': longitude,
                'hemisphere': hemisphere, 'tropical_station': tropical_station,
                'season_id': season_id,
                'season_start': str(start.date()), 'season_end': str(end.date()),
                'temp_coverage': round(coverage, 4), 'status': 'included',
                'reason': '', 'train_windows': len(train_x), 'test_windows': len(test_x),
            })

    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, default=str)
    included = sum(item.get('status') == 'included' for item in manifest)
    skipped = sum(item.get('status') == 'skipped' for item in manifest)
    print('created {} season tasks; skipped {} incomplete/unusable tasks'.format(
        included, skipped))
    print('manifest: {}'.format(manifest_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', default='./data/global_surface/raw')
    parser.add_argument('--output-dir', default='./data/global_surface/processed')
    parser.add_argument('--features', default='TEMP',
                        help='comma-separated GSOD numeric feature columns; default TEMP')
    parser.add_argument('--input-days', type=int, default=31)
    parser.add_argument('--forecast-days', type=int, default=7)
    parser.add_argument('--min-temp-coverage', type=float, default=0.90)
    parser.add_argument('--edge-tolerance-days', type=int, default=7)
    parser.add_argument('--interpolation-limit', type=int, default=2)
    args = parser.parse_args()
    features = [feature.strip().upper() for feature in args.features.split(',') if feature.strip()]
    if not features:
        parser.error('--features must contain at least one column')
    if not 0.0 <= args.min_temp_coverage <= 1.0:
        parser.error('--min-temp-coverage must be between 0 and 1')
    if args.input_days < 1 or args.forecast_days < 1 or args.interpolation_limit < 0:
        parser.error('window lengths must be positive and interpolation limit nonnegative')
    try:
        build_tasks(args.raw_dir, args.output_dir, features,
                    args.input_days, args.forecast_days,
                    args.min_temp_coverage, args.edge_tolerance_days,
                    args.interpolation_limit)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()

