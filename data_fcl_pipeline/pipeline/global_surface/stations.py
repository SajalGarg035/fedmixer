"""Shared helpers for validating GSOD station identifiers."""

import re


def read_station_ids(path=None, values=None):
    station_ids = list(values or [])
    if path:
        with open(path, encoding='utf-8') as handle:
            station_ids.extend(
                line.strip() for line in handle
                if line.strip() and not line.lstrip().startswith('#')
            )
    station_ids = list(dict.fromkeys(station_ids))
    invalid = [station_id for station_id in station_ids
               if not re.fullmatch(r'\d{11}', station_id)]
    if not station_ids:
        raise ValueError('provide --stations or --stations-file')
    if invalid:
        raise ValueError('station IDs must be 11 digits: {}'.format(', '.join(invalid)))
    return station_ids

