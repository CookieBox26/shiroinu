import pandas as pd
import os


def col_rule_weather(c: str) -> str:
    """Fix specific garbled column names in weather.csv.
    """
    if c.startswith('SWDR'):
        return 'SWDR (W/m^2)'
    if c.startswith('PAR'):
        return 'PAR (umol/m^2/s)'
    if c.startswith('max. PAR'):
        return 'max. PAR (umol/m^2/s)'
    return c


if __name__ == '__main__':
    df = pd.read_csv(os.path.expanduser('~/dataset/weather/weather.csv'))
    df.rename(columns=lambda c: col_rule_weather(c), inplace=True)
    df.to_csv(os.path.expanduser('~/dataset/weather/weather.csv'), index=False)
