import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path

# Konfiguration
PATH_HIST   = Path("/Users/clara/Desktop/uni_dreck/Menden_S26/Wetter_Zeitreihe_Historisch_19470101_20241231_05705.txt")
PATH_RECENT = Path("/Users/clara/Desktop/uni_dreck/Menden_S26/produkt_klima_tag_20241110_20260513_05705.txt")

# Hilfsfunktionen
def read_txt(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=';', skipinitialspace=True, encoding='latin-1')

def bereinige(series: pd.Series, name: str, clip_lower: float = None) -> pd.Series:
    n_vorher = series.isna().sum()
    series   = series.reindex(pd.date_range(series.index[0], series.index[-1], freq='D'))
    series   = series.interpolate(method='linear').bfill().ffill()
    if clip_lower is not None:
        series = series.clip(lower=clip_lower)
    print(f"  {name}: {n_vorher} fehlende Werte ersetzt → verbleibend: {series.isna().sum()}")
    return series

# Laden & Zusammenführen
df = (pd.concat([read_txt(PATH_HIST), read_txt(PATH_RECENT)], ignore_index=True)
        .pipe(lambda d: d.set_index(
            pd.to_datetime(d['MESS_DATUM'].astype(str).str.strip(), format='%Y%m%d')
        ).drop(columns=['MESS_DATUM']))
        .rename(columns=str.strip)
        .replace({-999.0: float('nan'), -999: float('nan')})
        .sort_index()
        .loc[lambda d: ~d.index.duplicated(keep='last')])
df = df.loc['1965-01-01':]

# Zeitreihen extrahieren & bereinigen
print('\n=== Datenbereinigung ===')

df_wind  = bereinige(df['FM'].rename('windgeschwindigkeit_ms'),  'Wind',       clip_lower=0.0).to_frame()
df_druck = bereinige(df['PM'].rename('luftdruck_hpa'),           'Luftdruck'                 ).to_frame()
df_temp  = bereinige(df['TMK'].rename('temperatur_mittel_c'),    'Temperatur'                ).to_frame()

print('\nBereinigung abgeschlossen ✅')
print(f"\nZeitraum: {df.index[0].date()} – {df.index[-1].date()}")
print(f"Beobachtungen gesamt: {len(df)}")