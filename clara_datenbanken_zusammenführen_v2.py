import pandas as pd
import numpy as np
from pathlib import Path

# Konfiguration
PATH_HIST   = Path("/Users/clara/Desktop/uni_dreck/Menden_S26/Wetter_Zeitreihe_Historisch_19470101_20241231_05705.txt")
PATH_RECENT = Path("/Users/clara/Desktop/uni_dreck/Menden_S26/produkt_klima_tag_20241110_20260513_05705.txt")

FEHLWERTE = [-999, -999.0, '-999', '-999.0']

def read_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';', skipinitialspace=True, encoding='latin-1')
    df.columns = df.columns.str.strip()
    return df

def bereinige(series: pd.Series, name: str, clip_lower: float = None) -> pd.Series:
    # 1. Vollständigen Tagesindex aufspannen (deckt fehlende Tage auf)
    voll_index = pd.date_range(series.index[0], series.index[-1], freq='D')
    series = series.reindex(voll_index)

    # 2. Jetzt erst zählen – erfasst alle Lücken inkl. fehlender Tage
    n_vorher = series.isna().sum()

    # 3. Lineare Interpolation für innere Lücken, dann Rand auffüllen
    series = series.interpolate(method='time').bfill().ffill()

    if clip_lower is not None:
        series = series.clip(lower=clip_lower)

    n_nachher = series.isna().sum()
    print(f"  {name}: {n_vorher} fehlende Werte → nach Bereinigung verbleibend: {n_nachher}")
    return series


# Laden
df_hist   = read_txt(PATH_HIST)
df_recent = read_txt(PATH_RECENT)

# Zusammenführen (historisch zuerst, damit keep='last' die aktuellere Datei bevorzugt)
df = (pd.concat([df_hist, df_recent], ignore_index=True)
        .pipe(lambda d: d.set_index(
            pd.to_datetime(d['MESS_DATUM'].astype(str).str.strip(), format='%Y%m%d')
        ).drop(columns=['MESS_DATUM']))
        .replace({v: np.nan for v in FEHLWERTE})
        .sort_index()
        .loc[lambda d: ~d.index.duplicated(keep='last')])

# Spalten sicher in numerisch umwandeln (fängt alle verbliebenen Textwerte ab)
for col in ['FM', 'PM', 'TMK']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Ab 1965 beschränken
df = df.loc['1965-01-01':]

# Zeitreihen extrahieren & bereinigen
print('\n=== Datenbereinigung ===')

df_wind  = bereinige(df['FM'].rename('windgeschwindigkeit_ms'),  'Wind',       clip_lower=0.0).to_frame()
df_druck = bereinige(df['PM'].rename('luftdruck_hpa'),           'Luftdruck'                 ).to_frame()
df_temp  = bereinige(df['TMK'].rename('temperatur_mittel_c'),    'Temperatur'                ).to_frame()

print('\nBereinigung abgeschlossen')
print(f"Zeitraum: {df.index[0].date()} – {df.index[-1].date()}")
print(f"Beobachtungen gesamt: {len(df)}")

# Abschlusskontrolle
for name, frame in [('Wind', df_wind), ('Luftdruck', df_druck), ('Temperatur', df_temp)]:
    assert frame.isna().sum().sum() == 0, f"Noch NaN in {name}!"
print("Alle Zeitreihen sind vollständig lückenfrei.")