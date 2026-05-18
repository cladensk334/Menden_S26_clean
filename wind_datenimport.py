import os
import pandas as pd

os.system("rm -rf Menden_S26_clean")
os.system("git clone https://github.com/cladensk334/Menden_S26_clean.git")

wind_clean = pd.read_csv(
    "Menden_S26_clean/data/originals&cleaned/wind_bereinigt.csv",
    index_col="datum",
    parse_dates=True
)["windgeschwindigkeit_ms"]

#wind_clean = wind_clean[wind_clean.index >= "1966-01-01"]
#wind_clean = wind_clean[wind_clean.index <= "2026-01-11"]

print(f"   Zeitraum:      {wind_clean.index[0].date()} bis {wind_clean.index[-1].date()}")
print(f"   Beobachtungen: {len(wind_clean)}")
print(f"   Fehlende Werte: {wind_clean.isna().sum()}")
