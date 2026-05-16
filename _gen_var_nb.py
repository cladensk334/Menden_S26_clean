# -*- coding: utf-8 -*-
"""Erzeugt var_modell_multivariat.ipynb"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "var_modell_multivariat.ipynb"

def md(text):
    lines = text.strip().splitlines(keepends=True)
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code(text):
    lines = text.strip().splitlines(keepends=True)
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": lines}

cells = []

# ── 0  Titel ─────────────────────────────────────────────────────────────────
cells.append(md(
    "# VAR-Modell – Multivariate Zeitreihenanalyse\n"
    "## Luftdruck · Windgeschwindigkeit · Temperatur (Würzburg, Station 5705)\n\n"
    "**Datenquelle:** `data/originals&cleaned/` – je eine bereinigte CSV pro Variable  \n"
    "**Methode:** Vector Autoregression (VAR)  \n\n"
    "**Analyse-Schritte:**  \n"
    "0. Daten laden & zusammenführen  \n"
    "1. Explorative Analyse (Plots, Korrelation)  \n"
    "2. Stationaritätstests (ADF, KPSS) je Variable  \n"
    "3. Transformation zur Stationarität  \n"
    "4. Lag-Selektion (AIC, BIC, HQIC)  \n"
    "5. Modellschätzung & Zusammenfassung  \n"
    "6. Residuendiagnose (Whiteness, Normalität)  \n"
    "7. Granger-Kausalität  \n"
    "8. Impuls-Antwort-Analyse (IRF) & Varianzdekomposition (FEVD)  \n"
    "9. Train/Val/Test-Split (70/15/15) & Evaluation  \n"
    "10. Time-Series-Cross-Validation (5-Fold)  \n"
    "11. 10-Tage-Prognose  \n"
    "11b. Holdout-Evaluation  \n"
))

# ── 1  Imports ────────────────────────────────────────────────────────────────
cells.append(code(
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n\n"
    "import os\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.dates as mdates\n"
    "from pathlib import Path\n\n"
    "from statsmodels.tsa.stattools import adfuller, kpss\n"
    "from statsmodels.tsa.vector_ar.var_model import VAR\n"
    "from statsmodels.stats.stattools import durbin_watson\n"
    "from sklearn.metrics import mean_squared_error, mean_absolute_error\n"
    "from sklearn.model_selection import TimeSeriesSplit\n"
    "from scipy import stats\n\n"
    "plt.rcParams['figure.dpi'] = 120\n"
    "plt.rcParams['font.family'] = 'DejaVu Sans'\n\n"
    "def _repo_root() -> Path:\n"
    "    p = Path().resolve()\n"
    "    for folder in [p, *p.parents]:\n"
    "        if (folder / '.git').is_dir():\n"
    "            return folder\n"
    "    return p\n\n"
    "ROOT       = _repo_root()\n"
    "PLOTORDNER = str(ROOT / 'Plots')\n"
    "os.makedirs(PLOTORDNER, exist_ok=True)\n"
    "print(f'Repo-Root  : {ROOT}')\n"
    "print(f'Plotordner : {PLOTORDNER}')\n"
))

# ── 2  Markdown ───────────────────────────────────────────────────────────────
cells.append(md("## 0. DATEN LADEN UND ZUSAMMENFÜHREN"))

# ── 3  Laden ──────────────────────────────────────────────────────────────────
cells.append(code(
    "DIR = ROOT / 'data' / 'originals&cleaned'\n\n"
    "def lade_csv(dateiname: str, spalte: str) -> pd.Series:\n"
    "    s = (pd.read_csv(DIR / dateiname, parse_dates=['datum'], index_col='datum')\n"
    "           .squeeze().asfreq('D'))\n"
    "    s.name = spalte\n"
    "    return s\n\n"
    "ts_luft = lade_csv('luftdruck_bereinigt.csv',  'luftdruck_hpa')\n"
    "ts_wind = lade_csv('wind_bereinigt.csv',        'windgeschwindigkeit_ms')\n"
    "ts_temp = lade_csv('temperatur_bereinigt.csv',  'temperatur_mittel_c')\n\n"
    "df = pd.concat([ts_luft, ts_wind, ts_temp], axis=1).dropna()\n\n"
    "# Cutoff & Holdout\n"
    "CUTOFF_DATE = pd.Timestamp('2026-01-01')\n"
    "df = df.loc[df.index < CUTOFF_DATE]\n\n"
    "df_holdout = df.iloc[-10:]\n"
    "df         = df.iloc[:-10]\n\n"
    "VARIABLEN = list(df.columns)\n\n"
    "print(f'Variablen     : {VARIABLEN}')\n"
    "print(f'Analyse-Reihe : {df.index[0].date()} bis {df.index[-1].date()}  ({len(df):,} Tage)')\n"
    "print(f'Holdout       : {df_holdout.index[0].date()} bis {df_holdout.index[-1].date()}  (10 Tage)')\n"
    "print()\n"
    "print(df.describe().round(2))\n"
))

# ── 4  Markdown EDA ───────────────────────────────────────────────────────────
cells.append(md("## 1. EXPLORATIVE ANALYSE"))

# ── 5  EDA ────────────────────────────────────────────────────────────────────
cells.append(code(
    "fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)\n"
    "farben    = ['steelblue', 'darkorange', 'seagreen']\n"
    "einheiten = ['hPa', 'm/s', '°C']\n\n"
    "for ax, col, farbe, einheit in zip(axes, VARIABLEN, farben, einheiten):\n"
    "    ax.plot(df.index, df[col], lw=0.6, color=farbe, alpha=0.9)\n"
    "    ax.set_ylabel(f'{col}\\n[{einheit}]', fontsize=9)\n"
    "    ax.grid(True, alpha=0.3)\n\n"
    "axes[0].set_title('Tägliche Zeitreihen – Würzburg (Station 5705)', fontsize=13)\n"
    "axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))\n"
    "axes[-1].xaxis.set_major_locator(mdates.YearLocator(10))\n"
    "axes[-1].tick_params(axis='x', rotation=45)\n"
    "plt.tight_layout()\n"
    "plt.savefig(os.path.join(PLOTORDNER, 'VAR_01_zeitreihen.png'), bbox_inches='tight')\n"
    "plt.show()\n\n"
    "# Korrelationsmatrix\n"
    "print('\\nKorrelationsmatrix (Pearson):')\n"
    "corr = df.corr()\n"
    "print(corr.round(3))\n\n"
    "fig, ax = plt.subplots(figsize=(6, 5))\n"
    "im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)\n"
    "ax.set_xticks(range(len(VARIABLEN)))\n"
    "ax.set_xticklabels(VARIABLEN, rotation=20, ha='right', fontsize=8)\n"
    "ax.set_yticks(range(len(VARIABLEN)))\n"
    "ax.set_yticklabels(VARIABLEN, fontsize=8)\n"
    "for i in range(len(VARIABLEN)):\n"
    "    for j in range(len(VARIABLEN)):\n"
    "        ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center',\n"
    "                fontsize=11, color='black' if abs(corr.iloc[i,j]) < 0.7 else 'white')\n"
    "plt.colorbar(im, ax=ax)\n"
    "ax.set_title('Korrelationsmatrix', fontsize=12)\n"
    "plt.tight_layout()\n"
    "plt.savefig(os.path.join(PLOTORDNER, 'VAR_02_korrelation.png'), bbox_inches='tight')\n"
    "plt.show()\n"
))

# ── 6  Markdown Stationaritaet ────────────────────────────────────────────────
cells.append(md("## 2. STATIONARITÄTSTESTS (ADF & KPSS)"))

# ── 7  Stationaritaet ─────────────────────────────────────────────────────────
cells.append(code(
    "print('=' * 70)\n"
    "print('STATIONARITÄTSTESTS  (5%-Niveau)')\n"
    "print('=' * 70)\n\n"
    "def adf_test(serie, name):\n"
    "    r = adfuller(serie.dropna(), autolag='AIC')\n"
    "    ok = r[1] < 0.05\n"
    "    print(f'  ADF  {name:<30}  p={r[1]:.4f}  =>  {\"STATIONAER\" if ok else \"NICHT STATIONAER\"}')\n"
    "    return ok\n\n"
    "def kpss_test(serie, name):\n"
    "    r = kpss(serie.dropna(), regression='c', nlags='auto')\n"
    "    ok = r[1] >= 0.05\n"
    "    print(f'  KPSS {name:<30}  p={r[1]:.4f}  =>  {\"STATIONAER\" if ok else \"NICHT STATIONAER\"}')\n"
    "    return ok\n\n"
    "stationaer_niveau = {}\n"
    "print('\\n--- Niveau ---')\n"
    "for col in VARIABLEN:\n"
    "    a = adf_test(df[col], col)\n"
    "    k = kpss_test(df[col], col)\n"
    "    stationaer_niveau[col] = (a and k)\n\n"
    "stationaer_diff = {}\n"
    "print('\\n--- Erste Differenz ---')\n"
    "for col in VARIABLEN:\n"
    "    d1 = df[col].diff().dropna()\n"
    "    a  = adf_test(d1, col + ' (d=1)')\n"
    "    k  = kpss_test(d1, col + ' (d=1)')\n"
    "    stationaer_diff[col] = (a and k)\n\n"
    "d_noetig = not all(stationaer_niveau.values())\n"
    "print(f'\\n  => Differenzierung noetig: {d_noetig}')\n"
    "print(f'     Stationaer im Niveau   : {stationaer_niveau}')\n"
))

# ── 8  Markdown Transformation ────────────────────────────────────────────────
cells.append(md("## 3. TRANSFORMATION ZUR STATIONARITÄT"))

# ── 9  Transformation ─────────────────────────────────────────────────────────
cells.append(code(
    "if d_noetig:\n"
    "    df_stat = df.diff().dropna()\n"
    "    d_VAR   = 1\n"
    "    print('  1-fache Differenzierung angewendet.')\n"
    "else:\n"
    "    df_stat = df.copy()\n"
    "    d_VAR   = 0\n"
    "    print('  Keine Differenzierung noetig – Niveau wird verwendet.')\n\n"
    "print(f'  Beobachtungen nach Transformation: {len(df_stat):,}')\n"
    "print(f'  Integrationsordnung d = {d_VAR}')\n\n"
    "fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)\n"
    "titel  = 'Differenzierte Reihen (d=1)' if d_VAR else 'Niveau-Reihen (d=0)'\n"
    "farben = ['steelblue', 'darkorange', 'seagreen']\n\n"
    "for ax, col, farbe in zip(axes, VARIABLEN, farben):\n"
    "    ax.plot(df_stat.index, df_stat[col], lw=0.6, color=farbe)\n"
    "    ax.axhline(0, color='red', ls='--', lw=0.8)\n"
    "    ax.set_ylabel(col, fontsize=9)\n"
    "    ax.grid(True, alpha=0.3)\n\n"
    "axes[0].set_title(f'Transformierte Zeitreihen – {titel}', fontsize=12)\n"
    "axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))\n"
    "axes[-1].xaxis.set_major_locator(mdates.YearLocator(10))\n"
    "axes[-1].tick_params(axis='x', rotation=45)\n"
    "plt.tight_layout()\n"
    "plt.savefig(os.path.join(PLOTORDNER, 'VAR_03_transformation.png'), bbox_inches='tight')\n"
    "plt.show()\n"
))

# ── 10  Markdown Lag ──────────────────────────────────────────────────────────
cells.append(md("## 4. LAG-SELEKTION"))

# ── 11  Lag-Selektion ─────────────────────────────────────────────────────────
cells.append(code(
    "print('=' * 70)\n"
    "print('LAG-SELEKTION – VAR(p)')\n"
    "print('=' * 70)\n\n"
    "modell_var = VAR(df_stat)\n"
    "lag_auswahl = modell_var.select_order(maxlags=15)\n"
    "print(lag_auswahl.summary())\n\n"
    "p_aic  = int(lag_auswahl.aic)  if int(lag_auswahl.aic)  > 0 else 1\n"
    "p_bic  = int(lag_auswahl.bic)  if int(lag_auswahl.bic)  > 0 else 1\n\n"
    "print(f'  AIC -> p = {p_aic}')\n"
    "print(f'  BIC -> p = {p_bic}')\n\n"
    "# Sparsamster Kandidat (analog zur ARIMA-Modellselektion)\n"
    "if p_aic == p_bic:\n"
    "    p_final = p_aic\n"
    "    grund = 'AIC und BIC einig'\n"
    "elif p_bic < p_aic:\n"
    "    p_final = p_bic\n"
    "    grund = f'BIC sparsamer ({p_bic} Lags) vs. AIC ({p_aic} Lags)'\n"
    "else:\n"
    "    p_final = p_aic\n"
    "    grund = f'AIC sparsamer ({p_aic} Lags) vs. BIC ({p_bic} Lags)'\n\n"
    "p_final = max(p_final, 1)\n"
    "print(f'\\n  => Gewaaehltes Modell: VAR({p_final})  [{grund}]')\n"
))

# ── 12  Markdown Fit ──────────────────────────────────────────────────────────
cells.append(md("## 5. MODELLSCHÄTZUNG"))

# ── 13  Fit ───────────────────────────────────────────────────────────────────
cells.append(code(
    "print('=' * 70)\n"
    "print(f'VAR({p_final})-MODELL SCHAETZEN')\n"
    "print('=' * 70)\n\n"
    "fit_var = modell_var.fit(p_final)\n"
    "print(fit_var.summary())\n"
))

# ── 14  Markdown Residuen ─────────────────────────────────────────────────────
cells.append(md("## 6. RESIDUENDIAGNOSE"))

# ── 15  Residuen ──────────────────────────────────────────────────────────────
cells.append(code(
    "print('=' * 70)\n"
    "print('RESIDUENDIAGNOSE')\n"
    "print('=' * 70)\n\n"
    "residuen = pd.DataFrame(fit_var.resid, columns=VARIABLEN,\n"
    "                        index=df_stat.index[p_final:])\n\n"
    "print('\\n  Durbin-Watson (Wert nahe 2 = keine Autokorrelation):')\n"
    "for col in VARIABLEN:\n"
    "    dw = durbin_watson(residuen[col])\n"
    "    print(f'    {col:<32}: {dw:.4f}')\n\n"
    "print('\\n  Portmanteau-Test (H0: keine Autokorrelation in Residuen):')\n"
    "pt = fit_var.test_whiteness(nlags=10, signif=0.05)\n"
    "print(pt.summary())\n\n"
    "print('\\n  Normalitaetstest (Jarque-Bera):')\n"
    "for col in VARIABLEN:\n"
    "    jb, p_jb = stats.jarque_bera(residuen[col].dropna())\n"
    "    entscheid = 'nicht abgelehnt' if p_jb > 0.05 else 'ABGELEHNT'\n"
    "    print(f'    {col:<32}: JB={jb:.2f}  p={p_jb:.4f}  => H0 {entscheid}')\n\n"
    "fig, axes = plt.subplots(3, 2, figsize=(14, 10))\n"
    "farben = ['steelblue', 'darkorange', 'seagreen']\n\n"
    "for row, (col, farbe) in enumerate(zip(VARIABLEN, farben)):\n"
    "    res = residuen[col].dropna()\n"
    "    axes[row, 0].plot(res.index, res.values, lw=0.5, color=farbe)\n"
    "    axes[row, 0].axhline(0, color='red', ls='--', lw=0.8)\n"
    "    axes[row, 0].set_title(f'Residuen: {col}', fontsize=10)\n"
    "    axes[row, 0].grid(True, alpha=0.3)\n"
    "    axes[row, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))\n"
    "    axes[row, 0].xaxis.set_major_locator(mdates.YearLocator(10))\n"
    "    axes[row, 0].tick_params(axis='x', rotation=30)\n"
    "    axes[row, 1].hist(res, bins=50, color=farbe, alpha=0.7, density=True, edgecolor='white')\n"
    "    x = np.linspace(res.min(), res.max(), 200)\n"
    "    axes[row, 1].plot(x, stats.norm.pdf(x, res.mean(), res.std()), 'r-', lw=2)\n"
    "    axes[row, 1].set_title(f'Histogramm: {col}', fontsize=10)\n"
    "    axes[row, 1].grid(True, alpha=0.3)\n\n"
    "plt.tight_layout()\n"
    "plt.savefig(os.path.join(PLOTORDNER, 'VAR_04_residuen.png'), bbox_inches='tight')\n"
    "plt.show()\n"
))

# ── 16  Markdown Granger ──────────────────────────────────────────────────────
cells.append(md("## 7. GRANGER-KAUSALITÄT"))

# ── 17  Granger ───────────────────────────────────────────────────────────────
cells.append(code(
    "print('=' * 70)\n"
    "print('GRANGER-KAUSALITAET  (H0: X verursacht Y NICHT)')\n"
    "print('=' * 70)\n"
    "print(f'  Lag p={p_final}, Signifikanzniveau 5%\\n')\n\n"
    "for ursache in VARIABLEN:\n"
    "    for wirkung in VARIABLEN:\n"
    "        if ursache == wirkung:\n"
    "            continue\n"
    "        gc  = fit_var.test_causality(wirkung, [ursache], kind='f', signif=0.05)\n"
    "        p   = gc.pvalue\n"
    "        sig = '**' if p < 0.01 else '*' if p < 0.05 else '  '\n"
    "        ent = 'Kausalitaet JA' if p < 0.05 else 'keine Kausalitaet'\n"
    "        print(f'  {ursache:<28} -> {wirkung:<28}  p={p:.4f}  {sig}  {ent}')\n"
))

# ── 18  Markdown IRF ──────────────────────────────────────────────────────────
cells.append(md("## 8. IMPULS-ANTWORT-ANALYSE (IRF) & VARIANZDEKOMPOSITION (FEVD)"))

# ── 19  IRF + FEVD ────────────────────────────────────────────────────────────
cells.append(code(
    "# Orthogonalisierte IRF (Cholesky)\n"
    "irf = fit_var.irf(periods=30)\n"
    "fig = irf.plot(orth=True, figsize=(14, 10))\n"
    "fig.suptitle(f'Orthogonalisierte Impuls-Antwort-Funktionen – VAR({p_final})', fontsize=13)\n"
    "plt.tight_layout()\n"
    "plt.savefig(os.path.join(PLOTORDNER, 'VAR_05_irf.png'), bbox_inches='tight')\n"
    "plt.show()\n"
    "print('[Abbildung gespeichert: VAR_05_irf.png]')\n\n"
    "# FEVD\n"
    "fevd = fit_var.fevd(periods=30)\n"
    "fig  = fevd.plot(figsize=(14, 8))\n"
    "plt.suptitle(f'Varianzdekomposition (FEVD) – VAR({p_final})', fontsize=13)\n"
    "plt.tight_layout()\n"
    "plt.savefig(os.path.join(PLOTORDNER, 'VAR_06_fevd.png'), bbox_inches='tight')\n"
    "plt.show()\n"
    "print('[Abbildung gespeichert: VAR_06_fevd.png]')\n\n"
    "print('\\n  FEVD-Zusammenfassung (nach 10 Perioden):')\n"
    "print(fevd.summary())\n"
))

# ── 20  Markdown Train ────────────────────────────────────────────────────────
cells.append(md("## 9. TRAIN / VAL / TEST-SPLIT (70/15/15) & MODELL-EVALUATION"))

# ── 21  Train ─────────────────────────────────────────────────────────────────
cells.append(code(
    "print('=' * 70)\n"
    "print('TRAIN/VAL/TEST-SPLIT (70/15/15)')\n"
    "print('=' * 70)\n\n"
    "n         = len(df_stat)\n"
    "train_end = int(n * 0.70)\n"
    "val_end   = int(n * 0.85)\n\n"
    "df_train = df_stat.iloc[:train_end]\n"
    "df_val   = df_stat.iloc[train_end:val_end]\n"
    "df_test  = df_stat.iloc[val_end:]\n\n"
    "print(f'  Gesamt     : {n:,}')\n"
    "print(f'  Train      : {len(df_train):,}  ({df_train.index[0].date()} – {df_train.index[-1].date()})')\n"
    "print(f'  Validation : {len(df_val):,}   ({df_val.index[0].date()} – {df_val.index[-1].date()})')\n"
    "print(f'  Test       : {len(df_test):,}   ({df_test.index[0].date()} – {df_test.index[-1].date()})')\n\n"
    "fit_train = VAR(df_train).fit(p_final)\n"
    "n_test    = len(df_test)\n"
    "fc_arr    = fit_train.forecast(df_train.values[-p_final:], steps=n_test)\n"
    "fc_test   = pd.DataFrame(fc_arr, index=df_test.index, columns=VARIABLEN)\n\n"
    "print('\\n  Test-Set-Metriken (Multi-Step-Prognose):')\n"
    "print(f'  {\"Variable\":<32} {\"RMSE\":>8} {\"MAE\":>8} {\"MAPE\":>8}')\n"
    "print('  ' + '-' * 60)\n\n"
    "metriken_test = {}\n"
    "for col in VARIABLEN:\n"
    "    ist  = df_test[col].values\n"
    "    prog = fc_test[col].values\n"
    "    rmse = np.sqrt(mean_squared_error(ist, prog))\n"
    "    mae  = mean_absolute_error(ist, prog)\n"
    "    with np.errstate(divide='ignore', invalid='ignore'):\n"
    "        mape = np.mean(np.abs((ist - prog) / np.where(ist == 0, np.nan, ist))) * 100\n"
    "    metriken_test[col] = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}\n"
    "    print(f'  {col:<32} {rmse:>8.4f} {mae:>8.4f} {mape:>8.2f}')\n\n"
    "fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)\n"
    "farben = ['steelblue', 'darkorange', 'seagreen']\n"
    "einheiten = ['hPa', 'm/s', '°C']\n\n"
    "for ax, col, farbe, einheit in zip(axes, VARIABLEN, farben, einheiten):\n"
    "    ax.plot(df_test.index, df_test[col], color=farbe, lw=1.0, label='Ist-Werte')\n"
    "    ax.plot(fc_test.index, fc_test[col], color='red', lw=1.0, ls='--', label='Prognose')\n"
    "    m = metriken_test[col]\n"
    "    ax.set_title(f'{col} [{einheit}]  |  RMSE={m[\"RMSE\"]:.4f}  MAE={m[\"MAE\"]:.4f}', fontsize=10)\n"
    "    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)\n\n"
    "axes[0].set_title(f'VAR({p_final}) – Test-Evaluation (15%)', fontsize=12)\n"
    "axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))\n"
    "axes[-1].tick_params(axis='x', rotation=45)\n"
    "plt.tight_layout()\n"
    "plt.savefig(os.path.join(PLOTORDNER, 'VAR_07_test_evaluation.png'), bbox_inches='tight')\n"
    "plt.show()\n"
))

# ── 22  Markdown CV ───────────────────────────────────────────────────────────
cells.append(md("## 10. TIME-SERIES-CROSS-VALIDATION (5-Fold)"))

# ── 23  CV ────────────────────────────────────────────────────────────────────
cells.append(code(
    "print('=' * 70)\n"
    "print('5-FOLD TIME-SERIES-CROSS-VALIDATION')\n"
    "print('=' * 70)\n\n"
    "CV_FENSTER  = 1095\n"
    "CV_TESTSIZE = 30\n"
    "N_SPLITS    = 5\n\n"
    "df_cv  = df_train.iloc[-min(CV_FENSTER, len(df_train)):]\n"
    "tscv   = TimeSeriesSplit(n_splits=N_SPLITS, test_size=CV_TESTSIZE)\n\n"
    "print(f'  CV-Pool    : letzte {len(df_cv)} Tage der Trainingsdaten')\n"
    "print(f'  Fold-Groesse: {CV_TESTSIZE} Tage  |  Folds: {N_SPLITS}\\n')\n\n"
    "fold_records = []\n"
    "for fold, (idx_tr, idx_te) in enumerate(tscv.split(df_cv)):\n"
    "    cv_tr = df_cv.iloc[idx_tr]\n"
    "    cv_te = df_cv.iloc[idx_te]\n"
    "    fc    = VAR(cv_tr).fit(p_final).forecast(cv_tr.values[-p_final:], steps=len(cv_te))\n"
    "    fc_cv = pd.DataFrame(fc, index=cv_te.index, columns=VARIABLEN)\n"
    "    row   = {'Fold': fold + 1}\n"
    "    for col in VARIABLEN:\n"
    "        row[f'RMSE_{col}'] = np.sqrt(mean_squared_error(cv_te[col].values, fc_cv[col].values))\n"
    "        row[f'MAE_{col}']  = mean_absolute_error(cv_te[col].values, fc_cv[col].values)\n"
    "    fold_records.append(row)\n\n"
    "cv_res = pd.DataFrame(fold_records).set_index('Fold')\n\n"
    "header = f'  {\"Fold\":<6}'\n"
    "for col in VARIABLEN:\n"
    "    header += f'  {\"RMSE_\" + col[:6]:<12}  {\"MAE_\" + col[:6]:<12}'\n"
    "print(header)\n"
    "print('  ' + '-' * (6 + 28 * len(VARIABLEN)))\n"
    "for fold, row in cv_res.iterrows():\n"
    "    line = f'  {fold:<6}'\n"
    "    for col in VARIABLEN:\n"
    "        line += f'  {row[f\"RMSE_{col}\"]:>12.4f}  {row[f\"MAE_{col}\"]:>12.4f}'\n"
    "    print(line)\n"
    "print('  ' + '-' * (6 + 28 * len(VARIABLEN)))\n"
    "line = f'  {\"Mittel\":<6}'\n"
    "for col in VARIABLEN:\n"
    "    line += f'  {cv_res[f\"RMSE_{col}\"].mean():>12.4f}  {cv_res[f\"MAE_{col}\"].mean():>12.4f}'\n"
    "print(line)\n"
))

# ── 24  Markdown Prognose ─────────────────────────────────────────────────────
cells.append(md("## 11. 10-TAGE-PROGNOSE"))

# ── 25  Prognose ──────────────────────────────────────────────────────────────
cells.append(code(
    "print('=' * 70)\n"
    "print('10-TAGE-PROGNOSE (Gesamtmodell)')\n"
    "print('=' * 70)\n\n"
    "N_PROGNOSE  = 10\n"
    "fc_arr_prog = fit_var.forecast(df_stat.values[-p_final:], steps=N_PROGNOSE)\n"
    "prog_index  = pd.date_range(df_stat.index[-1] + pd.Timedelta(days=1),\n"
    "                            periods=N_PROGNOSE, freq='D')\n"
    "fc_prog     = pd.DataFrame(fc_arr_prog, index=prog_index, columns=VARIABLEN)\n\n"
    "print(f'\\n  Prognose ab {prog_index[0].strftime(\"%d.%m.%Y\")}:')\n"
    "print(f'  {\"Datum\":<14}', end='')\n"
    "for col in VARIABLEN:\n"
    "    print(f'  {col[:22]:>22}', end='')\n"
    "print()\n"
    "print('  ' + '-' * (14 + 24 * len(VARIABLEN)))\n"
    "for d, row in fc_prog.iterrows():\n"
    "    print(f'  {d.strftime(\"%d.%m.%Y\"):<14}', end='')\n"
    "    for col in VARIABLEN:\n"
    "        print(f'  {row[col]:>22.4f}', end='')\n"
    "    print()\n\n"
    "n_hist = 90\n"
    "fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)\n"
    "farben    = ['steelblue', 'darkorange', 'seagreen']\n"
    "einheiten = ['hPa', 'm/s', '°C']\n\n"
    "for ax, col, farbe, einheit in zip(axes, VARIABLEN, farben, einheiten):\n"
    "    hist = df_stat[col].iloc[-n_hist:]\n"
    "    ax.plot(hist.index, hist.values, color=farbe, lw=1.0, label='Historisch (90 Tage)')\n"
    "    ax.plot(fc_prog.index, fc_prog[col].values, color='red', lw=2,\n"
    "            marker='o', markersize=5, label='Prognose (10 Tage)')\n"
    "    ax.axvline(df_stat.index[-1], color='gray', ls=':', lw=1.5)\n"
    "    ax.set_ylabel(f'{col} [{einheit}]', fontsize=9)\n"
    "    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)\n"
    "    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))\n"
    "    ax.tick_params(axis='x', rotation=30)\n\n"
    "axes[0].set_title(f'VAR({p_final}) – 10-Tage-Prognose', fontsize=13)\n"
    "plt.tight_layout()\n"
    "plt.savefig(os.path.join(PLOTORDNER, 'VAR_08_prognose.png'), bbox_inches='tight')\n"
    "plt.show()\n"
))

# ── 26  Markdown Holdout ──────────────────────────────────────────────────────
cells.append(md("## 11b. HOLDOUT-EVALUATION: Prognose vs. tatsächliche Werte"))

# ── 27  Holdout ───────────────────────────────────────────────────────────────
cells.append(code(
    "print('=' * 70)\n"
    "print('HOLDOUT-EVALUATION (10-Tage-Prognose vs. Ist-Werte)')\n"
    "print('=' * 70)\n\n"
    "# Holdout ggf. differenzieren\n"
    "if d_VAR == 1:\n"
    "    df_ho_stat = pd.concat([df.iloc[[-1]], df_holdout]).diff().dropna()\n"
    "else:\n"
    "    df_ho_stat = df_holdout.copy()\n\n"
    "# Forecast-Index mit Holdout abgleichen\n"
    "fc_ho = fc_prog.reindex(df_ho_stat.index)\n\n"
    "print(f'  {\"Variable\":<32} {\"RMSE\":>8} {\"MAE\":>8} {\"MAPE\":>8}')\n"
    "print('  ' + '-' * 60)\n"
    "metriken_ho = {}\n"
    "for col in VARIABLEN:\n"
    "    ist  = df_ho_stat[col].dropna().values\n"
    "    prog = fc_ho[col].dropna().values\n"
    "    n    = min(len(ist), len(prog))\n"
    "    if n == 0:\n"
    "        continue\n"
    "    ist, prog = ist[:n], prog[:n]\n"
    "    rmse = np.sqrt(mean_squared_error(ist, prog))\n"
    "    mae  = mean_absolute_error(ist, prog)\n"
    "    with np.errstate(divide='ignore', invalid='ignore'):\n"
    "        mape = np.mean(np.abs((ist - prog) / np.where(ist == 0, np.nan, ist))) * 100\n"
    "    metriken_ho[col] = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}\n"
    "    print(f'  {col:<32} {rmse:>8.4f} {mae:>8.4f} {mape:>8.2f}')\n\n"
    "fig, axes = plt.subplots(3, 2, figsize=(14, 10),\n"
    "                          gridspec_kw={'width_ratios': [3, 1]})\n"
    "farben    = ['steelblue', 'darkorange', 'seagreen']\n"
    "einheiten = ['hPa', 'm/s', '°C']\n\n"
    "for row, (col, farbe, einheit) in enumerate(zip(VARIABLEN, farben, einheiten)):\n"
    "    ist_s  = df_ho_stat[col].values\n"
    "    prog_s = fc_ho[col].values\n"
    "    idx    = df_ho_stat.index\n"
    "    n      = min(len(ist_s), len(prog_s))\n"
    "    ist_s, prog_s, idx = ist_s[:n], prog_s[:n], idx[:n]\n\n"
    "    # Linienplot\n"
    "    ax = axes[row, 0]\n"
    "    ax.plot(idx, ist_s,  color=farbe, lw=2, marker='o', ms=6, label='Tatsächlich')\n"
    "    ax.plot(idx, prog_s, color='red',  lw=2, marker='o', ms=6, ls='--', label='Prognose')\n"
    "    for d, p, a in zip(idx, prog_s, ist_s):\n"
    "        ax.plot([d, d], [p, a], color='gray', lw=1, ls=':', alpha=0.7)\n"
    "    m = metriken_ho.get(col, {})\n"
    "    ax.set_title(f'{col} [{einheit}]  RMSE={m.get(\"RMSE\",0):.4f}  MAE={m.get(\"MAE\",0):.4f}', fontsize=9)\n"
    "    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)\n"
    "    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.'))\n"
    "    ax.tick_params(axis='x', rotation=30)\n\n"
    "    # Residuen-Balken\n"
    "    ax2 = axes[row, 1]\n"
    "    res = ist_s - prog_s\n"
    "    farben_bar = ['tomato' if r < 0 else 'steelblue' for r in res]\n"
    "    ax2.bar(range(len(res)), res, color=farben_bar, alpha=0.8)\n"
    "    ax2.axhline(0, color='black', lw=1)\n"
    "    ax2.set_title('Residuen', fontsize=9)\n"
    "    ax2.set_xlabel('Tag'); ax2.grid(True, alpha=0.3, axis='y')\n\n"
    "axes[0, 0].set_title(f'VAR({p_final}) – Holdout-Evaluation (10 Tage)', fontsize=11)\n"
    "plt.tight_layout()\n"
    "plt.savefig(os.path.join(PLOTORDNER, 'VAR_09_holdout.png'), bbox_inches='tight')\n"
    "plt.show()\n"
))

# ── 28  Markdown Zusammenfassung ──────────────────────────────────────────────
cells.append(md("## ZUSAMMENFASSUNG"))

# ── 29  Zusammenfassung ───────────────────────────────────────────────────────
cells.append(code(
    "print('=' * 70)\n"
    "print('ZUSAMMENFASSUNG – VAR-MODELL')\n"
    "print('=' * 70)\n"
    "print(f'''\n"
    "  Datensatz      : Würzburg, Station 5705\n"
    "  Variablen      : {', '.join(VARIABLEN)}\n"
    "  Zeitraum       : {df.index[0].date()} – {df.index[-1].date()}\n"
    "  Beobachtungen  : {len(df):,} Tageswerte  |  Holdout: 10 Tage\n\n"
    "  Integrationsordnung : d = {d_VAR}\n"
    "  Gewaaehltes Modell  : VAR({p_final})\n"
    "''')\n"
    "print('  --- Test-Set (15%, Multi-Step) ---')\n"
    "for col in VARIABLEN:\n"
    "    m = metriken_test.get(col, {})\n"
    "    print(f'  {col:<32}  RMSE={m.get(\"RMSE\",0):.4f}  MAE={m.get(\"MAE\",0):.4f}')\n"
    "print()\n"
    "print('  --- Holdout (10-Tage-Prognose) ---')\n"
    "for col in VARIABLEN:\n"
    "    m = metriken_ho.get(col, {})\n"
    "    print(f'  {col:<32}  RMSE={m.get(\"RMSE\",0):.4f}  MAE={m.get(\"MAE\",0):.4f}')\n"
    "print('\\n' + '=' * 70)\n"
))

# ── Notebook speichern ────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

with open(str(OUT), "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

n_md   = sum(1 for c in cells if c["cell_type"] == "markdown")
n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Erstellt : {OUT.name}")
print(f"Zellen   : {n_md} Markdown  +  {n_code} Code  =  {len(cells)} gesamt")
