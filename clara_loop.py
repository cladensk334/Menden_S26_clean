for name, splits in zeitreihen_split.items():
    
    ts_train = splits['train'].interpolate(method='linear').dropna()
    ts_val   = splits['val'].interpolate(method='linear').dropna()
    ts_test  = splits['test'].interpolate(method='linear').dropna()
    
    # Integrationsordnung nur auf Trainingsdaten bestimmen
    adf = adfuller(ts_train, regression='c', autolag='AIC')
    d   = 0 if adf[1] < 0.05 else 1
    
    modell_ergebnisse = []
    
    for p in range(0, 4):
        for q in range(0, 4):
            try:
                # Modell auf Trainingsdaten schätzen
                m = ARIMA(ts_train, order=(p, d, q)).fit()
                
                # Prognose auf Validierungsdaten
                fc      = m.get_forecast(steps=len(ts_val))
                fc_mean = fc.predicted_mean.values
                actual  = ts_val.values[:len(fc_mean)]
                
                # Metriken auf Validierungsdaten berechnen
                mse  = np.mean((actual - fc_mean)**2)
                rmse = np.sqrt(mse)
                mae  = np.mean(np.abs(actual - fc_mean))
                mape = np.mean(np.abs((actual - fc_mean) / actual)) * 100
                
                modell_ergebnisse.append({
                    'Zeitreihe': name,
                    'p': p, 'd': d, 'q': q,
                    'AIC':  m.aic,
                    'BIC':  m.bic,
                    'MSE':  round(mse,  4),
                    'RMSE': round(rmse, 4),
                    'MAE':  round(mae,  4),
                    'MAPE': round(mape, 4),
                })
            except Exception:
                pass
    
    # Bestes Modell nach RMSE auf Validierungsdaten
    df_mod = pd.DataFrame(modell_ergebnisse).sort_values('RMSE')
    beste  = df_mod.iloc[0]
    print(f"\n{name} – Bestes Modell: ARIMA({int(beste.p)},{int(beste.d)},{int(beste.q)})")
    print(f"  RMSE={beste.RMSE}, MAE={beste.MAE}, MAPE={beste.MAPE}%")
    
    # Finale Bewertung auf Testdaten
    bestes_modell = ARIMA(ts_train, order=(int(beste.p), int(beste.d), int(beste.q))).fit()
    fc_test       = bestes_modell.get_forecast(steps=len(ts_val) + len(ts_test))
    fc_test_mean  = fc_test.predicted_mean.values[-len(ts_test):]
    actual_test   = ts_test.values[:len(fc_test_mean)]
    
    rmse_test = np.sqrt(np.mean((actual_test - fc_test_mean)**2))
    mae_test  = np.mean(np.abs(actual_test - fc_test_mean))
    print(f"  Finale Testbewertung → RMSE={rmse_test:.4f}, MAE={mae_test:.4f}")
    
    beste_modelle[name] = {
        'modell':   bestes_modell,
        'ts_train': ts_train,
        'ts_val':   ts_val,
        'ts_test':  ts_test,
        'p': int(beste.p), 'd': int(beste.d), 'q': int(beste.q)
    }
    
    alle_ergebnisse.extend(modell_ergebnisse)