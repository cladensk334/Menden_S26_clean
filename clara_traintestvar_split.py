#Train / Validation / Test Split
n = len(df)

train_end = int(n * 0.70)
val_end   = int(n * 0.85)

df_train = df.iloc[:train_end]
df_val   = df.iloc[train_end:val_end]
df_test  = df.iloc[val_end:]

print(f"Gesamte Beobachtungen : {n}")
print(f"Train      : {len(df_train)} ({df_train.index[0].date()} – {df_train.index[-1].date()})")
print(f"Validation : {len(df_val)}  ({df_val.index[0].date()} – {df_val.index[-1].date()})")
print(f"Test       : {len(df_test)}  ({df_test.index[0].date()} – {df_test.index[-1].date()})")

# Zeitreihen ebenfalls aufteilen
zeitreihen_split = {
    'Temperatur': {'train': df_train['TMK'], 'val': df_val['TMK'], 'test': df_test['TMK']},
    'Wind':       {'train': df_train['FM'],  'val': df_val['FM'],  'test': df_test['FM']},
    'Luftdruck':  {'train': df_train['PM'],  'val': df_val['PM'],  'test': df_test['PM']},
}