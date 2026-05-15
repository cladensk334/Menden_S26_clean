#Train/Test Split

splits = {}
for name, frame in [('Temperatur', df_temp), ('Wind', df_wind), ('Luftdruck', df_druck)]:
    train = frame.iloc[:-TEST_TAGE]
    test  = frame.iloc[-TEST_TAGE:]

    splits[name] = {'train': train, 'test': test}

print('\n=== Train / Test Split ===')
for name, s in splits.items():
    print(f"  {name}:")
    print(f"    Train : {len(s['train'])} Tage ({s['train'].index[0].date()} – {s['train'].index[-1].date()})")
    print(f"    Test  : {len(s['test'])} Tage ({s['test'].index[0].date()} – {s['test'].index[-1].date()})")