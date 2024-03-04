import pandas as pd

def get_data(archivo_path):
    df = pd.read_csv(archivo_path)
    df = df.drop(df[df['age'] > 99].index)
    df_age = df[['age', 'pixels']]

    for index, row in df_age.iterrows():
        df_age.loc[index, 'age'] = int(df_age.loc[index, 'age'] * 20 / 100)

    return df_age