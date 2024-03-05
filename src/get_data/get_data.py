import pandas as pd


def get_data(archivo_path, type='age'):
    df = pd.read_csv(archivo_path)
    df_to_send = []

    if type == 'age':
        df = df.drop(df[df['age'] > 99].index)
        df_to_send = df[[type, 'pixels']]
        for index, row in df_to_send.iterrows():
            df_to_send.loc[index, 'age'] = int(df_to_send.loc[index, 'age'] * 20 / 100)
    elif type == 'gender':
        df_to_send = df[[type, 'pixels']]
        # print(df_to_send)
        # # Convertir la columna 'gender' en columnas dummy
        # gender_dummies = pd.get_dummies(df_to_send['gender'], prefix=['male', 'female'])
        #
        # # Concatenar las columnas dummy con el DataFrame original
        # df_to_send = pd.concat([gender_dummies, df_to_send['pixels']], axis=1)
        #
        # print(df_to_send)

    return df_to_send
