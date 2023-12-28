import pandas
import tensorflow as tf
from keras.models import load_model


def convert_dataset():
    filepath = "data/train"

    dataframe = pandas.read_xml(filepath + '.xml')

    dataframe.drop(columns=['movie', 'review_id', 'name', 'user_id'], inplace=True)

    dataframe['commentaire'].fillna(" ", inplace=True)
    dataframe['commentaire'].apply(lambda comment: comment.lower())

    if 'note' in dataframe.columns:
        dataframe['note'].fillna("0,5", inplace=True)
        dataframe['note'].apply(lambda note: float(note.replace(',', '.')))
        dataframe.rename(columns={"note": "label"}, inplace=True)

    dataframe.rename(columns={"commentaire": "sentence1"}, inplace=True)
    # Sauvegarder le DataFrame modifié dans un nouveau fichier CSV
    dataframe.to_csv(filepath + '.csv', index=False)

if __name__ == '__main__':
    convert_dataset()
