import pandas as pd
import os


# pour elasticsearch

def processing(document):
    dictUser = {}
    count = 0

    for index, row in document.iterrows():
        # Lire la valeur actuelle dans la colonne spécifiée
        user_id = row["user_id"]
        note = row["note"]

        if user_id not in dictUser:
            dictUser[user_id] = count
            count = count + 1

        document.at[index, "user_id"] = dictUser[user_id]
        document.at[index, "note"] = float(note.replace(',', '.'))
        document.at[index, "len_comment"] = len(str(row['commentaire']))

    return document


if __name__ == '__main__':
    if not os.path.exists('data/dev.csv'):
        document = pd.read_xml('data/dev.xml')

        document = processing(document)

        # Sauvegarder le DataFrame modifié dans un nouveau fichier CSV
        document.to_csv('data/dev.csv', index=False)

    if not os.path.exists('data/train.csv'):
        document = pd.read_xml('data/train.xml')
        document = document.head(100400)

        document = processing(document)

        document.to_csv('data/train.csv', index=False)
