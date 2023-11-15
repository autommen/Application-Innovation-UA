import pandas as pd
    
import csv

if __name__ == '__main__':
    """document = pd.read_xml('data/dev.xml')
    document.to_csv('data/dev.csv', index=False)
    
    train = pd.read_xml('data/train.xml')
    train = train.head(100400)
    train.to_csv('data/train_tronc.csv', index=False)"""

    dictUser = {}
    count = 0


     # Charger le fichier CSV dans un DataFrame
    df = pd.read_csv('data/dev.csv')
    
    
    # Parcourir les lignes du DataFrame
    for index, ligne in df.iterrows():
        # Lire la valeur actuelle dans la colonne spécifiée
        valeur_actuelle = ligne["user_id"]
        valeur_actuelle_note = ligne["note"]
        
        if valeur_actuelle not in dictUser :
            dictUser[valeur_actuelle] = count
            count = count + 1
        
        df.at[index, "user_id"] = dictUser[valeur_actuelle]
        df.at[index, "note"] = float(valeur_actuelle_note.replace(',', '.'))
        df.at[index, "len_comment"] = len(str(ligne['commentaire']))

    
    # Sauvegarder le DataFrame modifié dans un nouveau fichier CSV
    df.to_csv('data/dev_changeUser_changeNote.csv', index=False)
    
