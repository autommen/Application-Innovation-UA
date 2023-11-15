import pandas as pd
    
import csv

if __name__ == '__main__':
    #document = pandas.read_xml('data/dev.xml')
    #document.to_csv('data/dev.csv', index=False)
    
    """train = pandas.read_xml('data/train.xml')
    train = train.head(100400)
    train.to_csv('data/train_tronc.csv', index=False)"""

    dictUser = {}
    count = 0

    with open('data/dev.csv', 'r', encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            print(row)


     # Charger le fichier CSV dans un DataFrame
    df = pd.read_csv('data/dev.csv')
    
    
    # Parcourir les lignes du DataFrame
    for index, ligne in df.iterrows():
        # Lire la valeur actuelle dans la colonne spécifiée
        valeur_actuelle = ligne["user_id"]
        
        if valeur_actuelle not in dictUser :
            dictUser[valeur_actuelle] = count
            count = count + 1
        
        df.at[index, "user_id"] = dictUser[valeur_actuelle]
        #print(ligne['commentaire'])
        #df.at[index, "len_comment"] = len(ligne['commentaire'])

    
    # Sauvegarder le DataFrame modifié dans un nouveau fichier CSV
    df.to_csv('data/dev_changeUser_rajoutLen.csv', index=False)
    
