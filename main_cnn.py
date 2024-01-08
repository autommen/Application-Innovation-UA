import csv
import sys
import pandas
import os
import numpy as np
import tensorflow as tf
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Embedding
from keras import models, Input

from pandas import DataFrame
import string

from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.src.saving import load_model

tf.config.set_visible_devices([], 'GPU')

vocabulary_size = 3000
compteur = [0]
EMBEDDING_DIM = 100
max_length = 20

path = 'models/mots_choisis.txt'
path_tsv = './models/cnn/vectors.tsv'
path_train = './data/train'

mots_choisis = './models/cnn/mots_choisis.txt'


def get_dataframe(filepath: str) -> DataFrame:
    if not os.path.exists(filepath + '.csv'):
        dataframe = pandas.read_xml(filepath + '.xml')

        dataframe['commentaire'].fillna(" ", inplace=True)
        dataframe['commentaire'] = dataframe['commentaire'].str.replace(f"[{string.punctuation}]", '')
        dataframe['commentaire'].apply(lambda comment: comment.lower())
        # dataframe['commentaire'] = dataframe['commentaire'].apply(remove_stopwords)

        if 'note' in dataframe.columns:
            dataframe['note'].fillna("0,5", inplace=True)
            dataframe['note'].apply(lambda note: float(note.replace(',', '.')))

        # Sauvegarder le DataFrame modifié dans un nouveau fichier CSV
        dataframe.to_csv(filepath + '.csv', index=False)
    else:
        dataframe = pandas.read_csv(filepath + '.csv')
    return dataframe


"""def dataset(data):
    document['commentaire'] = document['commentaire'].apply(lambda commentaire: process_sentence(commentaire.split(), dict_mots_choisis))
"""


def process_commentaire(row, max_length, embedding_layer, sentences_train, compteur):
    sentence = row
    difference = max_length - len(sentence)
    sentence.extend([vocabulary_size] * difference)
    if len(sentence) != max_length:
        sentence = sentence[:max_length]
    # print(np.array(sentence))
    sentence = embedding_layer(np.array(sentence))
    sentences_train.append(sentence)
    compteur[0] = compteur[0] + 1


def process_sentence(commentaire, dict_mots_choisis):
    sentence = [dict_mots_choisis[word] - 1 for word in commentaire if word in dict_mots_choisis]
    return sentence


dict_mots_choisis = {}

with open(mots_choisis, 'r', encoding='utf-8') as mots:
    count = 1
    for line in mots:
        if len(line.strip()) != 0:
            dict_mots_choisis[line.strip()] = count
            count = count + 1
            if count == vocabulary_size + 1:
                break

print(len(dict_mots_choisis))

# Trouver la nouvelle valeur de max_length après l'application
# max_length = document['commentaire'].apply(len).max()
max_length = 50

embedding_matrix = np.zeros((vocabulary_size + 1, EMBEDDING_DIM))

with open(path_tsv, 'r', newline='', encoding='utf-8') as file:
    tsv_reader = csv.reader(file, delimiter='\t')

    count_lines = 0
    for row in tsv_reader:
        embedding_matrix[count_lines] = [float(value) for value in row[:EMBEDDING_DIM]]
        count_lines = count_lines + 1
        if count_lines == vocabulary_size:
            break
    embedding_matrix[vocabulary_size] = [0] * EMBEDDING_DIM

# va de 0 à 2999
# 3000 pour le padding
embedding_layer = Embedding(vocabulary_size + 1,
                            EMBEDDING_DIM,
                            #input_length=max_length,
                            #trainable=False
                            )
embedding_layer.build()
embedding_layer.set_weights([embedding_matrix])
document = get_dataframe(path_train)

# Appliquer la fonction process_sentence à chaque élément de la colonne 'commentaire'
document['commentaire'] = document['commentaire'].apply(
    lambda commentaire: process_sentence(commentaire.split(), dict_mots_choisis))

sentences_train = []

# Appliquer la fonction process_row à chaque ligne de la colonne 'commentaire'
document['commentaire'].apply(
    lambda row: process_commentaire(row, max_length, embedding_layer, sentences_train, compteur))

print(np.array(compteur).mean())

label_encoder = LabelEncoder()
# labels_encoded = label_encoder.fit_transform(labels_train)
labels_encoded = label_encoder.fit_transform(document['note'])

labels_one_hot = to_categorical(labels_encoded, 10)


def train():
    model = models.Sequential()
    model.add(Input(shape=(max_length, EMBEDDING_DIM)))
    model.add(Conv1D(EMBEDDING_DIM, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    model.fit(np.array(sentences_train), labels_one_hot, epochs=20, batch_size=32)
    model.save('models/model_cnn.keras')


def test():
    model = load_model('models/model_cnn.keras')
    path_test = './data/test'
    test = get_dataframe(path_test)

    sentences_test = []

    test['commentaire'] = test['commentaire'].apply(
        lambda commentaire: process_sentence(commentaire.split(), dict_mots_choisis))
    test['commentaire'].apply(
        lambda row: process_commentaire(row, max_length, embedding_layer, sentences_test, compteur))

    print(len(sentences_test))

    predictions = model.predict(np.array(sentences_test))
    print(predictions)

    predicted_indices = np.argmax(predictions, axis=1)

    # Obtenez la vraie classe à partir de l'indice
    true_classes = label_encoder.inverse_transform(predicted_indices)

    # Affichez les vraies classes
    print(true_classes)

    render = pandas.DataFrame({'review_id': test['review_id'], 'note': true_classes})

    render.to_csv("./data/render.txt", sep=" ", header=False, index=False)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        print("Testing model...")
        path = 'data/test'
    else:
        print("Training model...")
        path = 'data/dev'

    if len(sys.argv) == 2:
        test()
    else:
        train()
