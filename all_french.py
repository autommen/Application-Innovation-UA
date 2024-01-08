import pandas
import os

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

import langid


def get_dataframe(filepath: str) -> DataFrame:
    if not os.path.exists(filepath + '.csv'):
        dataframe = pandas.read_xml(filepath + '.xml')

        dataframe['commentaire'].fillna(" ", inplace=True)
        dataframe['commentaire'].apply(lambda comment: comment.lower())

        if 'note' in dataframe.columns:
            dataframe['note'].fillna("0,5", inplace=True)
            dataframe['note'].apply(lambda note: float(note.replace(',', '.')))

        # Sauvegarder le DataFrame modifié dans un nouveau fichier CSV
        dataframe.to_csv(filepath + '.csv', index=False)
    else:
        dataframe = pandas.read_csv(filepath + '.csv')
    return dataframe


def prepare_data(document: DataFrame, tfidf_vectorizer=None, label_encoder=None):
    text_data = document['commentaire']

    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('french'), max_features=3000,
                                           strip_accents='unicode')
        x_tfidf = tfidf_vectorizer.fit_transform(text_data.values)
    else:
        x_tfidf = tfidf_vectorizer.transform(text_data.values)

    x = pandas.DataFrame(x_tfidf.toarray())
    if label_encoder is None:
        y = document['note']
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        return ((x, y), (tfidf_vectorizer, label_encoder))
    else:
        return x


def detect_language(comment):
    lang, _ = langid.classify(comment)
    return lang


if __name__ == '__main__':
    path = 'data/train'
    document = get_dataframe(path)

    # Filtrer les commentaires qui sont reconnus comme français
    all_french = document[document['commentaire'].apply(lambda x: detect_language(x) == 'fr')]

    # Afficher le DataFrame résultant
    print(all_french)

    (x, y), (tfidf_vectorizer, label_encoder) = prepare_data(all_french)

    print(x, y)

    if len(x) == len(y):
        svc = svm.LinearSVC(verbose=True)

        svc.fit(x, y)

        document_test = get_dataframe('data/test')

        x_test = prepare_data(document_test, tfidf_vectorizer, label_encoder)

        y_pred = label_encoder.inverse_transform(svc.predict(x_test))
        print(x_test, y_pred)

        render = pandas.DataFrame({'review_id': document_test['review_id'], 'note': y_pred})

        render.to_csv("data/render.txt", sep=" ", header=False, index=False)
    else:
        print("Error: Inconsistent number of samples between x (" + str(len(x)) + ") and y (" + str(len(y)) + ").")
