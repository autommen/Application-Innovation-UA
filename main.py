import pandas
import os

import sklearn
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder


def get_dataframe(filepath: str) -> DataFrame:
    if not os.path.exists(filepath + '.csv'):
        dataframe = pandas.read_xml(filepath + '.xml')

        dataframe['note'].apply(lambda note: float(note.replace(',', '.')))

        # Sauvegarder le DataFrame modifié dans un nouveau fichier CSV
        dataframe.to_csv(filepath + '.csv', index=False)
    else:
        dataframe = pandas.read_csv(filepath + '.csv')
    return dataframe


if __name__ == '__main__':
    path = 'data/train'
    document = get_dataframe(path).dropna()

    text_data = document['commentaire']

    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('french'), max_features=4000)
    x_tfidf = tfidf_vectorizer.fit_transform(text_data.values)

    #other_features = document.drop(['note', 'commentaire'], axis=1)
    #x_combined = pandas.concat([other_features, pandas.DataFrame(x_tfidf.toarray())], axis=1)
    #x = x_combined.columns.astype(str)

    x = pandas.DataFrame(x_tfidf.toarray())
    y = document['note']

    lab = LabelEncoder()
    y = lab.fit_transform(y)

    print(x,y)

    if len(x) == len(y):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.25, random_state=42)
        svc = svm.LinearSVC(verbose=True)
        svc.fit(x_train, y_train)
        y_pred = svc.predict(x_test)
        print(classification_report(y_test, y_pred))
    else:
        print("Error: Inconsistent number of samples between x ("+str(len(x))+") and y ("+str(len(y))+").")

