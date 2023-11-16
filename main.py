import pandas
import os

import sklearn
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import DataFrame


def get_dataframe(filepath: str) -> DataFrame:
    if not os.path.exists(filepath + '.csv'):
        dataframe = pandas.read_xml(filepath + '.xml')

        # Parcourir les lignes du DataFrame
        for index, row in dataframe.iterrows():
            dataframe.at[index, "note"] = float(row["note"].replace(',', '.'))

        # Sauvegarder le DataFrame modifié dans un nouveau fichier CSV
        dataframe.to_csv(filepath + '.csv', index=False)
    else:
        dataframe = pandas.read_csv(filepath + '.csv')
    return dataframe


if __name__ == '__main__':
    path = 'data/dev'
    document = get_dataframe(path)

    text_data = document['commentaire']
    text_data = text_data.fillna('')

    tfidf_vectorizer = TfidfVectorizer()
    x_tfidf = tfidf_vectorizer.fit_transform(text_data)

    other_features = document.drop(['note', 'commentaire'], axis=1)
    x_combined = pandas.concat([other_features, pandas.DataFrame(x_tfidf.toarray())], axis=1)

    x = x_combined.columns.astype(str)
    y = document['note']

    if len(x) == len(y):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.25, random_state=42)

        svc = svm.SVC(verbose=True)
        svc.fit(x_train, y_train)
        y_pred = svc.predict(x_test)
        accuracy = sklearn.metrics.accuracy_score(y_test, y_pred) * 100
        print("Accuracy: {:.2f}%".format(accuracy))
    else:
        print("Error: Inconsistent number of samples between x ("+str(len(x))+") and y ("+str(len(y))+").")

