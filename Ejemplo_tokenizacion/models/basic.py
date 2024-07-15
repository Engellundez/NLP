from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


class Basic:
    def __init__(self, list_datasets) -> None:
        self.X_train = list_datasets[0]
        self.X_test = list_datasets[1]
        self.y_train = list_datasets[2]
        self.y_test = list_datasets[3]
        self.cv = CountVectorizer()
        self._train_model()

    def __str__(self) -> str:
        return f'Score: {self.score}'

    def _train_model(self):
        self._vectorizer_X()
        self.model = MultinomialNB()
        self.model.fit(self.X_train_transformed, self.y_train)
        self._return_score()

    def _vectorizer_X(self):
        self.X_train_transformed = self.cv.fit_transform(self.X_train)
        self.X_test_transformed = self.cv.transform(self.X_test)

    def _return_score(self):
        y_pred = self.model.predict(self.X_test_transformed)
        self.score = metrics.accuracy_score(self.y_test, y_pred)
