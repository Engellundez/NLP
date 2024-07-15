import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from loguru import logger as log


class Lemma:
    def __init__(self, list_datasets) -> None:
        self.X_train = list_datasets[0]
        self.X_test = list_datasets[1]
        self.y_train = list_datasets[2]
        self.y_test = list_datasets[3]
        self.score = 0.0
        self.nlp = spacy.load('es_core_news_sm')
        self.stop_words = set(stopwords.words('spanish'))
        self.cv = CountVectorizer()
        self._train_model()

    def __str__(self) -> str:
        return f'Score: {self.score} con el modelo de entrenamiento de matriz {self.X_train_transformed}'

    def _train_model(self):
        self._vectorizer_X()
        self.model = MultinomialNB()
        self.model.fit(self.X_train_transformed, self.y_train)
        self._return_score()

    def _vectorizer_X(self):
        self.X_train = self.X_train.apply(self._process_text)
        self.X_test = self.X_test.apply(self._process_text)

        self.X_train_transformed = self.cv.fit_transform(self.X_train)
        self.X_test_transformed = self.cv.transform(self.X_test)

    def _return_score(self):
        y_pred = self.model.predict(self.X_test_transformed)
        self.score = metrics.accuracy_score(self.y_test, y_pred)

    def _process_text(self, text=''):
        # Filtramos las palabras de parada
        tokens = self.nlp(text.lower())
        filter_tokens = [token.lemma_ for token in tokens if token.lemma_ not in self.stop_words]
        lemmas = [token for token in filter_tokens if token.isalpha]
        return ' '.join(lemmas)
