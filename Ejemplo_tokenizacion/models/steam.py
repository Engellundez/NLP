from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('punkt')
nltk.download('stopwords')


class Steam:
    def __init__(self, list_datasets) -> None:
        self.X_train = list_datasets[0]
        self.X_test = list_datasets[1]
        self.y_train = list_datasets[2]
        self.y_test = list_datasets[3]
        self.score = 0.0
        self.stop_words = set(stopwords.words('spanish'))
        self.cv = CountVectorizer()
        # Es importante mantener el argumento en minúsculas
        self.stemmer = SnowballStemmer('spanish')
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
        # Tokenizer el texto en minúsculas
        tokens = word_tokenize(text.lower())
        # Filtramos las palabras de parada
        filter_tokens = [word for word in tokens if word not in self.stop_words]
        # aplicar steaming
        stems = [self.stemmer.stem(token)
                 for token in filter_tokens if token.isalpha()]

        return ' '.join(stems)
