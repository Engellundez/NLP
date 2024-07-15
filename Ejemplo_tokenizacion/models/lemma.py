import spacy


class Lemma():
    def __init__(self, X_train, X_test, y_train, y_test) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.nlp = spacy.load('es_core_news_sm')
        self._train_model_lemma()

    def _train_model_lemma(self):
        self.X_lemma_transformed_train = []
        for train in self.X_train:
            self.X_lemma_transformed_train.append(self._lemmatize_text(train))

        print(self.X_train, self.X_lemma_transformed_train)

    def _lemmatize_text(self, text=''):
        doc = self.nlp(text.lower())
        lemmas = [token.lemma_ for token in doc if token.is_alpha]
        return ' '.join(lemmas)
