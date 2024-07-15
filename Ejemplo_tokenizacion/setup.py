import pandas as pd

from toolkit import set_same_train_and_test
from models.basic import Basic
from models.lemma import Lemma
from models.steam import Steam

if __name__ == '__main__':
    dataset = pd.read_csv('Ejemplo_tokenizacion/df_total.csv', encoding='UTF-8')

    list_datasets = set_same_train_and_test(dataset,'news', 'Type')

    # score_model_basic = Basic(list_datasets)
    score_model_steam = Steam(list_datasets)
    # score_model_lemma = Lemma(list_datasets)

    # print(score_model_basic)
