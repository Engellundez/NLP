import pandas as pd
from loguru import logger as log


from toolkit import set_same_train_and_test
from models.basic import Basic
from models.lemma import Lemma
from models.steam import Steam

if __name__ == '__main__':
    log.info('Creando el dataframe del dataset')
    dataset = pd.read_csv('Ejemplo_tokenizacion/df_total.csv', encoding='UTF-8')

    log.info('Separando dataset')
    list_datasets = set_same_train_and_test(dataset, 'news', 'Type')
    log.info('Dataset estaplecido y separado')

    log.info('Train and load Model Basic')
    score_model_basic = Basic(list_datasets)
    log.success(f"the basic model {score_model_basic}")
    log.info("Train and load Model Steam")
    score_model_steam = Steam(list_datasets)
    log.success(f"the steam model {score_model_steam}")
    log.info("Train and load Model Lemma")
    score_model_lemma = Lemma(list_datasets)
    log.success(f"the steam model {score_model_lemma}")
