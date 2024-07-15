import pandas as pd
from loguru import logger as log
import time

from toolkit import set_same_train_and_test
from models.basic import Basic
from models.lemma import Lemma
from models.steam import Steam

if __name__ == '__main__':
    log.info('Creando el dataframe del dataset')
    # dataset = pd.read_csv('Ejemplo_tokenizacion/data_raw.csv', encoding='UTF-8')
    dataset = pd.read_csv('Ejemplo_tokenizacion/spanish_phrases.csv', encoding='UTF-8')
    # dataset = pd.read_csv('Ejemplo_tokenizacion/df_total.csv', encoding='UTF-8')

    log.info('Separando dataset')
    # list_datasets = set_same_train_and_test(dataset, 'traducido', 'class')
    list_datasets = set_same_train_and_test(dataset, 'phrase', 'Type')
    # list_datasets = set_same_train_and_test(dataset, 'news', 'Type')
    log.info('Dataset estaplecido y separado')

    log.info('Train and load Model Basic')
    start_time = time.time()
    score_model_basic = Basic(list_datasets)
    final_time = time.time() - start_time
    log.success(f"the traine times is {final_time} for the basic model {score_model_basic}")

    log.info("Train and load Model Steam")
    start_time = time.time()
    score_model_steam = Steam(list_datasets)
    final_time = time.time() - start_time
    log.success(f"the traine times is {final_time} for the steam model {score_model_steam}")

    log.info("Train and load Model Lemma")
    start_time = time.time()
    score_model_lemma = Lemma(list_datasets)
    final_time = time.time() - start_time
    log.success(f"the traine times is {final_time} for the steam model {score_model_lemma}")
