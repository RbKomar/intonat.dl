# -*- coding: utf-8 -*-
import click
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from multiprocessing import Pool, Lock
import logging
import math
import os
import numpy as np
import librosa
import time
import json
from datetime import datetime

logs_format = "%(asctime)s: %(message)s"
handler = logging.FileHandler('logs//DataRetrievingLogs_' + datetime.now().strftime("%H_%M_%S") + '.log')
logging.basicConfig(format=logs_format, level=logging.INFO,
                    datefmt="%H:%M:%S")
logger = logging.getLogger()
logger.addHandler(handler)

DOB = {'black': '1943',
       'bowman': '1942',
       'bruce': '1956',
       'byrne': '1934',
       'cooke': '1908',
       'cronkite': '1916',
       'doyle': '1952',
       'dunne': '1955',
       'finucane': '1950',
       'gogan': '1938',
       'lawlor': '1962',
       'lockwood': '1916',
       'lynn': '1956',
       'magee': '1935',
       'neill': '1956',
       'bhriain': '1952',
       'nick': '1956',
       'odulaing': '1935',
       'plomley': '1914',
       'queen': '1926',
       'reagan': '1911',
       'suzy': '1956',
       'symon': '1956',
       'thatcher': '1925',
       'tony': '1956',
       'andrew': '1956'
       }


def get_person_age(name, path_person_recording):
    age = np.abs(int(DOB[name]) -
                 int(path_person_recording.lower().replace('.wav', '').replace('-', '_').split('_')[0]))
    return int(age)


# custom pool initialization in order to further lock use
def init_pool(l):
    global lock
    lock = l


def run_multiprocessing_on_loop_over_path(**kwargs):
    path = kwargs["looping_folder"]
    args = kwargs["args"]
    multiprocessing_func = kwargs["func"]
    multiprocessing_args = []
    l = Lock()
    for folder_path in os.listdir(path):
        args_in_loop = args.copy()
        args_in_loop.append(folder_path)
        multiprocessing_args.append(args_in_loop)  # list of parameters for the pool

    with Pool(initializer=init_pool, initargs=(l,)) as pool:
        pool.starmap(multiprocessing_func, multiprocessing_args)


def read_data(main_folder_path='TCDSA_main', num_segments=10, ):
    # enumerate is used for debugging and testing
    for i, folder_path in enumerate(os.listdir(main_folder_path)):
        # if i > 2:
        #     break
        read_all_recordings_of_one_person(main_folder_path, num_segments, folder_path,)


def read_all_recordings_of_one_person(database_path, num_segments, folder,):
    start_timer = time.perf_counter()
    name = folder
    logger.info("GET PERSON DATA | %s: starting data collection", name)
    folder_path = database_path + r"\\" + folder
    run_multiprocessing_on_loop_over_path(
        looping_folder=folder_path,
        args=[name, folder_path, num_segments, ],
        func=get_features_multiprocessing,)
    end_timer = time.perf_counter()
    logger.info("GET PERSON DATA | %s: finishing data collection with time: %0.2f s", name, end_timer-start_timer)


def get_features_multiprocessing(name, folder_path, num_segments, file_path):
    """
    num_segments: int - in deep learning we do need a lot of data so we will cut the audio into the segments
    the function written with help of
    https://youtu.be/szyGiObZymo
    """
    start_timer = time.perf_counter()
    mfcc_segments = []

    filename = folder_path + r'\\' + file_path
    age = get_person_age(name, file_path)
    logger.info("GET FEATURES | %s/%d: starting feature collection", name, age)
    signal, sr = librosa.load(filename)
    hop_length = 512
    n_fft = 2048

    duration = librosa.get_duration(signal, sr=sr)
    num_samples_per_segment = int((sr * duration) / num_segments)
    expected_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # segment extracting mfcc and spectogram
    for s in range(num_segments):

        start_sample = num_samples_per_segment * s
        finish_sample = start_sample + num_samples_per_segment
        mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                    n_fft=n_fft,
                                    hop_length=hop_length,
                                    n_mfcc=15)
        mfcc = mfcc.T
        if len(mfcc) == expected_mfcc_vectors_per_segment:
            # mfcc_mean = np.mean(MFFCs, axis=1).tolist()
            # mfcc_std = np.std(MFFCs, axis=1).tolist()
            # mfcc_variance = np.var(MFFCs, axis=1).tolist()
            mfcc_segments.append(mfcc.tolist())

    lock.acquire()

    with open("data.json", "r+") as fp:
        try:
            data = json.load(fp)
        except json.decoder.JSONDecodeError as jde:
            data = {
                "name": [],
                "age": [],
                "mfcc": [],
            }
        data["name"].append(name)
        data["age"].append(age)
        data["mfcc"].append(mfcc_segments)
        fp.seek(0)  # file pointer to '0' position
        json.dump(data, fp, indent=4)
        fp.truncate()  # clearing all data existing behind fp position after writing to the file

    lock.release()
    end_timer = time.perf_counter()
    logger.info("GET FEATURES | %s/%d: finishing feature collection with time: %0.2f s", name, age, end_timer-start_timer)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
