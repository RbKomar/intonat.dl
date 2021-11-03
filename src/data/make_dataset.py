# -*- coding: utf-8 -*-
import click
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from multiprocessing import Pool, Lock
import numpy as np
import logging
import math
import os
import librosa
import time
import json
from datetime import datetime

logs_format = "%(asctime)s: %(message)s"
handler = logging.FileHandler('logs//DataRetrievingLogs_' + datetime.now().strftime("%H_%M_%S") + '.log')
logging.basicConfig(format=logs_format, level=logging.INFO,
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
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


def read_data(main_folder_path='TCDSA_main', ):
    # enumerate is used for debugging and testing
    for i, folder_path in enumerate(os.listdir(main_folder_path)):
        #if i > 2:
        #    break
        read_all_recordings_of_one_person(main_folder_path, folder_path,)


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


def read_all_recordings_of_one_person(database_path, folder):
    start_timer = time.perf_counter()
    name = folder
    logger.info("GET PERSON DATA | %s: starting data collection", name)
    folder_path = database_path + r"\\" + folder
    run_multiprocessing_on_loop_over_path(
        looping_folder=folder_path,
        args=[name, folder_path],
        func=get_features_multiprocessing,)
    end_timer = time.perf_counter()
    logger.info("GET PERSON DATA | %s: finishing data collection with time: %0.2f s", name, end_timer-start_timer)


def HNR_RJT(signal, sr, n_fft):
    """
    HNR extraction -> https://www.scitepress.org/Papers/2009/15529/15529.pdf
    A NEW ACCURATE METHOD OF HARMONIC-TO-NOISERATIO EXTRACTION
    Ricardo J. T. de Sousa - School of Engineering , University of Porto, Rua Roberto Frias, Porto, Portugal
    Robert Komar implementation 2021
    """
    h_range = n_fft//2
    s = np.abs(librosa.stft(signal, n_fft=n_fft))
    fft_freqs = librosa.fft_frequencies(sr=sr)
    s_harm = librosa.interp_harmonics(s, fft_freqs, range(h_range), axis=0)
    noise_spec = s[h_range::] - s_harm
    return 10*np.log(np.sum(s_harm**2) / np.sum(noise_spec**2))


def HNR(signal, sr, pitch_period):
    # harmonics-to-noise ratio implementation with help of
    # https://github.com/eesungkim/Speech_Emotion_Recognition_DNN-ELM/blob/master/utils/speech_features.py
    def autocorrelation(s):
        x = s-np.mean(s)
        correlation = np.correlate(x, x, mode='ful') / np.sum(x**2)
        n = len(correlation)//2
        return correlation[n::]

    t = int(sr*pitch_period)
    acf = autocorrelation(signal)
    t0 = acf[0]
    t1 = acf[t]

    return 10*np.log(np.abs(t1/(t0-t1)))


def get_features_multiprocessing(name, folder_path, file_path):
    """
    num_segments: int - in deep learning we do need a lot of data so we will cut the audio into the segments
    the function written with help of https://youtu.be/szyGiObZymo
    """
    start_timer = time.perf_counter()
    mfcc_segments = []
    delta_mfcc_segments = []
    delta2_mfcc_segments = []
    fundamental_frequency_segments = []
    hnr_segments = []

    filename = folder_path + r'\\' + file_path
    age = get_person_age(name, file_path)
    logger.info("GET FEATURES | %s/%d: starting feature collection", name, age)
    signal, sr = librosa.load(filename)
    hop_length = 512
    n_fft = 2048

    duration = librosa.get_duration(signal, sr=sr)
    # min divide recording into 10 segments and max into 100 -> min(100, max(10,g(x)))
    # g(x) = 1/6 * recording time -> 60 seconds recording gives 10 segments
    num_segments = min(max(10, math.ceil(1/6 * duration)), 100)
    num_samples_per_segment = int((sr * duration) / num_segments)
    expected_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # segment extracting to get more features
    for s in range(num_segments):
        start_sample = num_samples_per_segment * s
        finish_sample = start_sample + num_samples_per_segment
        sampled_signal = signal[start_sample:finish_sample]

        fundamental_frequency, _, _ = librosa.pyin(sampled_signal,
                                                   fmin=librosa.note_to_hz('C2'),
                                                   fmax=librosa.note_to_hz('C7'),
                                                   hop_length=hop_length,)
        fundamental_frequency_segments.append(fundamental_frequency.tolist())

        # getting rid of nans from f0
        fundamental_frequency = fundamental_frequency[~np.isnan(fundamental_frequency)]
        # pitch period is inverse f0
        pitch_period = 1./np.mean(fundamental_frequency)
        hnr = HNR(sampled_signal, sr, pitch_period)
        hnr_segments.append(hnr)

        mfcc = librosa.feature.mfcc(sampled_signal,
                                    n_fft=n_fft,
                                    hop_length=hop_length,
                                    n_mfcc=15)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        mfcc = mfcc.T
        delta_mfcc = delta_mfcc.T
        delta2_mfcc = delta2_mfcc.T

        if len(mfcc) == expected_vectors_per_segment:
            mfcc_segments.append(mfcc.tolist())
            delta_mfcc_segments.append(delta_mfcc.tolist())
            delta2_mfcc_segments.append(delta2_mfcc.tolist())

    lock.acquire()
    with open(r"D:\PROJEKTY\intonat.dl\data\interim\data.json", "r+") as fp:
        try:
            data = json.load(fp)
        except json.decoder.JSONDecodeError as jde:
            data = {
                "name": [],
                "sr": [],
                "age": [],
                "fundamental_frequency": [],
                "mfcc": [],
                "delta_mfcc": [],
                "delta2_mfcc": [],
                "hnr": [],
            }
        data["name"].append(name)
        data["sr"].append(sr)
        data["age"].append(age)
        data["mfcc"].append(mfcc_segments)
        data["delta_mfcc"].append(delta_mfcc_segments)
        data["delta2_mfcc"].append(delta2_mfcc_segments)
        data["fundamental_frequency"].append(fundamental_frequency_segments)
        data["hnr"].append(hnr_segments)

        fp.seek(0)  # file pointer to '0' position
        json.dump(data, fp, indent=4)
        fp.truncate()  # clearing all data existing behind fp position after writing to the file
    lock.release()

    end_timer = time.perf_counter()
    logger.info("GET FEATURES | %s/%d: finishing feature collection with time: %0.2f s",
                name, age, end_timer-start_timer)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True), default=r"D:\PROJEKTY\intonat.dl\data\raw\TCDSA_main")
@click.argument('output_filepath', type=click.Path(), default=r"D:\PROJEKTY\intonat.dl\data\interim\data.json")
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    with open(output_filepath, 'r+') as f:
        f.truncate(0)
    logger.info("Starting data pre-processing")
    start_timer = time.perf_counter()
    read_data(input_filepath)
    end_timer = time.perf_counter()
    logger.info("Finishing data pre-processing in %d s", end_timer-start_timer)


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
