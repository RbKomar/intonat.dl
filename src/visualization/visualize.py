import librosa
import matplotlib.pyplot as plt
import json


def plot_spectogram_db(y_signal, sr, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    log_y_signal = librosa.power_to_db(y_signal)
    librosa.display.specshow(log_y_signal,
                             sr=sr,
                             x_axis="time",
                             y_axis=y_axis)
    plt.colorbar(format="%_2.f")
    plt.show()


def plot_mfccs(mffcs, sr):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(mffcs,
                             x_axis="time",
                             sr=sr)
    plt.colorbar(format="%_2.f")
    plt.show()


def main(input_file=r"D:\PROJEKTY\intonat.dl\data\interim\data.json"):

    with open(input_file) as fp:
        data = json.load(fp)
        mfcc = data["mfcc"][0][0]
        delta_mfcc = data["delta_mfcc"][0][0]
        delta2_mfcc = data["delta2_mfcc"][0][0]
        sr = data["sr"][0]
        plot_mfccs(mfcc, sr)
        plot_mfccs(delta_mfcc, sr)
        plot_mfccs(delta2_mfcc, sr)


if __name__ == "__main__":
    main()
