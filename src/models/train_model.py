from CVAE import CVAE
from ..features.build_features import load_data
import keras_tuner as kt
from datetime import datetime
BATCH_SIZE = 64
EPOCHS = 150


def build_model(hp):
    conv_filters = []
    conv_kernels = []
    conv_strides = []
    latent_space_dim = None
    for i in range(hp.Int("num_layers", 1, 10)):
        conv_filters.append(hp.Int('conv_' + str(i) + '_filter', min_value=32, max_value=128, step=16))
        conv_kernels.append(hp.Choice('conv_' + str(i) + '_kernel', values=[3, 5]))
        conv_strides.append(hp.Choice('conv_' + str(i) + '_strides', values=[1, 2]))
        latent_space_dim = hp.Int("latent_spade_dimension", min_value=16, max_value=128, step=8)
    model = CVAE(
        input_shape=(28, 28, 1),
        conv_filters=conv_filters,
        conv_kernels=conv_kernels,
        conv_strides=conv_strides,
        latent_space_dim=latent_space_dim
    )
    return model


def train_vae(train, test, batch_size, epochs):
    tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=5)
    tuner.search(train, train, batch_size=batch_size, epochs=epochs, validation_data=(test, test))
    best_model = tuner.get_best_models()[0]
    return best_model


if __name__ == "__main__":
    x_train, _, x_test, _ = load_data()
    best_model = train_vae(x_train[:10000], BATCH_SIZE, EPOCHS)
    best_model.save(r"../../models/" + datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p"))
