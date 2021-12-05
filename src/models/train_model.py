from CVAE import CVAE
from ..features.build_features import load_data

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


def train(x_train, batch_size, epochs):
    autoencoder.summary()
    autoencoder.compile()
    autoencoder.train(x_train, x_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    x_train, _, _, _ = load_data()
    autoencoder = train(x_train[:10000], BATCH_SIZE, EPOCHS)
