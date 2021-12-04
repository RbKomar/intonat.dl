from CVAE import CVAE
from ..features.build_features import load_data

BATCH_SIZE = 64
EPOCHS = 150

def train(x_train, batch_size, epochs):
    autoencoder = CVAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()
    autoencoder.compile()
    autoencoder.train(x_train, x_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    x_train, _, _, _ = load_data()
    autoencoder = train(x_train[:10000], BATCH_SIZE, EPOCHS)
