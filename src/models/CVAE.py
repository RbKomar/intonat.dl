from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Dense, \
    Reshape, Conv2DTranspose, Lambda
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adamax
import mlflow
import numpy as np
import os
import pickle


class CVAE:
    """
    Autoencoder represents a Deep Convolutional autoencoder architecture with
    mirrored encoder and decoder components.
    original by https://github.com/musikalkemist/generating-sound-with-neural-networks/tree/main/13%20Training%20VAE%20with%20audio%20data
    """

    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self.log_vars = None
        self.mu = None

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()


    def train(self, x_train, y_train, batch_size, num_epochs):
        def exponential_decay(lr0):
            def exponential_decay_fn(epoch):
                return lr0 * 0.1**(epoch / 20)
            return exponential_decay_fn

        exponential_decay_fn = exponential_decay(lr0=0.01)

        callbacks = [
            LearningRateScheduler(exponential_decay_fn),
            EarlyStopping(monitor="loss", patience=4),
            ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
            TensorBoard(r"logs/")
        ]
        with mlflow.start_run():
            self.model.fit(x_train,
                           y_train,
                           batch_size=batch_size,
                           epochs=num_epochs,
                           shuffle=True,
                           callbacks=callbacks)

    def _reconstruction_loss(self, y_target, y_predict):
        """MSE"""
        reconstruction_loss = K.mean(K.square(y_target - y_predict), axis=[1, 2, 3])
        return reconstruction_loss

    def _kl_loss(self, y_target, y_predict):
        kl_loss = -0.5 * K.sum(1 + self.log_vars - K.square(self.mu) - K.exp(self.log_vars), axis=1)
        return kl_loss

    def _combined_loss(self, y_target, y_predict):
        reconstruction_loss = self._reconstruction_loss(y_target, y_predict)
        kl_loss = self._kl_loss(y_target, y_predict)
        return reconstruction_loss + kl_loss

    def compile(self, ):
        optimizer = Adamax()
        self.model.compile(optimizer=optimizer,
                           loss=self._combined_loss,
                           metrics=[self._reconstruction_loss,
                                    self._kl_loss])

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_cvae()

    def _build_encoder(self):
        encoder_input = Input(shape=self.input_shape, name="encoder_input")
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_conv_layers(self, encoder_input):
        """Create all convolutional blocks in encoder."""
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """
        Add a convolutional block to a graph of layers, consisting of
        conv 2d + SeLU & Lecun Normalization + Dropout.
        """
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            kernel_initializer='lecun_normal',
            activation='selu',
            name=f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = Dropout(0.3, name=f"encoder_dropout_{layer_number}")(x)  # references.txt [1]
        return x

    def _add_bottleneck(self, x):
        """Flatten data and add bottleneck with Guassian sampling (Dense
        layer).
        """
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        self.mu = Dense(self.latent_space_dim, name="mu")(x)
        self.log_vars = Dense(self.latent_space_dim,
                              name="log_variance")(x)

        def reparametrization_trick(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(self.mu), mean=0.,
                                      stddev=1.)
            sampled_point = mu + K.exp(log_variance / 2) * epsilon
            return sampled_point

        x = Lambda(reparametrization_trick,
                   name="encoder_output")([self.mu, self.log_vars])
        return x

    def _build_decoder(self):
        decoder_input = Input(shape=self.latent_space_dim, name="decoder_input")
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            kernel_initializer='lecun_normal',
            activation='selu',
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = Dropout(0.3, name=f"decoder_droupout_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            activation="sigmoid",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        output_layer = conv_transpose_layer(x)
        return output_layer

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = CVAE(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder
