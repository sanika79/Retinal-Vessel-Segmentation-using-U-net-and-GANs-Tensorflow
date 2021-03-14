import pickle

import matplotlib.pyplot as plt
from keras import backend as K, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization, UpSampling2D, Activation, ReLU
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dropout
from keras.layers.merge import concatenate, Concatenate
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from scipy.io import savemat
import tensorflow as tf
import os.path
import numpy as np


class UNet:
    '''

    Changes:
        Changed dropout values to make them the default 0.5. Can Tune later.


    Todos:
        Add batch norm, regularization.


        if feeding something not in 0-1 range, make it 0-1 range




    '''

    def help(self):
        print(
            '''To use this class, first instantiate it as model_variable = UNet().
            Then, you must use one of the create model functions that are built in. The model parameters 
            are populated in-place, so a call to model_variable.create_specified_model(**params) will work.
            '''
        )

    def __init__(self, modelname='default_model'):
        # all the constants are set here

        # image params
        self.IMG_WIDTH = 256
        self.IMG_HEIGHT = 256
        self.IMG_CHANNELS = 1

        # model params
        self.compile_args = {
            # 'optimizer': Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False),
            'optimizer': 'adam',
            'loss': 'binary_crossentropy',
            # 'metrics': [self.jaccard_distance_loss]
            'metrics': ['accuracy'],
        }

        self.activation_function = 'relu'
        self.kernel_init = 'he_normal'

        # Regular
        self.output_channels = 5

        # Bone Segmentation
        # self.output_channels = 4

        self.iterations_before_stop = 500

        self.modelname = modelname
        self.batch_size = 1

        self.alpha_recip = 1. / 10 if 10 > 0 else 0

        self.BUFFER_SIZE = 60000
        self.BATCH_SIZE = 256

        self.EPOCHS = 50
        self.noise_dim = 100
        self.num_examples_to_generate = 16

        # We will reuse this seed overtime (so it's easier)
        # to visualize progress in the animated GIF)
        seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])

        # Set keras backend parameters
        # K.set_epsilon(1e-05)

    def create_UNet_retina(self):
        self.IMG_HEIGHT = 256
        self.IMG_WIDTH = 256
        self.output_channels = 1

        ks_multiplier = 1
        activation_function = self.activation_function
        kernel_init = self.kernel_init
        compile_args = self.compile_args
        output_channels = self.output_channels

        # Build U-Net model
        inputs = Input((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))

        # lambda function to squash to 0, 1 range form 0-255

        # infer network size and parameters

        # down
        c1 = Conv2D(16 * ks_multiplier, (3, 3), activation=activation_function, kernel_initializer=kernel_init,
                    padding='same')(
            inputs)
        c1 = BatchNormalization()(c1)
        c1 = Conv2D(16 * ks_multiplier, (3, 3), activation=activation_function, kernel_initializer=kernel_init,
                    padding='same')(c1)
        c1 = BatchNormalization()(c1)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(0.1)(p1)

        c2 = Conv2D(32 * ks_multiplier, (3, 3), activation=activation_function, kernel_initializer=kernel_init,
                    padding='same')(p1)
        c2 = BatchNormalization()(c2)
        c2 = Conv2D(32 * ks_multiplier, (3, 3), activation=activation_function, kernel_initializer=kernel_init,
                    padding='same')(c2)
        c2 = BatchNormalization()(c2)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(0.1)(p2)

        c3 = Conv2D(64 * ks_multiplier, (3, 3), activation=activation_function, kernel_initializer=kernel_init,
                    padding='same')(p2)
        c3 = BatchNormalization()(c3)
        c3 = Conv2D(64 * ks_multiplier, (3, 3), activation=activation_function, kernel_initializer=kernel_init,
                    padding='same')(c3)
        c3 = BatchNormalization()(c3)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(0.1)(p3)

        c4 = Conv2D(128 * ks_multiplier, (3, 3), activation=activation_function, kernel_initializer=kernel_init,
                    padding='same')(p3)
        c3 = BatchNormalization()(c3)
        c4 = Conv2D(128 * ks_multiplier, (3, 3), activation=activation_function, kernel_initializer=kernel_init,
                    padding='same')(c4)
        c4 = BatchNormalization()(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)
        p4 = Dropout(0.1)(p4)

        # Middle
        c5 = Conv2D(256 * ks_multiplier, (3, 3), activation=activation_function, kernel_initializer=kernel_init,
                    padding='same')(p4)
        c5 = BatchNormalization()(c5)
        c5 = Conv2D(256 * ks_multiplier, (3, 3), activation=activation_function, kernel_initializer=kernel_init,
                    padding='same')(c5)
        c5 = BatchNormalization()(c5)

        # up
        u6 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = BatchNormalization()(u6)
        u6 = concatenate([u6, c4])
        u6 = Dropout(0.1)(u6)
        u6 = Conv2D(128 * ks_multiplier, (3, 3), activation=activation_function, kernel_initializer=kernel_init,
                    padding='same')(u6)
        u6 = BatchNormalization()(u6)
        u6 = Conv2D(128 * ks_multiplier, (3, 3), activation=activation_function, kernel_initializer=kernel_init,
                    padding='same')(u6)
        u6 = BatchNormalization()(u6)

        u7 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(u6)
        u7 = BatchNormalization()(u7)
        u7 = concatenate([u7, c3])
        u7 = Dropout(0.1)(u7)
        u7 = Conv2D(64 * ks_multiplier, (3, 3), activation=activation_function, kernel_initializer=kernel_init,
                    padding='same')(u7)
        u7 = BatchNormalization()(u7)
        u7 = Conv2D(64 * ks_multiplier, (3, 3), activation=activation_function, kernel_initializer=kernel_init,
                    padding='same')(u7)
        u7 = BatchNormalization()(u7)

        u8 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(u7)
        u8 = BatchNormalization()(u8)
        u8 = concatenate([u8, c2])
        u8 = Dropout(0.1)(u8)
        u8 = Conv2D(32 * ks_multiplier, (3, 3), activation=activation_function, kernel_initializer=kernel_init,
                    padding='same')(u8)
        u8 = BatchNormalization()(u8)
        u8 = Conv2D(32 * ks_multiplier, (3, 3), activation=activation_function, kernel_initializer=kernel_init,
                    padding='same')(u8)
        u8 = BatchNormalization()(u8)

        u9 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(u8)
        u9 = BatchNormalization()(u9)
        u9 = concatenate([u9, c1], axis=3)
        u9 = Dropout(0.1)(u9)
        u9 = Conv2D(16 * ks_multiplier, (3, 3), activation=activation_function, kernel_initializer=kernel_init,
                    padding='same')(u9)
        u9 = BatchNormalization()(u9)
        u9 = Conv2D(16 * ks_multiplier, (3, 3), activation=activation_function, kernel_initializer=kernel_init,
                    padding='same')(u9)
        u9 = BatchNormalization()(u9)

        outputs = Conv2D(output_channels, (1, 1), activation='sigmoid')(u9)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(**compile_args)

        # remember the compiled model and its summary
        self.results = None
        self.summary = model.summary()
        self.model = model

        return model

    def __repr__(self):
        return self.summary

    def fit_model(self, Xtrain, Ytrain, nepochs=100, val_split=0.20, batch_size=8, save_best_only=True, **kwargs):
        """f
        Use this function to train the model on the input data.

        :param Xtrain: input data to train on. Pixel entries expected to be in 0, 1 range.
        :param Ytrain: labels to train on. Pixel entries expected to be in 0, 1 range.
        :param nepochs: (optional) number of epochs to train for. Early stopping after default 50 (
        self.iterations_before_stop) iterations.
        :param val_split: split to use for validation and training.
        :param batch_size: size of the batch to train on.
        :return: None, will save the model to disk and remember the results for plotting loss and accuracy curves.
        """
        assert self.model != None, 'Model not properly created or stored. Refer to __init__ function of ' \
                                   + str(self.__class__)

        # Xtrain, Ytrain = self.clean_data_dimensions(Xtrain, Ytrain)

        self.batch_size = batch_size
        # load in the model
        model = self.model

        # configure early stopping
        earlystopper = EarlyStopping(patience=self.iterations_before_stop, verbose=1)
        filepath = "saved-model-{epoch:02d}-{val_loss:.2f}-{loss:.2f}.hdf5"
        checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=save_best_only)

        train_args = {
            'validation_split': val_split,
            'batch_size': batch_size,
            'epochs': nepochs,
            'callbacks': [earlystopper, checkpointer],
            'verbose': 1,
            'shuffle': True
        }

        results = model.fit(Xtrain, Ytrain, **train_args)

        self.results = results

        self.save_model()

    def produce_ouputs_mat(self, input, labels, save=True, savefile=''):
        assert self.model is not None, 'Load in a model first'

        if savefile == '':
            savefile = self.modelname + '_output.mat'

        # predict outputs
        outputs = self.model.predict(input, batch_size=self.batch_size)

        if save:
            savemat(savefile, {'outputs': outputs, 'inputs': input, 'labels': labels})

        return outputs

    def test_model(self, Xtest, Ytest):
        assert self.model is not None, 'Load in a model first'
        assert False, 'function to be implemented'

    def plot_accuracy(self):
        """
        Ref: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
        :return:
        """
        assert self.results is not None, 'No data for plotting accuracy'

        history = self.results
        # plt.plot(history.history['jaccard_distance_loss'])
        # plt.plot(history.history['val_jaccard_distance_loss'])
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Jaccard Distance Loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        return plt

    def plot_loss(self):
        """
        Ref: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
        :return:
        """
        assert self.results is not None, 'No data for plotting accuracy'
        history = self.results

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        return plt

        """
        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

        The jaccard distance loss is usefull for unbalanced datasets. This has been
        shifted so it converges on 0 and is smoothed to avoid exploding or disappearing
        gradient.

        Ref: https://en.wikipedia.org/wiki/Jaccard_index

        @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
        @author: wassname
        """
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth

    def get_model_loss_and_val(self):
        """
        :return: Returns tuple to model loss and val.
        """
        history = self.results
        val_loss = history['val_loss'][-1]
        loss = history['loss'][-1]

        return loss, val_loss

    def save_model(self, fname=None):
        assert self.model != None, 'Model not properly created or stored. Refer to __init__ function of ' \
                                   + str(self.__class__)

        # load in the model`
        model = self.model
        model_json = model.to_json()

        if fname is None:
            fname = self.modelname

        with open(fname + '.json', 'w') as f:
            f.write(model_json)

        model.save_weights(fname + '.h5')
        print('saved model to file at', fname)

        pickle.dump(self.results.history, open('%s_training_history' % self.modelname, 'wb'))

    def load_model(self, fname=None):
        if fname is None and self.modelname is None:
            assert False, 'Please specify a valid path to get the model from.'
        if fname is None:
            fname = self.modelname

        # check if file exists
        if os.path.isfile(fname + '.json'):
            print('loaded model from file', fname)
            f_json = open(fname + '.json')
            model_json = f_json.read()
            f_json.close()

            # load in model from json
            model = model_from_json(model_json)
        else:
            if self.model is None:
                print('Load failed, please instantiate a model using one of the create_* functions speficied for the '
                      'modelfile.')
                return None

            model = self.model

        # load weights from hdf5 file
        model.load_weights(fname + '.h5')
        self.model = model
        return model

    def load_model_hdf5(self, fname=None):
        if fname is None and self.modelname is None:
            assert False, 'Please specify a valid path to get the model from.'
        if fname is None:
            fname = self.modelname

        # check if file exists
        if os.path.isfile(fname):
            # load in model from json
            self.model.load_weights(fname)

        else:
            if self.model is None:
                print(
                    'Load failed, please instantiate a model using one of the create_* functions specified for the '
                    'modelfile.')
                return None

    def parallelize_model(self, num_gpus):
        """
        Takes the current model and parallelizes it to multiple GPUs.
        :param num_gpus: number of GPU units to parallelize to
        :return: None, it is a setting.
        """

        self.single_gpu_model = self.model
        self.model = multi_gpu_model(self.model, gpus=num_gpus)
        self.model.compile(**self.compile_args)

