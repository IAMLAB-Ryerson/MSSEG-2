# -*- coding: utf-8 -*-
"""
@author: Adam Gibicar, Samir Mitha
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Add, Input, Conv2D, Dropout,
                                     Activation, concatenate, MaxPooling2D,
                                     Conv2DTranspose, BatchNormalization)


class SC_Unet(Model):

    '''
    Description:
    Creates an SC-Unet model object for semantic segmentation.


    Example:
    >> scunet = SC_Unet()
    >> model = scunet.get_model()


    Raises:
    >> Assertion: batch normalization must be 'all', 'first' or 'none'


    Variables:
    >> img_shape(tuple): dimensions of input image
    >> initializer(str): weight initialization method (e.g. 'he_normal')
    >> kernel_size(tuple): size of convolution filter kernels (e.g 3 x 3)
    >> num_filters(list): number of filters at each level
    >> activation(str): activation functions used after convolutions
    >> activation_output(str): activation function used in final layer
    >> batch_norm(str): batch normalization configuration
    >> pretrained_weights(str): directory containing pretrained model

    Methods:
    >> get_model: returns the keras model
    >> get_params: returns a list of the U-net parameters
    >> __conv_block: returns a convolution block (i.e. conv, batch, act)
    >> __encoder_block: used to learn features and downsample images
    >> __decoder_block: used for localization (i.e. upsample, concat)
    >> __forward: assemble SC-Unet model architecture
    '''

    def __init__(self,
                 img_shape=(256, 256, 1),
                 initializer='he_normal',
                 kernel_size=(3, 3),
                 num_filters=[32, 64, 128, 256, 512],
                 activation='relu',
                 activation_output='sigmoid',
                 batch_norm='first',
                 pretrained_weights=None):

        # Initializations
        super(SC_Unet, self).__init__()
        self.img_shape = img_shape
        self.initializer = initializer
        self.kernel_size = kernel_size
        self.activation = activation
        self.activation_output = activation_output
        self.batch_norm = batch_norm.lower()
        self.pretrained_weights = pretrained_weights
        self.num_filters = num_filters

        # Assertions
        error_message = "Batch normalization must be 'first','all' or 'none.'"
        assert (batch_norm == 'all' or
                batch_norm == 'first' or
                batch_norm == 'none'), error_message


    def get_model(self):

        '''
        Description
        -----------
        Returns the SC-Unet model architecture.


        Example
        -------
        >> model = scunet.get_model()


        Parameters
        ----------
        >> None


        Returns
        -------
        >> model: keras model object
        '''

        return self.__forward()


    def get_params(self):

        '''
        Description
        -----------
        Returns a dictionary containing the parameters of the SC-Unet.


        Example
        -------
        >> params = scunet.get_params()


        Parameters
        ----------
        >> None


        Returns
        -------
        >> params(dict): dictionary of model parameters
        '''

        params = {'Input shape': self.img_shape,
                  'Initializer': self.initializer,
                  'Kernel size': self.kernel_size,
                  'Activation': self.activation,
                  'Activation (output)': self.activation_output,
                  'Number of filters': self.num_filters,
                  'Batch normalization': self.batch_norm,
                  'Pretrained weights': self.pretrained_weights,
                  'Model': 'SC-Unet'}

        return params


    def __conv_block(self, input_tensor, num_filters):

        '''
        Description
        -----------
        Returns the output of a convolution block, which consists of two
        2-D convolution layers, each followed by activation layers.
        Batch normalization layers can optionally be specified as 'none',
        'first', or 'all'.


        Example
        -------
        >> output = self.__conv_block(input_tensor, 32)


        Parameters
        ----------
        >> input_tensor(float): input to the convolution block
        >> num_filters(int): number of filters in each convolution layer


        Returns
        -------
        >> conv2: output of the final layer
        '''

        if(self.batch_norm == 'all'):

            conv1 = Conv2D(num_filters, self.kernel_size, padding='same', kernel_initializer=self.initializer)(input_tensor)
            conv1 = Activation(self.activation)(conv1)
            conv1 = BatchNormalization()(conv1)

            conv2 = Conv2D(num_filters, self.kernel_size, padding='same', kernel_initializer=self.initializer)(conv1)
            conv2 = Activation(self.activation)(conv2)
            conv2 = BatchNormalization()(conv2)

        elif(self.batch_norm == 'first'):

            conv1 = Conv2D(num_filters, self.kernel_size, padding='same', kernel_initializer=self.initializer)(input_tensor)
            conv1 = Activation(self.activation)(conv1)
            conv1 = BatchNormalization()(conv1)

            conv2 = Conv2D(num_filters, self.kernel_size, padding='same', kernel_initializer=self.initializer)(conv1)
            conv2 = Activation(self.activation)(conv2)

        elif(self.batch_norm == 'none'):

            conv1 = Conv2D(num_filters, self.kernel_size, padding='same', kernel_initializer=self.initializer)(input_tensor)
            conv1 = Activation(self.activation)(conv1)

            conv2 = Conv2D(num_filters, self.kernel_size, padding='same', kernel_initializer=self.initializer)(conv1)
            conv2 = Activation(self.activation)(conv2)

        return conv2


    def __encoder_block(self, input_tensor, num_filters):

        '''
        Description
        -----------
        Returns the output of an encoder block, which consists of a
        convolution block followed by a max-pooling operation to perform
        down-sampling and increase the receptive field.


        Example
        -------
        >> pool, encoder = self.__encoder_block(input_tensor, 64)


        Parameters
        ----------
        >> input_tensor(float): input to the encoder block
        >> num_filters(int): number of filters in each convolution layer


        Returns
        -------
        >> pool: output following max-pooling
        >> encoder: output after convolution block but before pooling
        >> sc_output: additional output for skip-connection
        '''

        encoder = self.__conv_block(input_tensor, num_filters)
        pool = MaxPooling2D((2, 2), strides=(2, 2))(encoder)
        sc_output = Conv2D(2*num_filters, (1, 1), padding='same', kernel_initializer=self.initializer)(pool)

        return pool, encoder, sc_output


    def __decoder_block(self, input_tensor, concat_tensor, num_filters):

        '''
        Description
        -----------
        Returns the output of a decoder block, which consists of 2-D
        transposed convolution for upsampling, concatenation with feature
        channels from the encoding path and a convolution block.


        Example
        -------
        >> decoder = self.__decoder_block(input_tensor, encoder, 128)


        Parameters
        ----------
        >> input_tensor(float): input to the decoder block
        >> concat_tensor(float): encoder output for concatenation
        >> num_filters(int): number of filter in each convolution layer


        Returns
        -------
        >> decoder: output of decoder block
        '''

        decoder = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
        decoder = concatenate([concat_tensor, decoder], axis=-1)
        decoder = self.__conv_block(decoder, num_filters)

        return decoder


    def __forward(self):

        '''
        Description
        -----------
        Returns the final SC-Unet model architecture by assembling
        various combinations of convolution blocks, encoder blocks and
        decoder blocks.


        Example
        -------
        >> model = self.__architecture()


        Parameters
        ----------
        >> None


        Returns
        -------
        >> model: final model
        '''

        # Input layer
        input_layer = Input(shape=self.img_shape)

        # Encoding path
        encoder_pool1, encoder1, sc1 = self.__encoder_block(input_layer, self.num_filters[0])
        encoder_pool2, encoder2, sc2 = self.__encoder_block(encoder_pool1, self.num_filters[1])
        encoder_pool3, encoder3, sc3 = self.__encoder_block(encoder_pool2, self.num_filters[2])
        encoder_pool4, encoder4, sc4 = self.__encoder_block(encoder_pool3, self.num_filters[3])

        # Center path
        center = self.__conv_block(encoder_pool4, self.num_filters[4])
        center = Add()([center, sc4])

        # Decoding path
        decoder4 = self.__decoder_block(center, encoder4, self.num_filters[3])
        decoder4 = Add()([decoder4, sc3])
        decoder3 = self.__decoder_block(decoder4, encoder3, self.num_filters[2])
        decoder3 = Add()([decoder3, sc2])
        decoder2 = self.__decoder_block(decoder3, encoder2, self.num_filters[1])
        decoder2 = Add()([decoder2, sc1])
        decoder1 = self.__decoder_block(decoder2, encoder1, self.num_filters[0])
                
        # Output layer
        output_layer = Conv2D(1, (1, 1), activation=self.activation_output)(decoder1)
        model = Model(inputs=input_layer, outputs=output_layer)

        # Pretrained weights
        if(self.pretrained_weights):
            model.load_weights(self.pretrained_weights)

        return model
        

def select_model(input_params={}, choice='scunet'):

    '''
    Description
    -----------
    Creates a model specified by the user. The user can select between
    one of the following architectures: 'unet', 'uresnet', 'uresnet2',
    'tiramisu, 'scunet' or 'multiresunet'.


    Example
    -------
    >> choice = 'uresnet'
    
    >> params = {'img_shape': (256, 256, 1),
                 'batch_norm': 'pre-activation',
                 'kernel_size': (5, 5)}
    
    >> model = select_model(params, choice)


    Parameters
    ----------
    >> input_params(dict): dictionary of model parameters
    >> choice(str): model name


    Returns
    -------
    >> model: keras model
    >> model_params(dict): final model parameters
    '''

    choice = choice.lower()

    if(choice == 'scunet'):
        scunet = SC_Unet(**input_params)
        model = scunet.get_model()
        model_params = scunet.get_params()

    return model, model_params
