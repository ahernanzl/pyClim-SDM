"""
This module contains the definition of the deep learning models for
statistical downscaling. References to each of the models are provided
in the docstring of each class.

Authors:
    Jose González-Abad
    Alfonso Hernanz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


class DeepESD_Discriminator(torch.nn.Module):
    """
    Discriminator for the DeepESD-based cGAN.
    Receives (X, Y) pairs and predicts the probability of Y being real.

    X: (batch, channels, lat, lon)
    Y: (batch, stations)
    Output: scalar in [0,1] per sample.

    Architecture:
      - CNN encoder for X
      - Concatenate flattened X features + Y
      - Fully-connected layers for classification

    Parameters
    ----------
    x_shape : tuple
        Shape of the data used as predictor. This must have dimension 4
        (time, channels/variables, lon, lat).

    y_shape : tuple
        Shape of the data used as predictand. This must have dimension 2
        (time, gridpoint)

    filters_last_conv : int
        Number of filters/kernels of the last convolutional layer
    """

    def __init__(self, x_shape: tuple, y_shape: tuple, filters_last_conv: int = 25):
        super(DeepESD_Discriminator, self).__init__()

        if (len(x_shape) != 4) or (len(y_shape) != 2):
            raise ValueError("X must have 4 dims and Y must have 2 dims.")

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv

        self.conv_1 = torch.nn.Conv2d(in_channels=self.x_shape[1],
                                      out_channels=50,
                                      kernel_size=3,
                                      padding=1)

        self.conv_2 = torch.nn.Conv2d(in_channels=50,
                                      out_channels=25,
                                      kernel_size=3,
                                      padding=1)

        self.conv_3 = torch.nn.Conv2d(in_channels=25,
                                      out_channels=self.filters_last_conv,
                                      kernel_size=3,
                                      padding=1)

        n_features_x = self.x_shape[2] * self.x_shape[3] * self.filters_last_conv
        n_features_total = n_features_x + self.y_shape[1]

        self.fc1 = torch.nn.Linear(n_features_total, 256)
        self.fc2 = torch.nn.Linear(256, 64)
        self.fc3 = torch.nn.Linear(64, 1)

        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Predictor tensor (batch, channels, lat, lon)
        y : torch.Tensor
            Predictand tensor (batch, stations)

        Returns
        -------
        torch.Tensor
            Probability that (x, y) is real, shape (batch, 1)
        """
        x = self.leaky_relu(self.conv_1(x))
        x = self.leaky_relu(self.conv_2(x))
        x = self.leaky_relu(self.conv_3(x))

        x = torch.flatten(x, start_dim=1)

        xy = torch.cat((x, y), dim=1)

        xy = self.leaky_relu(self.fc1(xy))
        xy = self.leaky_relu(self.fc2(xy))
        out = self.sigmoid(self.fc3(xy))

        return out

class DeepESDtas(torch.nn.Module):

    """
    DeepESD model as proposed in Baño-Medina et al. 2024 for temperature
    downscasling. This implementation allows for a deterministic (MSE-based)
    and stochastic (NLL-based) definition.

    Baño-Medina, J., Manzanas, R., Cimadevilla, E., Fernández, J., González-Abad,
    J., Cofiño, A. S., and Gutiérrez, J. M.: Downscaling multi-model climate projection
    ensembles with deep learning (DeepESD): contribution to CORDEX EUR-44, Geosci. Model
    Dev., 15, 6747–6758, https://doi.org/10.5194/gmd-15-6747-2022, 2022.

    Parameters
    ----------
    x_shape : tuple
        Shape of the data used as predictor. This must have dimension 4
        (time, channels/variables, lon, lat).

    y_shape : tuple
        Shape of the data used as predictand. This must have dimension 2
        (time, gridpoint)

    filters_last_conv : int
        Number of filters/kernels of the last convolutional layer

    stochastic: bool
        If set to True, the model is composed of two final dense layers computing
        the mean and log fo the variance. Otherwise, the models is composed of one
        final layer computing the values.
    """


    def __init__(self, x_shape: tuple, y_shape: tuple,
                 filters_last_conv: int, stochastic: bool):

        super(DeepESDtas, self).__init__()

        if (len(x_shape) != 4) or (len(y_shape) != 2):
            error_msg =\
            'X and Y data must have a dimension of length 4'
            'and 2, correspondingly'

            raise ValueError(error_msg)

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv
        self.stochastic = stochastic

        self.conv_1 = torch.nn.Conv2d(in_channels=self.x_shape[1],
                                      out_channels=50,
                                      kernel_size=3,
                                      padding=1)

        self.conv_2 = torch.nn.Conv2d(in_channels=50,
                                      out_channels=25,
                                      kernel_size=3,
                                      padding=1)

        self.conv_3 = torch.nn.Conv2d(in_channels=25,
                                      out_channels=self.filters_last_conv,
                                      kernel_size=3,
                                      padding=1)

        if self.stochastic:
            self.out_mean = torch.nn.Linear(in_features=\
                                            self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                            out_features=self.y_shape[1])

            self.out_log_var = torch.nn.Linear(in_features=\
                                               self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                               out_features=self.y_shape[1])

        else:
            self.out = torch.nn.Linear(in_features=\
                                       self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                       out_features=self.y_shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv_1(x)
        x = torch.relu(x)

        x = self.conv_2(x)
        x = torch.relu(x)

        x = self.conv_3(x)
        x = torch.relu(x)

        x = torch.flatten(x, start_dim=1)

        if self.stochastic:
            mean = self.out_mean(x)
            log_var = self.out_log_var(x)
            out = torch.cat((mean, log_var), dim=1)
        else:
            out = self.out(x)

        return out

class DeepESDpr(torch.nn.Module):

    """
    DeepESD model as proposed in Baño-Medina et al. 2024 for precipitation
    downscaling. This implementation allows for a deterministic (MSE-based)
    and stochastic (NLL-based) definition.

    Baño-Medina, J., Manzanas, R., Cimadevilla, E., Fernández, J., González-Abad,
    J., Cofiño, A. S., and Gutiérrez, J. M.: Downscaling multi-model climate projection
    ensembles with deep learning (DeepESD): contribution to CORDEX EUR-44, Geosci. Model
    Dev., 15, 6747–6758, https://doi.org/10.5194/gmd-15-6747-2022, 2022.

    Parameters
    ----------
    x_shape : tuple
        Shape of the data used as predictor. This must have dimension 4
        (time, channels/variables, lon, lat).

    y_shape : tuple
        Shape of the data used as predictand. This must have dimension 2
        (time, gridpoint)

    filters_last_conv : int
        Number of filters/kernels of the last convolutional layer

    stochastic: bool
        If set to True, the model is composed of three final dense layers computing
        the p, shape and scale of the Bernoulli-gamma distribution. Otherwise,
        the models is composed of one final layer computing the values.

    last_relu: bool, optional
        If set to True, the output of the last dense layer is passed through a
        ReLU activation function. This does not apply when stochastic=True. By
        default is set to False.
    """

    def __init__(self, x_shape: tuple, y_shape: tuple,
                 filters_last_conv: int, stochastic: bool,
                 last_relu: bool=False):

        super(DeepESDpr, self).__init__()

        if (len(x_shape) != 4) or (len(y_shape) != 2):
            error_msg =\
            'X and Y data must have a dimension of length 4'
            'and 2, correspondingly'

            raise ValueError(error_msg)

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv
        self.stochastic = stochastic
        self.last_relu = last_relu

        self.conv_1 = torch.nn.Conv2d(in_channels=self.x_shape[1],
                                      out_channels=50,
                                      kernel_size=3,
                                      padding=1)

        self.conv_2 = torch.nn.Conv2d(in_channels=50,
                                      out_channels=25,
                                      kernel_size=3,
                                      padding=1)

        self.conv_3 = torch.nn.Conv2d(in_channels=25,
                                      out_channels=self.filters_last_conv,
                                      kernel_size=3,
                                      padding=1)

        if self.stochastic:
            self.p = torch.nn.Linear(in_features=\
                                            self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                            out_features=self.y_shape[1])

            self.log_shape = torch.nn.Linear(in_features=\
                                             self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                             out_features=self.y_shape[1])

            self.log_scale = torch.nn.Linear(in_features=\
                                             self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                             out_features=self.y_shape[1])

        else:
            self.out = torch.nn.Linear(in_features=\
                                       self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                       out_features=self.y_shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv_1(x)
        x = torch.relu(x)

        x = self.conv_2(x)
        x = torch.relu(x)

        x = self.conv_3(x)
        x = torch.relu(x)

        x = torch.flatten(x, start_dim=1)

        if self.stochastic:
            p = self.p(x)
            p = torch.sigmoid(p)

            log_shape = self.log_shape(x)
            log_scale = self.log_scale(x)

            out = torch.cat((p, log_shape, log_scale), dim = 1)
        else:
            out = self.out(x)
            if self.last_relu: out = torch.relu(out)

        return out

class UnitConv(nn.Module):
    
    """
    Implement the following set of layers:
    2D convolution => Batch Normalization (opt.) => ReLU (x2)

    Parameters
    ----------
    in_channels : int
        Input channels to the block

    out_channels : int
        Output channels of the block

    kernel_size : int
        Kernel size of all convolutions applied within the
        block

    padding: str
        Padding (same or valid) to apply before each convolutional
        layer
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 padding: int, batch_norm: bool):
        super().__init__()
 
        if batch_norm:
            self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU())
        else:

            self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.ReLU())

    def forward(self, x):
        return self.conv(x)

class UpLayer(nn.Module):
    
    """
    Implement one of the following set of layers:
    (2D transposed conv) or (up sampling => 2D convolution)

    Parameters
    ----------
    in_channels : int
        Input channels to the block

    out_channels : int
        Output channels of the block

    trans_conv: bool
        Whether to apply the transposed convolution (True)
        or the up-sampling + 2D convolution
    """

    def __init__(self, in_channels: int, out_channels: int, trans_conv: bool):
        super().__init__()
 
        if trans_conv:
            self.layer_op = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=2, stride=2)
        else:
            self.layer_op = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )

    def forward(self, x):
        return self.layer_op(x)

class UnetTas(torch.nn.Module):

    def __init__(self, x_shape, y_shape, stochastic,
                 input_padding, kernel_size, padding,
                 batch_norm, trans_conv):

        super(UnetTas, self).__init__()

        if (len(x_shape) != 4) or (len(y_shape) != 2):
            error_msg =\
            'X and Y data must have a dimension of length 4'
            'and 2, correspondingly'

            raise ValueError(error_msg)

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.stochastic = stochastic
        self.input_padding = input_padding
        self.kernel_size = int(kernel_size)
        self.padding = padding
        self.batch_norm = batch_norm
        self.trans_conv = trans_conv

        # Encoder
        self.down_conv_1 = UnitConv(in_channels=self.x_shape[1], out_channels=64,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    batch_norm=self.batch_norm)
        self.maxpool_1 = nn.MaxPool2d((2, 2))

        self.down_conv_2 = UnitConv(in_channels=64, out_channels=128,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    batch_norm=self.batch_norm)
        self.maxpool_2 = nn.MaxPool2d((2, 2))

        self.down_conv_3 = UnitConv(in_channels=128, out_channels=256,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    batch_norm=self.batch_norm)
        self.maxpool_3 = nn.MaxPool2d((2, 2))

        self.down_conv_4 = UnitConv(in_channels=256, out_channels=512,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    batch_norm=self.batch_norm)
        self.maxpool_4 = nn.MaxPool2d((2, 2))

        # Decoder
        self.trans_conv_1 = UpLayer(in_channels=512, out_channels=256,
                                    trans_conv=self.trans_conv)
        self.up_conv_1 = UnitConv(in_channels=512, out_channels=256,
                                  kernel_size=self.kernel_size, padding=self.padding,
                                  batch_norm=self.batch_norm)

        self.trans_conv_2 = UpLayer(in_channels=256, out_channels=128,
                                    trans_conv=self.trans_conv)
        self.up_conv_2 = UnitConv(in_channels=256, out_channels=128,
                                  kernel_size=self.kernel_size, padding=self.padding,
                                  batch_norm=self.batch_norm)

        self.trans_conv_3 = UpLayer(in_channels=128, out_channels=64,
                                    trans_conv=self.trans_conv)
        self.up_conv_3 = UnitConv(in_channels=128, out_channels=64,
                                  kernel_size=self.kernel_size, padding=self.padding,
                                  batch_norm=self.batch_norm)

        # Final segment
        self.trans_conv_4 = UpLayer(in_channels=64, out_channels=64,
                                    trans_conv=self.trans_conv)
        self.up_conv_4 = UnitConv(in_channels=64, out_channels=64,
                                  kernel_size=self.kernel_size, padding=self.padding,
                                  batch_norm=self.batch_norm)

        self.trans_conv_5 = UpLayer(in_channels=64, out_channels=64,
                                    trans_conv=self.trans_conv)
        self.up_conv_5 = UnitConv(in_channels=64, out_channels=64,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    batch_norm=self.batch_norm)

        if self.stochastic:
            self.out_mean = nn.Conv2d(in_channels=64, out_channels=1,
                                      kernel_size=1)

            self.out_log_var = nn.Conv2d(in_channels=64, out_channels=1,
                                         kernel_size=1)
        else:
            self.out = nn.Conv2d(in_channels=64, out_channels=1,
                                 kernel_size=1)

    def forward(self, x):

        x = F.pad(x, self.input_padding)

        # Encoder
        x1 = self.down_conv_1(x)
        x1_maxpool = self.maxpool_1(x1)

        x2 = self.down_conv_2(x1_maxpool)
        x2_maxpool = self.maxpool_2(x2)

        x3 = self.down_conv_3(x2_maxpool)
        x3_maxpool = self.maxpool_3(x3)

        x4 = self.down_conv_4(x3_maxpool)

        # Decoder
        x5 = self.trans_conv_1(x4)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.up_conv_1(x5)

        x6 = self.trans_conv_2(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.up_conv_2(x6)

        x7 = self.trans_conv_3(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.up_conv_3(x7)

        # Final segment
        x8 = self.trans_conv_4(x7)
        x8 = self.up_conv_4(x8)

        x9 = self.trans_conv_5(x8)
        x9 = self.up_conv_5(x9)

        # Final layers
        if self.stochastic:
            mean = self.out_mean(x9) 
            log_var = self.out_log_var(x9)

            mean = torch.flatten(mean, start_dim=1)
            log_var = torch.flatten(log_var, start_dim=1)

            out = torch.cat([mean, log_var], dim=1)
        else:
            out = self.out(x9)
            out = torch.flatten(out, start_dim=1)

        return out

class UnetPr(torch.nn.Module):

    def __init__(self, x_shape, y_shape, stochastic,
                 input_padding, kernel_size, padding,
                 batch_norm, trans_conv):

        super(UnetPr, self).__init__()

        if (len(x_shape) != 4) or (len(y_shape) != 2):
            error_msg =\
            'X and Y data must have a dimension of length 4'
            'and 2, correspondingly'

            raise ValueError(error_msg)

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.stochastic = stochastic
        self.input_padding = input_padding
        self.kernel_size = int(kernel_size)
        self.padding = padding
        self.batch_norm = batch_norm
        self.trans_conv = trans_conv

        # Encoder
        self.down_conv_1 = UnitConv(in_channels=self.x_shape[1], out_channels=64,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    batch_norm=self.batch_norm)
        self.maxpool_1 = nn.MaxPool2d((2, 2))

        self.down_conv_2 = UnitConv(in_channels=64, out_channels=128,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    batch_norm=self.batch_norm)
        self.maxpool_2 = nn.MaxPool2d((2, 2))

        self.down_conv_3 = UnitConv(in_channels=128, out_channels=256,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    batch_norm=self.batch_norm)
        self.maxpool_3 = nn.MaxPool2d((2, 2))

        self.down_conv_4 = UnitConv(in_channels=256, out_channels=512,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    batch_norm=self.batch_norm)
        self.maxpool_4 = nn.MaxPool2d((2, 2))

        # Decoder
        self.trans_conv_1 = UpLayer(in_channels=512, out_channels=256,
                                    trans_conv=self.trans_conv)
        self.up_conv_1 = UnitConv(in_channels=512, out_channels=256,
                                  kernel_size=self.kernel_size, padding=self.padding,
                                  batch_norm=self.batch_norm)

        self.trans_conv_2 = UpLayer(in_channels=256, out_channels=128,
                                    trans_conv=self.trans_conv)
        self.up_conv_2 = UnitConv(in_channels=256, out_channels=128,
                                  kernel_size=self.kernel_size, padding=self.padding,
                                  batch_norm=self.batch_norm)

        self.trans_conv_3 = UpLayer(in_channels=128, out_channels=64,
                                    trans_conv=self.trans_conv)
        self.up_conv_3 = UnitConv(in_channels=128, out_channels=64,
                                  kernel_size=self.kernel_size, padding=self.padding,
                                  batch_norm=self.batch_norm)

        # Final segment
        self.trans_conv_4 = UpLayer(in_channels=64, out_channels=64,
                                    trans_conv=self.trans_conv)
        self.up_conv_4 = UnitConv(in_channels=64, out_channels=64,
                                  kernel_size=self.kernel_size, padding=self.padding,
                                  batch_norm=self.batch_norm)

        self.trans_conv_5 = UpLayer(in_channels=64, out_channels=64,
                                    trans_conv=self.trans_conv)
        self.up_conv_5 = UnitConv(in_channels=64, out_channels=64,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    batch_norm=self.batch_norm)

        if self.stochastic:
            self.p = nn.Conv2d(in_channels=64, out_channels=1,
                               kernel_size=1)

            self.log_shape = nn.Conv2d(in_channels=64, out_channels=1,
                                       kernel_size=1)

            self.log_scale = nn.Conv2d(in_channels=64, out_channels=1,
                                       kernel_size=1)

        else:
            self.out = nn.Conv2d(in_channels=64, out_channels=1,
                                 kernel_size=1)
    def forward(self, x):

        x = F.pad(x, self.input_padding)

        # Encoder
        x1 = self.down_conv_1(x)
        x1_maxpool = self.maxpool_1(x1)

        x2 = self.down_conv_2(x1_maxpool)
        x2_maxpool = self.maxpool_2(x2)

        x3 = self.down_conv_3(x2_maxpool)
        x3_maxpool = self.maxpool_3(x3)

        x4 = self.down_conv_4(x3_maxpool)

        # Decoder
        x5 = self.trans_conv_1(x4)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.up_conv_1(x5)

        x6 = self.trans_conv_2(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.up_conv_2(x6)

        x7 = self.trans_conv_3(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.up_conv_3(x7)

        # Final segment
        x8 = self.trans_conv_4(x7)
        x8 = self.up_conv_4(x8)

        x9 = self.trans_conv_5(x8)
        x9 = self.up_conv_5(x9)

        # Final layers
        if self.stochastic:
            p = self.p(x9)
            p = torch.sigmoid(p)
            p = torch.flatten(p, start_dim=1)

            log_shape = self.log_shape(x9)
            log_shape = torch.flatten(log_shape, start_dim=1)

            log_scale = self.log_scale(x9)
            log_scale = torch.flatten(log_scale, start_dim=1)

            out = torch.cat((p, log_shape, log_scale), dim = 1)
        else:
            out = self.out(x9)
            out = torch.flatten(out, start_dim=1)

        return out