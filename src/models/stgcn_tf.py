"""
ST-GCN TensorFlow Implementation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from .graph_utils import GraphWithPartition


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from .graph_utils import GraphWithPartition


class ConvTemporalGraphical(layers.Layer):
    """Graph convolution layer (NHWC)."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.t_kernel_size = t_kernel_size
        self.t_stride = t_stride
        self.t_padding = t_padding
        self.t_dilation = t_dilation
        self.bias = bias

        self.conv = layers.Conv2D(
            filters=out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            strides=(t_stride, 1),
            padding='same' if t_padding > 0 else 'valid',
            dilation_rate=(t_dilation, 1),
            use_bias=bias
            # channels_last by default
        )

    def call(self, x, A):
        # x: (N, T, V, C)
        x = self.conv(x)  # (N, T, V, out_channels*K)

        n = tf.shape(x)[0]
        t = tf.shape(x)[1]
        v = tf.shape(x)[2]
        kc = tf.shape(x)[3]  # out_channels * K

        # reshape for graph conv: (N, T, V, K, C) -> (N, K, C, T, V)
        x = tf.reshape(x, [n, t, v, self.kernel_size, kc // self.kernel_size])
        x = tf.transpose(x, [0, 3, 4, 1, 2])

        # Graph convolution: nkctv, kvw -> nctw
        x = tf.einsum('nkctv,kvw->nctw', x, A)

        # Convert back to NHWC
        x = tf.transpose(x, [0, 2, 3, 1])  # (N, T, V, C)
        return x, A

    def get_config(self):
        config = super().get_config()
        config.update({
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            't_kernel_size': self.t_kernel_size,
            't_stride': self.t_stride,
            't_padding': self.t_padding,
            't_dilation': self.t_dilation,
            'bias': self.bias,
        })
        return config


class STGCN_BLOCK(layers.Layer):
    """ST-GCN block (NHWC)."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_rate = dropout
        self.residual = residual

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        # Temporal conv + BN + ReLU + Dropout
        self.bn1 = layers.BatchNormalization(axis=-1)
        self.relu1 = layers.ReLU()
        self.conv_temporal = layers.Conv2D(
            filters=out_channels,
            kernel_size=(kernel_size[0], 1),
            strides=(stride, 1),
            padding='same'
        )
        self.bn2 = layers.BatchNormalization(axis=-1)
        self.dropout = layers.Dropout(dropout)

        # Residual
        if not residual:
         # Instead of zeros_like, use a 1x1 conv to match channels
         self.residual_conv = layers.Conv2D(
                filters=out_channels,
                kernel_size=1,
                strides=(stride, 1),
                padding='same'
                )
         self.residual_bn = layers.BatchNormalization(axis=-1)
         self.residual_func = lambda x: self.residual_bn(self.residual_conv(x))

        elif in_channels == out_channels and stride == 1:
            self.residual_func = lambda x: x
        else:
           self.residual_conv = layers.Conv2D(
            filters=out_channels,
            kernel_size=1,
            strides=(stride, 1),
            padding='same'# no data_format, default is NHWC
            )
           self.residual_bn = layers.BatchNormalization(axis=-1)
           self.residual_func = lambda x: self.residual_bn(self.residual_conv(x))


        self.relu2 = layers.ReLU()

    def call(self, x, A, training=False):
        res = self.residual_func(x)
        x, A = self.gcn(x, A)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.conv_temporal(x)
        x = self.bn2(x, training=training)
        x = self.dropout(x, training=training)
        x = x + res
        x = self.relu2(x)
        return x, A

    def get_config(self):
        config = super().get_config()
        config.update({
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dropout': self.dropout_rate,
            'residual': self.residual
        })
        return config


class STGCN(keras.Model):
    """Full ST-GCN (NHWC)."""
    def __init__(self, in_channels, graph_args, edge_importance_weighting, n_out_features=256, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.graph_args = graph_args
        self.edge_importance_weighting = edge_importance_weighting
        self.n_out_features = n_out_features
        self.dropout = kwargs.get('dropout', 0)

        self.graph = GraphWithPartition(
            num_nodes=graph_args['num_nodes'],
            center=graph_args['center'],
            inward_edges=graph_args['inward_edges']
        )
        self.A = tf.constant(self.graph.A, dtype=tf.float32)

        spatial_kernel_size = self.A.shape[0]
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        # BatchNorm for input
        self.data_bn = layers.BatchNormalization(axis=-1)

        # Build ST-GCN blocks
        kwargs0 = {k: v for k, v in kwargs.items() if k != "dropout"}
        dropout = kwargs.get('dropout', 0)
        self.st_gcn_networks = [
            STGCN_BLOCK(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            STGCN_BLOCK(64, 64, kernel_size, 1, dropout=dropout),
            STGCN_BLOCK(64, 64, kernel_size, 1, dropout=dropout),
            STGCN_BLOCK(64, 64, kernel_size, 1, dropout=dropout),
            STGCN_BLOCK(64, 128, kernel_size, 2, dropout=dropout),
            STGCN_BLOCK(128, 128, kernel_size, 1, dropout=dropout),
            STGCN_BLOCK(128, 128, kernel_size, 1, dropout=dropout),
            STGCN_BLOCK(128, 256, kernel_size, 2, dropout=dropout),
            STGCN_BLOCK(256, 256, kernel_size, 1, dropout=dropout),
            STGCN_BLOCK(256, n_out_features, kernel_size, 1, dropout=dropout)
        ]

        # Edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = [
                self.add_weight(
                    name=f'edge_importance_{i}',
                    shape=self.A.shape,
                    initializer='ones',
                    trainable=True
                )
                for i in range(len(self.st_gcn_networks))
            ]
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def call(self, x, training=False):
        # Detect input format
        if len(x.shape) == 4:
            if x.shape[-1] != self.in_channels and x.shape[1] == self.in_channels:
                # N, C, T, V → N, T, V, C
                x = tf.transpose(x, [0, 2, 3, 1])

        # x is now NHWC (N, T, V, C)
        N, T, V, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        # Input batchnorm
        x = self.data_bn(x, training=training)

        # ST-GCN blocks
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance, training=training)

        # Global average pooling (time + nodes)
        x = tf.reduce_mean(x, axis=[1, 2])  # (N, C)
        return x



class FC(layers.Layer):
    """
    Fully connected layer head
    
    Args:
        n_features (int): Number of features in input
        num_class (int): Number of classes for classification
        dropout_ratio (float): Dropout ratio. Default: 0.2
        batch_norm (bool): Whether to use batch norm. Default: False
    """
    
    def __init__(self, n_features, num_class, dropout_ratio=0.2, batch_norm=False, **kwargs):
        super().__init__(**kwargs)
        
        self.n_features = n_features
        self.num_class = num_class
        self.dropout_ratio = dropout_ratio
        self.batch_norm = batch_norm
        
        self.dropout = layers.Dropout(dropout_ratio)
        self.use_bn = batch_norm
        if batch_norm:
            self.bn = layers.BatchNormalization()
        
        # Initialize like PyTorch: normal(0, sqrt(2/num_class))
        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=np.sqrt(2.0 / num_class))
        self.classifier = layers.Dense(num_class, kernel_initializer=initializer)
    
    def call(self, x, training=False):
        """
        Args:
            x: (batch_size, n_features)
        
        Returns:
            logits: (batch_size, num_class)
        """
        x = self.dropout(x, training=training)
        if self.use_bn:
            x = self.bn(x, training=training)
        x = self.classifier(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_features': self.n_features,
            'num_class': self.num_class,
            'dropout_ratio': self.dropout_ratio,
            'batch_norm': self.batch_norm,
        })
        return config


class Network(keras.Model):
    """
    Complete network: Encoder (STGCN) + Decoder (FC)
    """
    
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    def call(self, x, training=False):
        x = self.encoder(x, training=training)
        x = self.decoder(x, training=training)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'encoder': self.encoder,
            'decoder': self.decoder,
        })
        return config
