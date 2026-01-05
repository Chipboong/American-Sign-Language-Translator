import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# --- MOCK GRAPH UTILS ---
class GraphWithPartition:
    def __init__(self, num_nodes, center, inward_edges):
        self.num_nodes = num_nodes
        # Shape: (3, num_nodes, num_nodes) for spatial kernel size 3
        self.A = np.random.rand(3, num_nodes, num_nodes).astype(np.float32)

# --- YOUR MODEL CODE ---

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
        )

    def call(self, x, A):
        x = self.conv(x)
        n = tf.shape(x)[0]
        t = tf.shape(x)[1]
        v = tf.shape(x)[2]
        kc = tf.shape(x)[3]

        x = tf.reshape(x, [n, t, v, self.kernel_size, kc // self.kernel_size])
        x = tf.transpose(x, [0, 3, 4, 1, 2])
        x = tf.einsum('nkctv,kvw->nctw', x, A)
        x = tf.transpose(x, [0, 2, 3, 1])
        return x, A


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

        if not residual:
            self.residual_func = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual_func = lambda x: x
        else:
            self.residual_conv = layers.Conv2D(
                filters=out_channels,
                kernel_size=1,
                strides=(stride, 1),
                padding='same'
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

        self.data_bn = layers.BatchNormalization(axis=-1)

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
        if len(x.shape) == 4:
            if x.shape[-1] != self.in_channels and x.shape[1] == self.in_channels:
                x = tf.transpose(x, [0, 2, 3, 1])

        x = self.data_bn(x, training=training)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance, training=training)

        x = tf.reduce_mean(x, axis=[1, 2])
        return x


class FC(layers.Layer):
    """Fully connected layer head"""
    def __init__(self, n_features, num_class, dropout_ratio=0.2, batch_norm=False, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.num_class = num_class
        self.dropout = layers.Dropout(dropout_ratio)
        self.use_bn = batch_norm
        if batch_norm:
            self.bn = layers.BatchNormalization()
        
        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=np.sqrt(2.0 / num_class))
        self.classifier = layers.Dense(num_class, kernel_initializer=initializer)
    
    def call(self, x, training=False):
        x = self.dropout(x, training=training)
        if self.use_bn:
            x = self.bn(x, training=training)
        x = self.classifier(x)
        return x


# --- VERIFICATION BLOCK ---
if __name__ == "__main__":
    print("--- Initializing STGCN Model ---")
    
    metadata = {
        'num_frames': 60,    # T
        'num_nodes': 27,     # V
        'num_channels': 3,   # C
        'num_classes': 20
    }
    
    graph_args = {
        'num_nodes': metadata['num_nodes'],
        'center': 0,
        'inward_edges': [] 
    }

    # 1. Instantiate the Sub-Components
    stgcn_encoder = STGCN(
        in_channels=metadata['num_channels'], 
        graph_args=graph_args,
        edge_importance_weighting=True,
        n_out_features=256
    )

    fc_decoder = FC(
        n_features=256, 
        num_class=metadata['num_classes']
    )

    # 2. Build a FUNCTIONAL Model Graph
    # This explicitly tells Keras: Input -> STGCN -> FC -> Output.
    # This guarantees the 'Output Shape' column in the summary will be populated.
    
    inputs = keras.Input(shape=(metadata['num_frames'], metadata['num_nodes'], metadata['num_channels']), name="input_frames")
    
    # Pass inputs through encoder
    features = stgcn_encoder(inputs)
    
    # Pass features through decoder
    outputs = fc_decoder(features)
    
    # Create the model container
    model = keras.Model(inputs=inputs, outputs=outputs, name="stgcn_network")

    # 3. Print Summary
    print("\n--- Model Summary (Functional Graph) ---")
    model.summary()
    
    print(f"\n✅ Verification:")
    print(f"   Input:   {inputs.shape}")
    print(f"   Encoder: {features.shape} (Should be Batch, 256)")
    print(f"   Decoder: {outputs.shape} (Should be Batch, 20)")