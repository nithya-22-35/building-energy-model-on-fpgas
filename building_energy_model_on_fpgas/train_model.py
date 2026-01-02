# QKeras model definition and training
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1

from qkeras import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu


def build_anomaly_model(input_dim):
    """
    Build a QKeras binary classification model
    for energy anomaly detection.
    """

    model = Sequential()

    # Layer 1
    model.add(
        QDense(
            32,
            input_shape=(input_dim,),
            name="fc1",
            kernel_quantizer=quantized_bits(6, 0, alpha=1),
            bias_quantizer=quantized_bits(6, 0, alpha=1),
            kernel_initializer="lecun_uniform",
            kernel_regularizer=l1(1e-4),
        )
    )
    model.add(QActivation(quantized_relu(6), name="relu1"))

    # Layer 2
    model.add(
        QDense(
            16,
            name="fc2",
            kernel_quantizer=quantized_bits(6, 0, alpha=1),
            bias_quantizer=quantized_bits(6, 0, alpha=1),
            kernel_initializer="lecun_uniform",
            kernel_regularizer=l1(1e-4),
        )
    )
    model.add(QActivation(quantized_relu(6), name="relu2"))

    # Output layer (binary classification)
    model.add(
        QDense(
            1,
            name="output",
            kernel_quantizer=quantized_bits(6, 0, alpha=1),
            bias_quantizer=quantized_bits(6, 0, alpha=1),
            kernel_initializer="lecun_uniform",
        )
    )

    # Sigmoid activation
    model.add(tf.keras.layers.Activation("sigmoid", name="sigmoid"))

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model
