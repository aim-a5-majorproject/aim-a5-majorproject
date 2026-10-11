import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

IMG_SIZE = (224, 224)
DEFAULT_LEARNING_RATE = 1e-4

def build_model(imagenet=True, learning_rate=DEFAULT_LEARNING_RATE):
    """Build the VGG16 binary classifier used in the Colab notebook."""
    base = VGG16(
        weights="imagenet" if imagenet else None,
        include_top=False,
        input_shape=(224, 224, 3),
        name="vgg16",
    )
    base.trainable = False
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(base.output)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dense(128, activation="relu", name="fc2")(x)
    output = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = Model(base.input, output, name="NeuroScan_VGG16")
    model.compile(
        optimizer=Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model
