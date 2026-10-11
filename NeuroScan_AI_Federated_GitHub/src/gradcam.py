import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

def make_gradcam(image_batch, model, conv_name="block5_conv3"):
    """Create the same VGG16 Grad-CAM heatmap used by the notebook."""
    vgg = model.get_layer("vgg16")
    conv = vgg.get_layer(conv_name)
    conv_model = Model(vgg.input, conv.output)

    inp = tf.keras.Input(shape=conv.output.shape[1:])
    x = inp
    start = False
    for layer in vgg.layers:
        if layer.name == conv_name:
            start = True
            continue
        if start:
            x = layer(x)

    x = model.get_layer("global_average_pool")(x)
    x = model.get_layer("fc1")(x)
    x = model.get_layer("fc2")(x)
    output = model.get_layer("output")(x)
    classifier = Model(inp, output)

    with tf.GradientTape() as tape:
        conv_out = conv_model(image_batch)
        tape.watch(conv_out)
        score = classifier(conv_out)[:, 0]

    grads = tape.gradient(score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    return tf.where(maximum > 0, heatmap / maximum, heatmap).numpy()

def overlay_heatmap(original_rgb, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_rgb.shape[1], original_rgb.shape[0]))
    colored = cv2.cvtColor(
        cv2.applyColorMap(
            np.uint8(np.clip(heatmap, 0, 1) * 255), cv2.COLORMAP_JET
        ),
        cv2.COLOR_BGR2RGB,
    )
    return np.clip((1 - alpha) * original_rgb + alpha * colored, 0, 255).astype(np.uint8)
