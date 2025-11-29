import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre

IMG_SIZE = 224

# -----------------------------------------
# Load YOUR VGG16 (same used during training)
# -----------------------------------------
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
last_conv_layer_name = "block5_conv3"

last_conv_layer = base_model.get_layer(last_conv_layer_name)

grad_model = Model(
    inputs=base_model.input,
    outputs=[last_conv_layer.output, GlobalAveragePooling2D()(last_conv_layer.output)]
)


def generate_gradcam(image):

    # Resize + preprocess
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    prep = np.expand_dims(img, axis=0).astype("float32")
    prep = vgg_pre(prep)

    # Grad-CAM algorithm
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(prep)
        loss = tf.reduce_max(predictions)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = np.zeros(shape=(conv_outputs.shape[0], conv_outputs.shape[1]))

    for i in range(pooled_grads.shape[0]):
        heatmap += pooled_grads[i] * conv_outputs[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8  # avoid divide by zero

    # Resize to original
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

    # Convert to color map
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlay on MRI
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    return heatmap_color, overlay
