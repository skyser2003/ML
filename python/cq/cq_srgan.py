from tensorflow import keras

from tensorflow.python.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model
import numpy as np


def calc_layer_loss(model: keras.Model, x):
    output = model.predict(x)
    print(output.shape)


base_model: keras.Model = VGG19(weights='imagenet', input_shape=(66, 66, 3), include_top=False, pooling="avg")
resnet_model: keras.Model = ResNet50(weights="imagenet", input_shape=(66, 66, 3), include_top=False, pooling="avg")

for layer in resnet_model.layers:
    print(layer.name)

loss_layers_name = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4", "block5_conv4"]
models = []

for layer in base_model.layers:
    layer: keras.layers.Layer = layer
    if layer.name in loss_layers_name:
        model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer.name).output)
        models.append(model)

img_path = '../cq_data/cq/face/ar_1_1.png'
img = image.load_img(img_path)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

for model in models:
    calc_layer_loss(model, x)

calc_layer_loss(base_model, x)
calc_layer_loss(resnet_model, x)
