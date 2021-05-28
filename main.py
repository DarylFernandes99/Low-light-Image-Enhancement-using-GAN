import numpy as np
import cv2, os
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array, save_img, array_to_img

model_path = "<enter model path>"
image_path = "<enter image path>"

# Processing Image
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_arr = (img_to_array(img) - 127.5) / 127.5
resized = cv2.resize(img_arr, (256, 256), interpolation=cv2.INTER_AREA)
ready_img = np.expand_dims(resized, axis=0)

# Loading Model
model = load_model(model_path)

# Prdicting Image
pred = model.predict(ready_img)
pred = (cv2.medianBlur(pred[0], 1) + 1) / 2
pred = array_to_img(pred)
save_img("./output.png", pred)