import keras
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
#gitt

model = load_model("cell_cycle_detection.keras")

imgPath = 'archive (1)/cellData/data/train/Prophase/12168_merged.jpg'

img = image.load_img(imgPath, target_size=(66, 66))

img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)

img_array /= 255.

processed_img = img_array

predictions = model.predict(processed_img)

print(predictions)

probabilities = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)

print(probabilities)

predicted_indices = np.argmax(probabilities, axis=1)
print(predicted_indices)
