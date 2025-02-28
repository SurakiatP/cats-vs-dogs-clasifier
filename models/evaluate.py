import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import json


MODEL_PATH = "models/cat_dog_classifier.h5"

def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def predict(image_path):
    model = load_model()
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return "Dog 🐶" if prediction[0][0] > 0.5 else "Cat 🐱"

if __name__ == "__main__":
    img_path = "data/raw/test_set/test_set/dogs/dog.4001.jpg"  # ใส่ path ของรูปที่ต้องการทดสอบ
    result = predict(img_path)
    print(f"Prediction: {result}")

    metrics = {"prediction": result}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)
