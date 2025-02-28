from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import uvicorn
import io
from PIL import Image

app = FastAPI()

MODEL_PATH = "models/cat_dog_classifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "🐶🐱 Welcome to Cat vs Dog Classifier API! Use /predict to classify an image."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return {"prediction": "Dog 🐶" if prediction[0][0] > 0.5 else "Cat 🐱"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
