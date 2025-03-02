import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = "./data/raw"
TRAIN_DIR = os.path.join(BASE_DIR, "training_set/training_set")  
TEST_DIR = os.path.join(BASE_DIR, "test_set/test_set")            

if not os.path.exists(TRAIN_DIR) or not os.listdir(TRAIN_DIR):
    raise FileNotFoundError(f"❌ No training images found in '{TRAIN_DIR}'. Please add dataset first!")

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(150, 150),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

def prepare_data():
    """
    ฟังก์ชัน prepare_data จะคืนค่า generator สำหรับ train, validation และ test
    """
    return train_generator, val_generator, test_generator

if __name__ == "__main__":
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    summary_file = os.path.join(processed_dir, "info.txt")
    with open(summary_file, "w") as f:
        f.write("Data Preparation Summary\n")
        f.write(f"Training images: {train_generator.samples}\n")
        f.write(f"Validation images: {val_generator.samples}\n")
        f.write(f"Test images: {test_generator.samples}\n")

    print("✅ Data Preparation Complete!")
