import tensorflow as tf
from models.transfer_learning import build_transfer_learning_model
from data.data_prep import prepare_data

# Load data
train_generator, val_generator, test_generator = prepare_data()

# Load model
model = build_transfer_learning_model()

# Train model
history = model.fit(train_generator, validation_data=val_generator, epochs=10)

# Save model
model.save("models/cat_dog_classifier.h5")
