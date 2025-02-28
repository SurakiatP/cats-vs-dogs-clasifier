import mlflow
import mlflow.keras
from models.transfer_learning import build_transfer_learning_model
from data.data_prep import prepare_data

# Load data
train_generator, val_generator, test_generator = prepare_data()

# Load model
model = build_transfer_learning_model()

# Start MLflow
mlflow.set_experiment("Cat_vs_Dog_Classification")

with mlflow.start_run():
    history = model.fit(train_generator, validation_data=val_generator, epochs=10)

    # Log Model
    mlflow.keras.log_model(model, "model")

    # Log Metrics
    mlflow.log_metric("final_accuracy", history.history["accuracy"][-1])
    mlflow.log_metric("final_val_accuracy", history.history["val_accuracy"][-1])

    print("Model Training Complete & Logged to MLflow")
