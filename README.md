# Cat vs Dog Classifier

![Project Banner](![alt text](image.png))

Cat vs Dog Classifier is a project that uses **Convolutional Neural Networks (CNN)** and **Transfer Learning** from **MobileNetV2** to classify images of cats and dogs. This project is built with **TensorFlow/Keras** and uses **DVC** for data and model version control, along with **GitHub Actions** for CI/CD.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [CI/CD with GitHub Actions](#cicd-with-github-actions)
- [Example Output](#example-output)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

This project is designed to classify images of cats and dogs using:
- **Computer Vision** and **CNN**
- **Transfer Learning** from MobileNetV2
- **Data Augmentation** to improve training data quality
- **DVC** to manage the dataset and model artifacts
- **GitHub Actions** for automated CI/CD pipelines

---

## Project Structure

```
cat-vs-dog-classifier/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions workflow file
├── data/
│   ├── raw/                   # Raw dataset (not committed to GitHub)
│   │   ├── training_set/
│   │   │   └── training_set/  # Images for training (cats, dogs)
│   │   └── test_set/
│   │       └── test_set/      # Images for testing (cats, dogs)
│   └── processed/             # Processed output (tracked by DVC)
│       └── info.txt           # Summary of prepared data
├── deployment/
│   └── api/
│       └── main.py            # API for model deployment (FastAPI)
├── mlops/
│   └── mlflow_tracking.py     # Example for experiment tracking with MLflow
├── models/
│   ├── baseline_model.py
│   ├── transfer_learning.py
│   ├── train.py               # Script for training the model
│   ├── evaluate.py            # Script for evaluating the model
│   └── cat_dog_classifier.h5  # Trained model (tracked by DVC)
├── .dvcignore                 # Files/directories ignored by DVC
├── dvc.yaml                   # DVC pipeline definitions
├── dvc.lock                   # DVC lock file
├── config.yaml                # Project parameter configurations
├── requirements.txt           # List of dependencies
├── README.md                  # Project documentation (this file)
└── .gitignore                 # Files/directories ignored by Git
```

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/SurakiatP/cat-vs-dog-classifier.git
   cd cat-vs-dog-classifier
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   # For Windows:
   venv\Scripts\activate
   # For Mac/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install "dvc[gdrive]"
   ```

4. **Set Up DVC:**
   - Run `dvc init` (if not already done).
   - Add the dataset with:
     ```bash
     dvc add data/raw/training_set/training_set
     dvc add data/raw/test_set/test_set
     ```
   - Configure the DVC remote (example using Google Drive):

     ```bash
     dvc remote add -d gdrive_remote gdrive://<GDRIVE_FOLDER_ID>/cat-vs-dog-classifier
     dvc remote modify gdrive_remote gdrive_client_id <YOUR_CLIENT_ID>
     dvc remote modify gdrive_remote gdrive_client_secret <YOUR_CLIENT_SECRET>
     ```

   > **Note:** Replace `<GDRIVE_FOLDER_ID>`, `<YOUR_CLIENT_ID>`, and `<YOUR_CLIENT_SECRET>` with your actual values from the Google Cloud Console.

---

## Usage

### Data Preparation

Run the script to prepare data:
```bash
python -m data.data_prep
```
Or use the DVC pipeline:
```bash
dvc repro prepare_data
```

### Training

To train the model, run:
```bash
python -m models.train
```
Or use the DVC pipeline:
```bash
dvc repro train
```

### Evaluation

To evaluate the model and see prediction results, run:
```bash
python -m models.evaluate
```
Or use the DVC pipeline:
```bash
dvc repro evaluate
```

---

## CI/CD with GitHub Actions

This project uses **GitHub Actions** to automatically run the pipeline whenever a push or pull request is made to the `main` branch.  
See the details in the [`.github/workflows/ci.yml`](.github/workflows/ci.yml) file.

---

## Example Output

Below is an example of the model prediction:

![Example Prediction](![alt text](image-1.png))

> **Note:** The example image shows a prediction output like "Dog 🐶" or "Cat 🐱".  
> You can update the sample image as needed.

---

## Contributing

Contributions are welcome! To contribute:
1. Fork this repository.
2. Create a new branch for your feature or bug fix.
3. Commit and push your changes.
4. Open a Pull Request against the `main` branch.

Please ensure that the pipeline and tests pass before submitting your PR.

## 🤝 Contributing
Feel free to fork the repository and submit a pull request with improvements.

---

## 📜 License
This project is licensed under the MIT License.

---

## ✨ Contact
For questions, reach out at: [surakiat.0723@gmail.com] or connect on [LinkedIn](https://www.linkedin.com/in/surakiat-kansa-ard-171942351/)

---
