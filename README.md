# MNIST Handwritten Digit Recognition using Artificial Neural Network

A deep learning project that trains an Artificial Neural Network (ANN) on the MNIST dataset to classify handwritten digits (0-9) with 97.11% test accuracy, and extends the model to predict digits from real-world custom images.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Custom Image Prediction](#custom-image-prediction)
- [Technologies Used](#technologies-used)
- [Key Insights](#key-insights)
- [Future Improvements](#future-improvements)
- [How to Run](#how-to-run)

---

## Project Overview

The MNIST dataset is a benchmark in computer vision. This project goes beyond simply training the model — it demonstrates a complete ML pipeline: data exploration, preprocessing, model design, evaluation via confusion matrix, and real-world inference on a custom handwritten digit image.

The model successfully identifies a handwritten digit "2" from a screenshot taken outside the dataset, validating its ability to generalize beyond training data.

---

## Dataset

- **Source**: `keras.datasets.mnist`
- **Training samples**: 60,000 grayscale images
- **Test samples**: 10,000 grayscale images
- **Image dimensions**: 28 x 28 pixels
- **Classes**: 10 (digits 0 through 9)
- **Pixel value range**: 0 to 255 (normalized to 0.0 to 1.0)

---

## Project Pipeline

```
Data Loading
    |
Exploratory Data Analysis (shape, type, class distribution, sample visualization)
    |
Data Preprocessing (normalization: pixel / 255)
    |
Model Building (ANN with Keras Sequential API)
    |
Model Training (10 epochs, Adam optimizer)
    |
Model Evaluation (test accuracy, confusion matrix heatmap)
    |
Custom Image Prediction (real-world digit from screenshot)
```

---

## Model Architecture

| Layer       | Type    | Units | Activation |
|-------------|---------|-------|------------|
| Input       | Flatten | 784   | -          |
| Hidden 1    | Dense   | 50    | ReLU       |
| Hidden 2    | Dense   | 50    | ReLU       |
| Output      | Dense   | 10    | Sigmoid    |

- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metric**: Accuracy

---

## Results

### Training Progress

| Epoch | Accuracy | Loss   |
|-------|----------|--------|
| 1     | 91.47%   | 0.2975 |
| 2     | 95.91%   | 0.1376 |
| 3     | 96.96%   | 0.1017 |
| 4     | 97.54%   | 0.0809 |
| 5     | 98.00%   | 0.0672 |
| 6     | 98.32%   | 0.0561 |
| 7     | 98.53%   | 0.0482 |
| 8     | 98.65%   | 0.0420 |
| 9     | 98.79%   | 0.0365 |
| 10    | 98.94%   | 0.0323 |

### Test Set Performance

| Metric        | Value   |
|---------------|---------|
| Test Accuracy | 97.11%  |
| Test Loss     | 0.1074  |

The model achieves 97.11% accuracy on unseen test data, demonstrating strong generalization.

---

## Custom Image Prediction

A handwritten digit "2" was captured via screenshot and fed into the model after preprocessing:

- Converted from RGB to Grayscale using OpenCV
- Resized from 476x472 to 28x28 pixels
- Normalized and reshaped to match model input format (1, 28, 28)

**Predicted Output**: The Handwritten Digit is recognized as **2** — correct prediction.

---

## Technologies Used

- Python 3.12
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV (cv2)
- Pillow (PIL)
- Google Colab

---

## Key Insights

- A simple ANN with just two hidden layers (50 neurons each) achieves over 97% test accuracy on MNIST, showing that dense networks can perform well on structured image data when the classification task is not spatially complex.
- Digit "5" had the highest misclassification rate, frequently confused with "3" and "8" — shapes that share curved structures.
- Normalization (dividing by 255) was essential to stable and faster convergence during training.
- The model generalizes to real-world custom images after standard preprocessing, validating practical usability.

---

## Future Improvements

- **Replace ANN with CNN**: Convolutional Neural Networks are architecturally suited for image data. Adding Conv2D and MaxPooling layers would likely push accuracy above 99% by capturing spatial features.
- **Data Augmentation**: Apply random rotations, shifts, and zooms to training images to improve robustness against varied handwriting styles in real-world usage.
- **Dropout Regularization**: Add Dropout layers between dense layers to reduce overfitting as model complexity increases.
- **Hyperparameter Tuning**: Experiment with different neuron counts, learning rates, and batch sizes using Keras Tuner.
- **Deploy as Web Application**: Wrap the model in a Flask or Streamlit app with a canvas interface where users draw digits and get real-time predictions.
- **Extend to Custom Datasets**: Retrain on domain-specific handwritten text datasets (such as Devanagari digits) for localized use cases.
- **Model Export**: Save the trained model using `model.save()` and serve it via TensorFlow Serving or ONNX for production deployment.

---

## How to Run

1. Open the notebook in Google Colab.
2. Run all cells in sequence.
3. To test custom image prediction, upload a grayscale PNG image of a handwritten digit and update the `input_image_path` variable in the respective cell.

```python
input_image_path = '/content/your_digit_image.png'
```

4. The model will output the predicted digit label.

---

