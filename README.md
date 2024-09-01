

# Face Expression Recognition using ResNet

## Overview

This project aims to classify human facial expressions into different categories using a deep learning model based on the ResNet architecture. Facial expression recognition is an important application in areas such as human-computer interaction, security systems, and emotional AI.

The project is implemented in a Jupyter Notebook (`face-expression-resnet.ipynb`) and demonstrates the complete process from data preprocessing to model training and evaluation.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset used for this project is the [FER-2013 (Facial Expression Recognition)](https://www.kaggle.com/datasets/msambare/fer2013) dataset. It consists of grayscale images of faces, each labeled with one of the following emotions:

- **Angry**
- **Disgust**
- **Fear**
- **Happy**
- **Sad**
- **Surprise**
- **Neutral**

The dataset contains 35,887 images in total, with each image being 48x48 pixels.

## Project Structure

```bash
├── data
│   ├── fer2013.csv               # Raw dataset in CSV format
├── face-expression-resnet.ipynb  # Jupyter Notebook for model development
├── models
│   ├── resnet_model.h5           # Saved trained model
├── README.md                     # Project README file
└── requirements.txt              # Python packages required
```

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/face-expression-recognition.git
   cd face-expression-recognition
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Preprocessing

Data preprocessing involves the following steps:

- **Data Loading:** The images and labels are loaded from the FER-2013 dataset.
- **Image Normalization:** The pixel values of images are scaled to the range [0, 1].
- **Data Augmentation:** Techniques such as horizontal flipping, rotation, and zooming are applied to augment the dataset and improve model generalization.

## Model Architecture

The model is based on the **ResNet-50** architecture, a deep residual network that allows for training very deep neural networks by using skip connections (or residual blocks). This architecture has proven effective for image classification tasks.

Key layers include:

- **Convolutional Layers:** Extract features from the input images.
- **Residual Blocks:** Allow for deeper networks by mitigating the vanishing gradient problem.
- **Fully Connected Layers:** Perform the final classification of facial expressions.

## Training

The model is trained using the following configuration:

- **Optimizer:** Adam optimizer with a learning rate of 0.001.
- **Loss Function:** Categorical Cross-Entropy.
- **Metrics:** Accuracy.
- **Epochs:** 25 epochs with early stopping based on validation accuracy.
- **Batch Size:** 64.

The training process is visualized using accuracy and loss plots to monitor the model's performance over time.

## Evaluation

The trained model is evaluated on the test set using the following metrics:

- **Accuracy:** Measures the overall percentage of correct predictions.
- **Confusion Matrix:** Provides insights into the model's ability to distinguish between different facial expressions.

## Results

- The model achieved an accuracy of **X%** on the test set.
- The confusion matrix shows that the model performs particularly well in recognizing certain expressions like "Happy" and "Neutral," but has more difficulty with others like "Fear" and "Disgust."
  
*Note: Replace **X%** with the actual accuracy achieved during your model's evaluation.*

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to create a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

