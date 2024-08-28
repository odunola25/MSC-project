Brain Tumor Classification using transfer learning and deep learning models.
Project Overview

This repository contains a deep learning project for classifying brain tumor types using MRI images. The model is built using TensorFlow and Keras, leveraging a fine-tuned MobileNet architecture, InceptionV3 and a convolutional Neural Network to achieve high accuracy in detecting four types of brain tumors: Glioma, Meningioma, No Tumor, and Pituitary tumors. A comparison of the evaluation matrix for all models is done to determine which model performs better in the classification of brain tumors.

Table of Contents

- Dataset
- Model Architecture
- Training Process
- Evaluation
- Dependencies
- How to Use

Dataset
The dataset used in this project consists of MRI images categorized into four classes:

1. Glioma
2. Meningioma
3. No Tumor
4. Pituitary

The dataset is split into training and testing sets, located in the following directories:

- Training data: `/content/drive/MyDrive/MSC Project/Brain Tumor MRI Dataset/Training_project`

- Testing data: `/content/drive/MyDrive/MSC Project/Brain Tumor MRI Dataset/Testing_project`

•	Training Data: 5712 images belonging to 4 classes.
•	Testing Data: 1311 images belonging to 4 classes.
•	Classes: Glioma, Meningioma, No Tumor, Pituitary.
The dataset is preprocessed using the ImageDataGenerator to augment and normalize the images.
- [Results](#results)
- [License](#license)

Brain Tumor MRI Classification Using InceptionV3 Model 
The model is built using the InceptionV3 architecture, pre-trained on ImageNet. The following modifications were made:
•	The InceptionV3 base is used as a feature extractor with include top=False.
•	A GlobalAveragePooling2D layer is added to reduce the feature maps to a 1D tensor.
•	A Batch Normalization layer and a Dropout layer are added for regularization.
•	Two fully connected (Dense) layers are added, with the final layer having 4 units corresponding to the 4 classes.
Training Details
•	Optimizer: Adam with a learning rate of 1e-5.
•	Loss Function: Categorical Crossentropy.
•	Metrics: Accuracy
•	Batch Size: 32/64
•	Callbacks:
o	EarlyStopping: Monitors val_loss with a patience of 3.
o	ModelCheckpoint: Saves the best model based on val_loss.
o	ReduceLROnPlateau: Reduces learning rate by a factor of 0.2 if val_loss plateaus.
Fine-Tuning
•	The InceptionV3 layers up to the 100th layer were frozen, and the remaining layers were trained.
Results
•	Test Accuracy: 99.46%
•	Confusion Matrix: Plotted to visualize model performance across all classes
•	Classification Report:
o	Precision, Recall, F1-Score for each class is nearly 1.00.


Brain Tumor MRI Classification Using MobileNet

Model Architecture

The model is based on the pre-trained MobileNet architecture with the following modifications:

1. Global Average Pooling: Replaces the fully connected layers.
2. Batch Normalization: Added after the pooling layer to improve model stability and performance.
3. Dense Layers: A fully connected layer with 1024 units and ReLU activation.
4. Dropout Layer: A dropout rate of 0.5 to prevent overfitting.
5. Output Layer: A dense layer with 4 units (corresponding to the 4 classes) and softmax activation.


Training Process

The model is trained using the following techniques:

- Data Augmentation: Applied to the training set to increase model robustness. Augmentations include rotation, width and height shifts, shearing, zooming, and horizontal flipping.
- Fine-Tuning: The MobileNet base layers are partially frozen during initial training, then the entire model is fine-tuned.
- Callbacks: Early stopping, model checkpointing, and learning rate reduction on plateau are used to prevent overfitting and optimize training.

Training Hyperparameters

- Epochs: 100
- Batch Size: 32/64
- Learning Rate: 1e-5
-Callbacks
	-EarlyStopping: Stops training if the validation loss does not improve after 5 epochs.
	-ModelCheckpoint: Saves the best model based on validation loss.
	-ReduceLROnPlateau: Reduces learning rate if validation loss does not improve.

Evaluation

The model is evaluated on the test set with the following metrics:

- Accuracy: 94.82%
- Precision, Recall, and F1-Score: Provided for each class in the classification report.
- Confusion Matrix: Plotted to visualize model performance across all classes.

Dependencies

The following Python packages are required:

- tensorflow
- numpy
- matplotlib
- seaborn
- scikit-learn

Install them using pip:
bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
Results
•	Test Accuracy: 94.82%
•	Confusion Matrix: Provides detailed insights into classification accuracy for each class.
•	Classification Report: Precision, recall, and F1-scores are calculated for all classes.

Brain Tumor MRI Classification Using CNN

Libraries

This project is built using Python 3 and requires the following libraries:

- tensorflow
- keras
- numpy
- matplotlib

You can install the required libraries using pip:

bash
pip install tensorflow numpy matplotlib
Data Preprocessing
The data is augmented using the ImageDataGenerator from Keras, which includes operations such as:
•	Rescaling
•	Rotation
•	Width and height shifts
•	Shear transformation
•	Zooming
•	Horizontal flip
This helps to increase the diversity of the training dataset, preventing overfitting.
Model Architecture
The CNN model is composed of:
•	3 Convolutional Layers: For feature extraction with 32, 64, and 128 filters respectively.
•	3 MaxPooling Layers: To reduce the spatial dimensions after convolution.
•	Flatten Layer: To convert the 2D matrix to a 1D vector.
•	Dense Layer: Fully connected layer with 512 neurons, followed by a dropout of 0.5 for regularization.
•	Output Layer: Softmax layer with 4 neurons corresponding to the 4 classes of the dataset.
Compilation and Training
The model is compiled with:
•	Optimizer: Adam with a learning rate of 1e-5.
•	Loss Function: Categorical Crossentropy for multi-class classification.
•	Metrics: Accuracy
Training is performed for up to 100 epochs, with callbacks for:
•	EarlyStopping: Stops training if the validation loss does not improve for 5 epochs.
•	ModelCheckpoint: Saves the best model based on validation loss.
•	ReduceLROnPlateau: Reduces the learning rate if the validation loss does not improve.
Training Process
The training process includes the following steps:
1.	Model Creation: A CNN model is built with the specified architecture.
2.	Data Generation: Training and testing data generators are created using ImageDataGenerator.
3.	Model Training: The model is trained using the training generator and validated on the testing generator.
4.	Model Saving: The best-performing model is saved based on the lowest validation loss.
Model Evaluation
After training, the model can be evaluated using the testing dataset to determine its accuracy and loss metrics.

How to Use
1.	Dataset Preparation: Ensure your dataset is organized into the training and testing directories as specified.
2.	Model Training: Run the provided code to train the model on the dataset.
3.	Model Evaluation: Evaluate the model's performance using the provided test data.

Conclusion
This project demonstrates the use of CNNs, MobileNet, and InceptionV3 for classifying MRI images of brain tumors. By applying data augmentation, using appropriate callbacks, and carefully designing the model architecture, we achieve a reliable model for brain tumor classification.

