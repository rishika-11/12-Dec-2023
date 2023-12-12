# Epileptic Seizure Recognition

# Dataset 
- The dataset likely contains various features or attributes that describe characteristics of EEG (Electroencephalogram) signals.
- 4097 data points into 23 chunks, each chunk contains 178 data points for 1 second, and each data point is the value of the EEG recording at a different point in 
  time. So now we have 23 x 500 = 11500 pieces of information(row), each information contains 178 data points for 1 second(column), the last column represents the 
  label y {1,2,3,4,5}.
- The response variable is y in column 179, the Explanatory variables X1, X2, …, X178
- y contains the category of the 178-dimensional input vector. Specifically y in {1, 2, 3, 4, 5}:
-  Dataset link :
   https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition


# Overview
The code loads an Epileptic Seizure Recognition dataset, preprocesses the data by handling class imbalances, visualizes samples, and splits it into training and testing sets. It then trains various machine learning models (Logistic Regression, SVM, KNN, Naive Bayes, ANN) and evaluates their performance, including PCA for dimensionality reduction. Finally, the code visualizes PCA components and presents a summary of model scores.


# Libraries
- Numpy
- Matplotlib.pyplot
- Pandas
- Seaborn
- Sklearn
- Keras


# Data Preprocessing
- Reading the "Epileptic Seizure Recognition.csv" dataset using pd.read_csv().
- Converting values greater than 1 in the target variable (y) to 0 for binary classification.
- Plotting a count plot of the target variable classes.
- Displaying the first few rows of the dataset.
- Visualizing a subset of samples for each class.
- Splitting the dataset into features (X) and target variable (y).
- Standardizing the features using StandardScaler.
- Applying PCA for dimensionality reduction.
- Transforming the training and testing sets using PCA.


# Model Evaluation
- Logistic Regression 
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Artificial Neural Network (ANN)
- Principal Component Analysis (PCA)


# Artificial Neural Network using Keras
- The Sequential class is used to create a linear stack of layers for building the neural network model.
- The model has three dense layers. The first two layers use the ReLU activation function, and the last layer uses the sigmoid activation function for binary 
  classification.
- The model is compiled with the Adam optimizer, binary crossentropy loss, and accuracy as the metric.
- The model is trained on the training set (X_train and y_train) using the fit method.
- Predictions are made on the test set using the predict method.
- The accuracy of the model is calculated.



