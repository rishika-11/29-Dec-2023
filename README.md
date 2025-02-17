# FACE EMOTION RECOGNITION USING FLASK

# Introduction
The project aims to integrate deep learning algorithms for real-time facial emotion recognition.

# Dataset
The project utilizes the FER 2013 dataset obtained from Kaggle. This dataset consists of 48x48 pixel grayscale images of faces, automatically registered for centered positioning. The dataset enables training a deep learning model to recognize facial expressions accurately.

# Dependencies
Ensure the installation of the following libraries before running the Colab notebook:

- numpy
- tensorflow
- opencv-python
- streamlit
- flask
  

# Data Preparation

Download the FER 2013 dataset from Kaggle.
Define training and validation sets.
Preprocess the data by rescaling and augmenting to enhance variability.

Build a Convolutional Neural Network (CNN) using TensorFlow and Keras.
The model comprises convolutional layers, batch normalization, max pooling, dropout, and fully connected layers.
Adam optimizer is used with categorical crossentropy loss for compilation.

Train the model using the prepared datasets.
Evaluate accuracy on both the training and validation sets.
Analyze the confusion matrix to identify areas of improvement.
Note that lower performance in classes like 'angry' and 'fear' may be attributed to limited data for these classes.

Utilize OpenCV and Streamlit to create a web app.
The app monitors live facial emotion recognition, successfully identifying and displaying emotions in real-time.
Multiple faces and their respective emotions can be detected simultaneously.

Collab Notebook link:https://colab.research.google.com/drive/1EqfDLy-B6yGtM2Nzxm3YbwbwmPIMDY9l?usp=sharing

# Using Flask
The provided code and model constitute an Emotion Detection system implemented as a web application. The frontend, designed using HTML and JavaScript, offers a user-friendly interface with buttons to capture an image and initiate emotion detection. The backend, powered by Flask in Python, integrates a pre-trained deep learning model for recognizing facial emotions. The model, loaded from the 'emotion_model.h5' file, is a Convolutional Neural Network (CNN) trained on the FER 2013 dataset, capable of categorizing emotions such as Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

The image capturing process utilizes the device's camera, allowing users to dynamically capture images. These images are then processed and sent to the backend for emotion detection through a POST request. The backend decodes the base64-encoded image, preprocesses it, and feeds it into the loaded model. The model predicts the dominant emotion, and the result is displayed in real-time on the web page.

Webpage link:http://127.0.0.1:5000
