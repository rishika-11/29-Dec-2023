from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import base64

app = Flask(__name__)
app.static_folder = 'static'  # Set the static folder
emotion_model = load_model('emotion_model.h5')  # Load  trained model

# Function to preprocess the image before feeding it to the model
def preprocess_image(base64_image):
    img_data = base64.b64decode(base64_image)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    img = img / 255.0  # Normalize pixel values
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    # Handle the captured image and return its path
    # This could involve saving the image on the server and returning the path
    return jsonify({'image_path': 'path/to/captured/image.jpg'})

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    try:
        # Receive base64-encoded image from the client
        image_base64 = request.json.get('image_base64')

        # Preprocess the image
        img = preprocess_image(image_base64)

        # Make a prediction
        predictions = emotion_model.predict(img)
        emotion_label = np.argmax(predictions)

        # Map the numeric label to the corresponding emotion
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        detected_emotion = emotions[emotion_label]

        # Return the detected emotion
        return jsonify({'emotion': detected_emotion})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
