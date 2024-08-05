import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

app = Flask(__name__)

# Path to the .h5 model file
model_path = r'./model/model.h5'
model = load_model(model_path)

UPLOADS_FOLDER = '/tmp/uploads'

# Ensure the uploads directory exists
if not os.path.exists(UPLOADS_FOLDER):
    os.makedirs(UPLOADS_FOLDER)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file:
        filepath = os.path.join(UPLOADS_FOLDER, file.filename)
        file.save(filepath)

        try:
            # Load and preprocess the image
            image = load_img(filepath, target_size=(150, 150))  # Adjust target_size as needed
            img_array = img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Predict using the model
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))

            # Clean up: remove the temporary file
            if os.path.exists(filepath):
                os.remove(filepath)

            return jsonify({'class': int(predicted_class), 'confidence': confidence})

        except Exception as e:
            # Handle any exceptions during prediction
            return jsonify({'error': str(e)}), 400

    return jsonify({'error': 'File not readable'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Set debug=True for development

