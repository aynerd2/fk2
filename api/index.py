import os
from flask import Flask, request, jsonify
from transformers import AutoModelForImageClassification, AutoProcessor
import torch
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_id = "Aynerd/NiaraApi"
model = AutoModel.from_pretrained(model_id)

AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

UPLOADS_FOLDER = '/tmp/uploads'

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

        image = Image.open(filepath)
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits

        predicted_class = torch.argmax(predictions)
        confidence = torch.max(predictions)

        os.remove(filepath)

        return jsonify({'class': predicted_class.item(), 'confidence': confidence.item()})

    return jsonify({'error': 'File not readable'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
