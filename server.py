from flask import Flask, request, jsonify
from flask_cors import CORS
from model import Model
import traceback
import torch

app = Flask(__name__)

CORS(app)

model = Model()
model.load_state_dict(torch.load('trained.pth', weights_only = True))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        predicted_class, confidence = model.predict(file)
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting web server on http://localhost:5000")
    app.run(debug = True, port = 5000)
