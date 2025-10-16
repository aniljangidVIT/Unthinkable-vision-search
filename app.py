import os
import json
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf



# Disable GPU
tf.config.set_visible_devices([], 'GPU')

# Backend directory
backend_dir = os.path.dirname(os.path.abspath(__file__))

# Frontend directory (outside backend folder)
frontend_dir = os.path.join(backend_dir, '..', 'frontend')

app = Flask(__name__)
CORS(app)

# Load the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

# Load pre-computed features
features_path = os.path.join(backend_dir, 'data/features.npy')
product_features = np.load(features_path)

# Load product data
products_path = os.path.join(backend_dir, 'data/products.json')
with open(products_path, 'r') as f:
    products = json.load(f)

def extract_features(img, model):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

# Serve frontend index.html
@app.route('/')
def index():
    return send_from_directory(frontend_dir, 'index.html')

# Serve other frontend static files (JS, CSS)
@app.route('/<path:path>')
def serve_frontend(path):
    return send_from_directory(frontend_dir, path)

# Image search API
@app.route('/api/search', methods=['GET', 'POST'])
def search():
    if request.method == 'GET':
        return (
            "<h3>POST an image to this endpoint to search for similar products.</h3>"
            "<p>Use the 'file' key in form-data for the image.</p>"
        )

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(file.stream).convert("RGB")

        # Extract features from the uploaded image
        query_features = extract_features(img, model)

        # Calculate cosine similarity
        similarities = cosine_similarity([query_features], product_features)[0]

        # Get the top 5 most similar products
        top_5_indices = similarities.argsort()[-5:][::-1]

        results = []
        for i in top_5_indices:
            results.append({
                'product': products[i],
                'similarity': float(similarities[i])
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
