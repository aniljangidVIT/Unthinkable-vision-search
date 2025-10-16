import json
import os
import requests
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import tensorflow as tf

# Disable GPU
tf.config.set_visible_devices([], 'GPU')

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

def get_model():
    base_model = ResNet50(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    return model

def download_image(url, path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

def main():
    products_file = os.path.join(script_dir, 'data/products.json')
    images_dir = os.path.join(script_dir, 'data/images')
    features_file = os.path.join(script_dir, 'data/features.npy')

    os.makedirs(images_dir, exist_ok=True)

    with open(products_file, 'r') as f:
        products = json.load(f)

    model = get_model()

    all_features = []

    for product in products:
        image_name = f"{product['id']}.jpg"
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            print(f"Downloading {product['image_url']}...")
            if not download_image(product['image_url'], image_path):
                continue

        print(f"Extracting features for {image_name}...")
        try:
            features = extract_features(image_path, model)
            all_features.append(features)
        except Exception as e:
            print(f"Error extracting features for {image_name}: {e}")
            # remove the image if it is corrupted
            if os.path.exists(image_path):
                os.remove(image_path)


    np.save(features_file, np.array(all_features))
    print(f"Features saved to {features_file}")

if __name__ == '__main__':
    main()