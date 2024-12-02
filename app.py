from flask import Flask, request, render_template, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from preprocess import preprocess_image

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model('model/dog_breed_model.h5')

# Load the breed info dataset
breed_info = pd.read_csv('dataset/dogs-ranking-dataset.csv')

# Automatically generate class indices by parsing folder names
image_folder = 'dataset/Images'
class_indices = {
    folder.split('-')[-1]: idx
    for idx, folder in enumerate(sorted(os.listdir(image_folder)))
}
reverse_class_indices = {v: k for k, v in class_indices.items()}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocess and predict
    img_array = preprocess_image(file_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    breed_name = reverse_class_indices[predicted_class]

    # Fetch breed info if available
    breed_data = breed_info[breed_info['Breed'] == breed_name]
    if not breed_data.empty:
        breed_details = breed_data.to_dict(orient='records')[0]
    else:
        breed_details = None

    return jsonify({
        'breed': breed_name,
        'confidence': float(predictions[0][predicted_class]),
        'breed_details': breed_details,
    })

if __name__ == '__main__':
    app.run(debug=True)
