import numpy as np
from tensorflow.keras.utils import img_to_array
from PIL import Image

def preprocess_image(image_input, target_size=(224, 224)):
    """
    Preprocess the image for the model.
    Supports both file paths and BytesIO objects as input.

    Parameters:
        image_input (str or BytesIO): The image file path or BytesIO object.
        target_size (tuple): The desired size for the image (width, height).

    Returns:
        np.ndarray: Preprocessed image ready for model prediction.
    """
    # Handle both BytesIO objects and file paths
    if isinstance(image_input, str):  # If file path
        img = Image.open(image_input)
    else:  # If BytesIO object
        img = Image.open(image_input)
    
    # Convert to RGB to handle grayscale or other formats
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize the image to the target size
    img = img.resize(target_size)
    
    # Convert image to a numpy array
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    
    # Add batch dimension
    return np.expand_dims(img_array, axis=0)
