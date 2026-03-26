from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False

app = Flask(__name__)

# Upload folder config
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = "model.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        if not GDOWN_AVAILABLE:
            raise RuntimeError(
                "model.h5 not found and gdown is not installed. "
                "Install gdown with: pip install gdown"
            )
        print("Downloading model...")
        file_id = "1wSRB2XOzB2uHj5Ecc5_F9GyvoiNJI6vy"
        # Using id=file_id directly is more reliable for large files in newer gdown versions
        gdown.download(id=file_id, output=MODEL_PATH, quiet=False)
        print("Model downloaded.")

# Call this before model load
download_model()

# Handle Keras version mismatches (quantization_config error on Render)
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

class CustomDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

# Now you can safely load the model
model = load_model(MODEL_PATH, custom_objects={'Dense': CustomDense}, compile=False)



# Class labels
class_names = {
    0: "Actinic Keratosis",
    1: "Acne",
    2: "Basal Cell Carcinoma",
    3: "Dermatofibroma",
    4: "Melanoma",
    5: "Melanocytic Nevus",
    6: "Squamous Cell Carcinoma",
    7: "Seborrheic Keratosis",
    8: "Vascular Lesions"
}

# Home route
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/disease")
def disease():
    return render_template("diseases.html")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")
# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('prediction'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        preds = model.predict(img_array)
        pred_index = np.argmax(preds)
        pred_class = class_names[pred_index]

        image_url = url_for('static', filename='uploads/' + filename)
        return render_template('prediction.html', prediction=pred_class, image_url=image_url)

    return "Something went wrong"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
