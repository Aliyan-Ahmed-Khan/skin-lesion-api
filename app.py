from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import base64
import cv2

from tf_explain.core.grad_cam import GradCAM
from disease_info import disease_info

from model import build_model

app = Flask(__name__)

# ---------- SETTINGS ----------

NUM_CLASSES = 39

MODEL_PATH = "Skin_Lesion_Model.keras"

# ---------- LOAD MODEL ----------

print("Loading model...")

model = build_model(NUM_CLASSES)

model.load_weights(MODEL_PATH)

explainer = GradCAM()

print("Model loaded successfully!")



# ---------- LOAD LABELS ----------

def load_labels():

    with open("labels.txt") as f:

        labels = f.read().splitlines()

    return labels

class_names = load_labels()

# ---------- PREPROCESS ----------

def preprocess_image(img):

    img = img.resize((224,224))

    img = np.array(img)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img

def generate_gradcam(img_array, model, class_index):

    # Find last Conv2D layer

    target_layer = None

    for layer in reversed(model.layers):
        if isinstance(layer,
                      tf.keras.layers.Conv2D):
            target_layer = layer.name
            break

    grid = explainer.explain(
        validation_data=(img_array, None),
        model=model,
        layer_name=target_layer,
        class_index=class_index
    )

    # Convert to image buffer

    fig = plt.figure(figsize=(4,4))

    plt.imshow(grid)
    plt.axis('off')

    buf = io.BytesIO()

    plt.savefig(
        buf,
        format='png',
        bbox_inches='tight'
    )

    buf.seek(0)

    image_base64 =base64.b64encode(buf.read()).decode('utf-8')

    buf.close()
    plt.close()

    return image_base64, grid


def analyze_heatmap(grid):

    # Convert heatmap to grayscale if needed

    if len(grid.shape) == 3:
        gray = cv2.cvtColor(
            grid,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = grid

    # Normalize values
    normalized = gray / 255.0

    # Threshold for active regions
    threshold = 0.6

    binary_mask = normalized > threshold

    # Calculate lesion coverage %
    coverage_percent = (
        np.sum(binary_mask)
        / binary_mask.size
    ) * 100

    # Count connected regions
    num_labels, labels = cv2.connectedComponents(binary_mask.astype(np.uint8))

    active_zones = num_labels - 1

    # Determine activation intensity
    max_val = np.max(normalized)

    if max_val > 0.8:
        highest_activation = "High Intensity"
    elif max_val > 0.6:
        highest_activation = "Moderate Intensity"
    else:
        highest_activation = "Low Intensity"

    return {
        "highest_activation": highest_activation,
        "coverage_percent":
            round(coverage_percent, 2),
        "active_zones":
            int(active_zones)
    }


def generate_heatmap_explanation(
    highest_activation,
    coverage_percent,
    active_zones
):

    activation_level = highest_activation.lower()

    if active_zones == 1:
        zone_text = "1 active region"
    else:
        zone_text = f"{active_zones} active regions"

    explanation = (
        f"The Grad-CAM heatmap indicates "
        f"{zone_text} with approximately "
        f"{round(coverage_percent,1)}% lesion coverage "
        f"and {activation_level} activation intensity, "
        f"suggesting focused attention on lesion areas."
    )

    return explanation


# ---------- ROUTE ----------
@app.route('/')

def home():
    return "Flask API Running Successfully"



@app.route('/predict', methods=['POST'])

def predict():

    if 'image' not in request.files:

        return jsonify({
            "error": "No image uploaded"
        })

    file = request.files['image']

    img = Image.open(file.stream)

    img_processed = preprocess_image(img)

    preds = model.predict(img_processed)

    index = np.argmax(preds)

    confidence = float(np.max(preds))

    prediction = class_names[index]

    # Generate Grad-CAM
    heatmap_base64, heatmap_grid = generate_gradcam(
        img_processed,
        model,
        index
    )

    analysis = analyze_heatmap(
    heatmap_grid
    )

    heatmap_explanation = generate_heatmap_explanation(
    analysis["highest_activation"],
    analysis["coverage_percent"],
    analysis["active_zones"]
    )

    info = disease_info.get(
        prediction,
        {
            "severity": "Unknown",
            "description": "Information not available.",
            "recommendation": "Consult dermatologist."
        }
    )


    return jsonify({

    "prediction": prediction,

    "confidence":
        round(confidence * 100, 2),

    "heatmap": heatmap_base64,

    "severity":
        info["severity"],

    "description":
        info["description"],

    "recommendation":
        info["recommendation"],


    # NEW HEATMAP ANALYSIS DATA

    "highest_activation":
        analysis["highest_activation"],

    "coverage_percent":
        analysis["coverage_percent"],

    "active_zones":
        analysis["active_zones"],

    "heatmap_explanation": heatmap_explanation

})

# ---------- RUN ----------

if __name__ == '__main__':

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )