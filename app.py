import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFile
import json
import redis
from werkzeug.utils import secure_filename
import warnings
import traceback

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)

# ================= UPLOAD FOLDER =================
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= REDIS =================
redis_client = redis.Redis(
    host='127.0.0.1',
    port=6379,
    decode_responses=True
)

food_data = {}

try:
    food_data_json = redis_client.get("food_details")
    if food_data_json:
        food_data = json.loads(food_data_json)
        print("✅ Food data loaded from Redis")
    else:
        print("⚠️ No food data found in Redis")
except Exception as e:
    print("Redis Error:", e)

# ================= LOAD MODELS =================
print("Loading models...")

custom_cnn_model = tf.keras.models.load_model('models/custom_weights.keras')
resnet_model     = tf.keras.models.load_model('models/resnet_weights.keras')
vgg16_model      = tf.keras.models.load_model('models/vgg16_weights.keras')

print("✅ Models Loaded Successfully")

# ================= CLASS MAPPING =================
with open('json/class_indices.json', 'r') as f:
    class_indices = json.load(f)

class_names      = {v: k for k, v in class_indices.items()}
class_names_list = list(class_names.values())

# ================= LOAD METRICS =================
with open('json/custom_CNN_metrics.json', 'r') as f:
    custom_metrics = json.load(f)

with open('json/resnet_CNN_metrics.json', 'r') as f:
    resnet_metrics = json.load(f)

with open('json/vgg16_CNN_metrics.json', 'r') as f:
    vgg16_metrics = json.load(f)

print("\n📋 custom_metrics top-level keys:", list(custom_metrics.keys())[:5])
print("📋 resnet_metrics top-level keys:", list(resnet_metrics.keys())[:5])
print("📋 vgg16_metrics top-level keys:", list(vgg16_metrics.keys())[:5])


# ================= HELPER: Extract Class Metrics =================
def extract_class_metrics(metrics_data, predicted_class):
    print(f"\n🔍 Searching metrics for: '{predicted_class}'")

    # Direct match
    block = metrics_data.get(predicted_class, {})

    # Case-insensitive match if direct not found
    if not block:
        for key in metrics_data:
            if key.lower() == predicted_class.lower():
                block = metrics_data[key]
                print(f"   ✅ Case-insensitive match found: '{key}'")
                break

    if block and "classification_report" in block:
        report = block["classification_report"]
        print(f"   ✅ Found classification_report: {report}")
        return {
            "precision": round(float(report.get("precision", 0)), 2),
            "recall":    round(float(report.get("recall", 0)), 2),
            "f1-score":  round(float(report.get("f1-score", 0)), 2)
        }

    print(f"   ❌ Not found for '{predicted_class}'")
    print(f"   Available keys: {list(metrics_data.keys())[:10]}")
    return {"precision": "N/A", "recall": "N/A", "f1-score": "N/A"}


# ================= HELPER: Extract Accuracy =================
def extract_accuracy(metrics_data, predicted_class):
    print(f"\n🔍 Searching accuracy for: '{predicted_class}'")

    # Direct match
    block = metrics_data.get(predicted_class, {})

    # Case-insensitive match if direct not found
    if not block:
        for key in metrics_data:
            if key.lower() == predicted_class.lower():
                block = metrics_data[key]
                print(f"   ✅ Case-insensitive match found: '{key}'")
                break

    if block and "overall_model_accuracy" in block:
        val = float(block["overall_model_accuracy"])
        result = round(val * 100, 2)
        print(f"   ✅ Accuracy found: {result}%")
        return result

    print(f"   ❌ Accuracy not found for '{predicted_class}'")
    return "N/A"


# ================= IMAGE PREPROCESS =================
def preprocess_image(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# ================= HOME =================
@app.route('/')
def home():
    return render_template('index.html', class_names=class_names_list)


# ================= PREDICT =================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"})

        image      = request.files['image']
        model_name = request.form.get('model')

        filename = secure_filename(image.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)

        img = preprocess_image(filepath)

        # ================= MODEL SELECTION =================
        if model_name == 'customcnn':
            model        = custom_cnn_model
            metrics_data = custom_metrics

        elif model_name == 'resnet':
            model        = resnet_model
            metrics_data = resnet_metrics

        elif model_name == 'vgg16':
            model        = vgg16_model
            metrics_data = vgg16_metrics

        else:
            return jsonify({"error": "Invalid model selected"})

        # ================= PREDICTION =================
        pred = model.predict(img)

        index           = int(np.argmax(pred))
        confidence      = round(float(np.max(pred)) * 100, 2)
        predicted_class = class_names.get(index, "Unknown")
        detected_image  = "/" + filepath

        print(f"\n✅ Predicted: {predicted_class} | Confidence: {confidence}%")

        # ================= METRICS =================
        class_report = extract_class_metrics(metrics_data, predicted_class)
        accuracy     = extract_accuracy(metrics_data, predicted_class)

        print(f"   Metrics  → {class_report}")
        print(f"   Accuracy → {accuracy}%")

        # ================= NUTRITION =================
        # ================= NUTRITION =================
        nutrition = (
                food_data.get(predicted_class)
                or food_data.get(predicted_class.lower())
                or {
                    "calories": "N/A",
                    "protein": "N/A",
                    "fat": "N/A",
                    "carbohydrates": "N/A",
                    "fiber": "N/A"
                }
        )

        # Handle capital/small keys safely
        nutrition = {
            "calories": nutrition.get("calories") or nutrition.get("Calories", "N/A"),

            "protein": nutrition.get("protein") or nutrition.get("Protein", "N/A"),

            "fat": nutrition.get("fat") or nutrition.get("Fat", "N/A"),

            # FIXED HERE
            "carbohydrates": (
                    nutrition.get("carbohydrates")
                    or nutrition.get("Carbohydrates")
                    or nutrition.get("carbs")
                    or nutrition.get("Carbs")
                    or "N/A"
            ),

            "fiber": nutrition.get("fiber") or nutrition.get("Fiber", "N/A")
        }

        # ================= RESPONSE =================
        return jsonify({
            "predicted_class": predicted_class,
            "confidence":      confidence,
            "accuracy":        accuracy,
            "detected_image":  detected_image,
            "nutrition":       nutrition,
            "class_report":    class_report
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


# ================= RUN =================
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)