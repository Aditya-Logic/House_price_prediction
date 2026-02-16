from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import gdown

app = Flask(__name__)

MODEL_FILE = 'model_new.pkl'
PIPELINE_FILE = 'pipeline_new.pkl'

FILE_ID = "1Cg7JchDLI1f4OOFZhDAh22FZgm7-Hm-A"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

print("Loading model and pipeline...")

def download_model():
    if not os.path.exists(MODEL_FILE):
        print("Downloading model...")
        gdown.download(URL, MODEL_FILE, quiet=False, fuzzy=True)

download_model()

try:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    print("Model and pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading files: {e}")
    model = None
    pipeline = None


@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if model is None or pipeline is None:
        return "Model not loaded. Please check deployment."

    try:
        form_data = {
            'longitude': [float(request.form['longitude'])],
            'latitude': [float(request.form['latitude'])],
            'housing_median_age': [float(request.form['housing_median_age'])],
            'total_rooms': [float(request.form['total_rooms'])],
            'total_bedrooms': [float(request.form['total_bedrooms'])],
            'population': [float(request.form['population'])],
            'households': [float(request.form['households'])],
            'median_income': [float(request.form['median_income'])],
            'ocean_proximity': [request.form['ocean_proximity']]
        }

        input_df = pd.DataFrame(form_data)
        prepared_data = pipeline.transform(input_df)
        prediction = model.predict(prepared_data)[0]

        result_string = f"Estimated Value: ${prediction:,.2f}"
        return render_template('index.html', prediction_text=result_string)

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))