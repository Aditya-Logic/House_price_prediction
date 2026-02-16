from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_FILE = 'model_new.pkl'
PIPELINE_FILE = 'pipeline_new.pkl'

# --- LOAD MODEL (Run once at startup) ---
print("Loading model and pipeline...")
if os.path.exists(MODEL_FILE) and os.path.exists(PIPELINE_FILE):
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    print("Model loaded!")
else:
    print("WARNING: Model files not found.")
    model, pipeline = None, None

@app.route("/", methods=['GET'])
def home():
    # Just show the empty form
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if not model or not pipeline:
        return "Model not loaded. Please train it first."

    try:
        # 1. Grab data from the HTML form
        # We create a dictionary that matches the structure your pipeline expects
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

        # 2. Convert to DataFrame
        input_df = pd.DataFrame(form_data)

        # 3. Transform and Predict
        prepared_data = pipeline.transform(input_df)
        prediction = model.predict(prepared_data)[0]

        # 4. Show the result on the page
        result_string = f"Estimated Value: ${prediction:,.2f}"
        return render_template('index.html', prediction_text=result_string)
 
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))