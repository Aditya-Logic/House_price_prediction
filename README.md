# HOUSE_price_prediction
# 🏠 House Price Prediction Web App (Flask + Machine Learning)

**Live Demo:** https://house-price-prediction-hfkv.onrender.com

## 📌 Project Overview

This project is a **Machine Learning–based House Price Prediction Web Application** built using **Python, Scikit-Learn, and Flask**.

The system trains a regression model on housing data and allows users to input property details through a web interface to estimate the **median house value**.

The application demonstrates the complete ML lifecycle:

* Data preprocessing with pipelines
* Model training and saving
* Model loading and inference
* Web deployment using Flask

---

## 🚀 Features

* Stratified sampling for better training distribution
* Automated preprocessing using Scikit-Learn Pipelines
* Random Forest Regression model
* Model persistence using Joblib (`.pkl` files)
* User-friendly web interface for predictions
* CSV test prediction generation

---

## 🧠 Technologies Used

* Python
* Pandas, NumPy
* Scikit-Learn
* Flask
* Joblib
* HTML (Frontend form)

---

## 📂 Project Structure

```
project/
│── main.py                # Model training and batch inference
│── flask.py               # Flask web application
│── model_new.pkl          # Saved trained model
│── pipeline_new.pkl       # Saved preprocessing pipeline
│── housing.csv            # Dataset
│── test.csv               # Test data for inference
│── output_test.csv        # Prediction results
│── templates/
│     └── index.html       # Web interface
│── README.md              # Project documentation
```

---



###  Clone Repository

```bash
git clone https://github.com/Aditya-Logic/House-price-prediction.git
cd House-price-prediction
```

## 🏋️ Model Training

Run the training script:

```bash
python main.py
```

This will:

* Train the model
* Save `model_new.pkl` and `pipeline_new.pkl`
* Generate test predictions

---

## 🌐 Running the Web App

Start Flask server:

```bash
python flask.py
```

Then open browser:

```
http://127.0.0.1:5000
```

---

## 📝 Input Features

The model uses the following features:

* Longitude
* Latitude
* Housing Median Age
* Total Rooms
* Total Bedrooms
* Population
* Households
* Median Income
* Ocean Proximity (Categorical)

---

## 📊 Model Details

* Algorithm: **Random Forest Regressor**
* Preprocessing:

  * Median imputation
  * Standard scaling
  * One-Hot encoding for categorical features

---

## 📸 Output Example

The application returns an estimated house price like:

```
Estimated Value: $245,300.45
```

---

## 🔮 Future Improvements

* Deploy on cloud (Render / Heroku / AWS)
* Add model evaluation metrics on UI
* Improve UI design
* Add multiple model comparison
* Real-time API endpoint

---

## 👨‍💻 Author

Aditya Verma

---

## 📜 License

This project is for educational purposes.
