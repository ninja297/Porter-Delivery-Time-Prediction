# 🚚 Porter Delivery Time Prediction

An AI-driven food delivery time prediction system using a Neural Network-based Regression model. This project aims to estimate the time required to deliver an order based on order details, restaurant information, and delivery partner availability.

## 📌 Project Overview

Porter is an intra-city logistics platform connecting customers, restaurants, and delivery partners. Accurate delivery time prediction can improve customer experience, optimize delivery partner allocation, and reduce delivery delays.

This project uses historical delivery data to build a regression model that predicts delivery time in minutes.

## 🎯 Objectives

- Analyze and preprocess the Porter delivery dataset.
- Perform Exploratory Data Analysis (EDA).
- Engineer meaningful features from order timestamps.
- Develop a Neural Network-based regression model.
- Evaluate model performance using regression metrics.
- Deploy the trained model using a Streamlit web application.

## 📂 Dataset Description

The dataset contains information about individual deliveries.

**Dataset Source:** [Porter Delivery Dataset](https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/015/039/original/dataset.csv.zip?1663710760)

Important features include:

| Feature | Description |
|---|---|
| market_id | Market identifier |
| created_at | Order creation timestamp |
| store_id | Restaurant/store identifier |
| actual_delivery_time | Actual delivery timestamp |
| store_primary_category | Restaurant category |
| order_protocol | Order placement protocol |
| total_items | Total number of items |
| subtotal | Final order price |
| num_distinct_items | Number of distinct items |
| min_item_price | Minimum item price |
| max_item_price | Maximum item price |
| total_onshift_partners | Delivery partners on shift |
| total_busy_partners | Busy delivery partners |
| total_outstanding_orders | Pending orders |

## ⚙️ Technologies Used

- Python
- Google Colab
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Joblib

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Loaded and explored the dataset.
2. Identified missing values and duplicate records.
3. Converted timestamp columns into datetime format.
4. Calculated delivery time in minutes.

   `delivery_time = (actual_delivery_time - created_at) / 60`

5. Removed invalid delivery time records.
6. Extracted temporal features such as:
   - Order hour
   - Order day
   - Order month
   - Order weekday
   - Weekend indicator
7. Handled missing values using imputation.
8. Applied One-Hot Encoding to categorical features.
9. Applied Standard Scaling to numerical features.

## 🧠 Neural Network Architecture

A Feed-Forward Artificial Neural Network was developed using TensorFlow/Keras.

**Model Architecture:**

- Input Layer
- Dense Layer: 128 neurons, ReLU activation
- Dropout Layer: 0.2
- Dense Layer: 64 neurons, ReLU activation
- Dropout Layer: 0.2
- Dense Layer: 32 neurons, ReLU activation
- Output Layer: 1 neuron, Linear activation

**Model Configuration:**

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Mean Squared Error |
| Evaluation Metric | Mean Absolute Error |
| Batch Size | 64 |
| Maximum Epochs | 100 |
| Train-Test Split | 80:20 |
| Early Stopping | Enabled |

## 📊 Model Evaluation

The model is evaluated using the following regression metrics:

- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **R² Score:** Coefficient of Determination

Actual evaluation results can be added here after model training.

## 🌐 Web Application

The trained Neural Network is deployed using Streamlit.

The web application allows users to:

1. Enter order-related information.
2. Submit the input features.
3. Process the inputs using the saved preprocessing pipeline.
4. Generate delivery time predictions.
5. Display the estimated delivery time in minutes.

## 📁 Project Structure

```text
Porter-Delivery-Time-Prediction/
│
├── app.py
├── porter_delivery_model.keras
├── preprocessor.pkl
├── schema.json
├── requirements.txt
│
└── README.md
```

## 🚀 How to Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/Porter-Delivery-Time-Prediction.git
```

**2. Navigate to the project directory**

```bash
cd Porter-Delivery-Time-Prediction
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the Streamlit application**

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔮 Future Improvements

- Experiment with different Neural Network architectures.
- Perform hyperparameter tuning.
- Compare ANN performance with traditional regression models.
- Incorporate additional traffic and weather information.
- Improve prediction accuracy through advanced feature engineering.
- Develop a more interactive prediction dashboard.

## 👩‍💻 Project Information

**Course:** AIMLDL Application (CS361)  
**Assignment:** Lab Assignment-II  
**Institution:** Rajiv Gandhi Institute of Petroleum Technology (RGIPT)

---

*Developed as part of the AIMLDL Application Lab Assignment-II.*
