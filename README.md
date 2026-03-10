# 🎗️ Breast Cancer Diagnostic AI: High-Precision Clinical System

An end-to-end Machine Learning web application designed to classify breast tumors as Malignant or Benign with high precision. This project leverages an optimized **HistGradientBoostingClassifier** to analyze 30 clinical parameters, providing real-time diagnostic certainty through a sophisticated Cyber-Medical UI.

## 🚀 Live Demo
[🔗 View Live App](https://breastcancerclassification-2fpwga9cmb3tjdjdcnurgr.streamlit.app/) *(Replace with your actual Streamlit link)*

## 🛠️ Tech Stack
- **Engine:** Python 3.10+
- **Machine Learning:** Scikit-Learn (HistGradientBoosting)
- **Web Framework:** Streamlit
- **Data Handling:** Pandas & NumPy
- **Visuals:** Plotly Express & Graph Objects

## 📊 Model Performance
The underlying model was trained on the UCI Breast Cancer Wisconsin dataset, achieving a high degree of predictive stability:
- **Optimization:** L2 Regularization & 500+ Boosting Iterations.
- **Reliability Index:** Real-time probability estimation for every diagnosis.
- **Key Features:** Mean Radius, Texture, Perimeter, Area, and Smoothness (30 total parameters).



## 💡 Features
- **30-Parameter Analysis:** Full-scale clinical input for maximum diagnostic depth.
- **Dynamic Reliability Index:** Shows exactly how certain the AI is about a specific case in the center of the dashboard.
- **Interactive Analytics:** Real-time visualization of dataset distribution and balance.
- **Persistent UI:** Optimized session handling to prevent tab resets during interaction.
- **Cyber-Medical Design:** A high-end dark mode interface with neon-glow sidebar integration.

## 📂 Project Structure
```text
├── app.py                      # Main Streamlit application code
├── requirements.txt            # Required Python libraries (pandas, sklearn, etc.)
└── README.md                   # Project documentation & setup guide
