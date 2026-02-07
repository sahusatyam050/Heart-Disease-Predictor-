# 🫀 Heart Disease Predictor

A machine learning-powered web application to predict heart disease risk based on patient health metrics.

## 🎯 Project Overview

This project uses machine learning algorithms to predict the likelihood of heart disease based on various health indicators such as age, blood pressure, cholesterol levels, and more. The application features an interactive Streamlit web interface for easy use.

## 📁 Project Structure

```
Heart_Disease_Predictor/
├── data/                  # Dataset files
├── models/                # Trained ML models
├── notebooks/             # Jupyter notebooks for exploration
├── app.py                 # Streamlit web application
├── train_model.py         # Model training script
├── predict.py             # Prediction functions
├── preprocessing.py       # Data preprocessing utilities
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd Heart_Disease_Predictor
   ```

2. **Activate the virtual environment**
   ```bash
   source venv/bin/activate  # On Linux/Mac
   # OR
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Train the model** (first time only)
   ```bash
   python train_model.py
   ```

2. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

3. **Access the app** in your browser at `http://localhost:8501`

## 📊 Dataset

The project uses the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease) containing patient health metrics and heart disease diagnoses.

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: Machine learning algorithms
- **Streamlit**: Web application framework
- **Matplotlib & Seaborn**: Data visualization

## 🔮 Features

- ✅ Interactive web interface
- ✅ Real-time predictions
- ✅ Data visualization
- ✅ Model performance metrics
- ✅ User-friendly input forms

## 📝 License

This project is open source and available for educational purposes.

## 👨‍💻 Author

Built as a machine learning portfolio project.

---

**Note**: This is a demonstration project and should not be used for actual medical diagnosis. Always consult healthcare professionals for medical advice.
