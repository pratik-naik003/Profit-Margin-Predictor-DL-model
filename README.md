📈 Profit Margin Prediction using Deep Learning

🔗 Open in Colab:
"Click here to run the notebook" (https://colab.research.google.com/github/Pratik872-bit/Profit-Margin-Predictor-DL-model/blob/main/profit_margin_predictor.ipynb)

---

📌 Project Overview

This project predicts profit margin using a Deep Learning regression model built with TensorFlow (Keras).
The model is trained on a Superstore sales dataset containing transactional, sales, and pricing information.

The entire workflow is implemented and executed on Google Colab, with the dataset loaded directly from Google Drive.

---

📊 Dataset Description

Dataset Name: Superstore Dataset
Total Rows: 9,994
Total Columns: 19

Key Columns Used

Column Name| Description
sales| Total sales value
quantity| Number of items sold
discount| Discount applied
profit| Net profit
profit_margin| Profit margin (Target Variable)

🔹 The target variable "profit_margin" is predicted using numerical sales-related features.

---

🧠 Model Architecture

The Deep Learning model is a fully connected neural network (Regression):

- Input Layer → Feature size based on dataset
- Dense Layer (128 neurons, ReLU)
- Dense Layer (64 neurons, ReLU)
- Dense Layer (32 neurons, ReLU)
- Output Layer (1 neuron, Linear activation)

---

⚙️ Technologies Used

- Python
- Pandas
- NumPy
- TensorFlow (Keras)
- Scikit-learn
- Google Colab

---

🔄 Workflow

1. Load dataset from Google Drive
2. Select input features and target variable
3. Split data into training and testing sets
4. Build Deep Learning regression model
5. Compile model using:
   - Optimizer: Adam
   - Loss: Mean Absolute Error (MAE)
6. Train model for 10 epochs
7. Validate model performance

---

📈 Training Results

- Loss Function: MAE
- Final Validation MAE: ~0.017
- Model shows stable learning and good convergence

---

🚀 How to Run

1. Open the Colab notebook using the link above
2. Upload "superstore_dataset.csv" to your Google Drive
3. Update the dataset path if needed
4. Run all cells sequentially

---

📌 Future Improvements

- Feature scaling using StandardScaler
- One-hot encoding for categorical features
- Hyperparameter tuning
- Model evaluation using RMSE & R² score
- Deployment using Streamlit or FastAPI

---

👨‍💻 Author

Pratik Naik
B.Tech AIML Student
Walchand College of Engineering, Sangli

---

⭐ If you found this project helpful, don’t forget to star the repository!