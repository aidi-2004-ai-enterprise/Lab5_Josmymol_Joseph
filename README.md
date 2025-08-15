# Lab 4: Binary Classification

This project is part of **AIDI 2004 - AI Enterprise** course. The objective of this lab is to perform binary classification using multiple machine learning models, evaluate their performance, and compare results with a benchmark model.

---

## 📌 Project Overview
- **Task:** Predict target labels (binary classification) using machine learning algorithms.
- **Dataset:** Provided CSV dataset (loaded locally in the script).
- **Goal:** Experiment with multiple models, compare their performance, and recommend the best one for deployment.

---

## ⚙️ Implementation Steps
1. **Data Loading and Exploration**
   - Load dataset using pandas.
   - Perform exploratory data analysis (EDA) including missing value check, summary statistics, and target distribution.

2. **Data Preprocessing**
   - Encode categorical variables.
   - Normalize numerical features.
   - Split dataset into training and test sets.

3. **Model Selection**
   - Benchmark Model: **Logistic Regression**
   - Other Models:
     - Decision Tree
     - Random Forest
     - XGBoost
     - Support Vector Machine (SVM)
     - K-Nearest Neighbors (KNN)

4. **Model Training & Evaluation**
   - Train models on the training set.
   - Evaluate models using accuracy, precision, recall, and F1-score.
   - Compare results in a summary table.

5. **Final Recommendation**
   - Select the model with the best balance of accuracy and recall for deployment.

---

## 📊 Results Summary
| Model                | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 0.87     | 0.80      | 0.76   | 0.78     |
| Decision Tree         | 0.84     | 0.79      | 0.71   | 0.75     |
| Random Forest         | 0.92     | 0.87      | 0.83   | 0.85     |
| XGBoost               | 0.93     | 0.89      | 0.85   | 0.87     |
| SVM                   | 0.90     | 0.85      | 0.82   | 0.83     |
| KNN                   | 0.88     | 0.82      | 0.78   | 0.80     |

➡️ **Best Model:** XGBoost (high accuracy, precision, and recall balance).

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Lab4-Binary-Classification.git
   cd Lab4-Binary-Classification
   ```
# Install dependencies:

```bash

pip install -r requirements.txt
```
# Run the script:

```bash
python test.py
```
# 📌 Challenges Faced
- Handling missing data and categorical encoding.

- Model tuning for improved performance.

- Preventing overfitting in tree-based models.

# 📌 Future Work
Extend research to Lab 5 for multi-class classification.

Implement hyperparameter tuning with GridSearchCV.

Deploy best-performing model using FastAPI/Flask.
