

# Traffic Accident Severity Prediction  

This project leverages machine learning to predict the severity of traffic accidents based on various factors, such as weather conditions, time of day, and location. It uses the dataset available [here](https://drive.google.com/file/d/1edKrdWNOcgbAo2JtckX-PEyM0FdEq4EG/view) and explores several models to determine the best-performing algorithm for this task.  

## Table of Contents  
1. [Overview](#overview)  
2. [Dataset](#dataset)  
3. [Technologies Used](#technologies-used)  
4. [Preprocessing and Feature Engineering](#preprocessing-and-feature-engineering)  
5. [Model Training and Evaluation](#model-training-and-evaluation)  
6. [Results](#results)  
7. [How to Run](#how-to-run)  

---

## Overview  
Traffic accidents are a major safety concern. Predicting accident severity can help mitigate risks by identifying high-risk factors and enhancing traffic management.  
This project implements:  
- Data preprocessing and feature engineering.  
- Exploratory Data Analysis (EDA).  
- Training multiple machine learning models to find the best-performing one.  

---

## Dataset  
The dataset contains information on traffic accidents, including time, location, weather conditions, and severity. You can download the dataset from [this link](https://drive.google.com/file/d/1edKrdWNOcgbAo2JtckX-PEyM0FdEq4EG/view).  

### Key Features:  
- **Start_Time** and **End_Time**: The start and end time of accidents.  
- **Weather_Condition**: The weather at the time of the accident.  
- **Severity**: The target variable, representing the severity of the accident (1–4).  

---

## Technologies Used  
- **Python**  
- **Libraries**:  
  - Pandas  
  - NumPy  
  - Matplotlib  
  - Scikit-learn  

---

## Preprocessing and Feature Engineering  
1. **Data Cleaning**:  
   - Handled missing values in key columns such as `End_Lat`, `End_Lng`.  
   - Converted `Start_Time` and `End_Time` to datetime objects.  
2. **Feature Engineering**:  
   - Created a `Duration` feature to capture accident duration in minutes.  
   - Encoded categorical variables like `Weather_Condition` using OneHotEncoding.  
3. **Normalization**: Standardized numerical features like `Distance(mi)` and `Temperature(F)`.  

---

## Model Training and Evaluation  
To ensure efficient training:  
1. A **sample space** was created from the large dataset.  
2. Multiple models were trained on this subset, and the best-performing model was selected for full training.  

### Models Evaluated:  
- **Random Forest**  
- **Gradient Boosting**  
- **Support Vector Machine (SVM)**  

### Evaluation Metrics:  
- Accuracy  
- F1 Score  
- ROC-AUC  

---

## Results  
The **Random Forest** model performed best on the sampled data and was selected for full training.  

### Random Forest Performance:  
- **Accuracy**: 76.29%  
- **F1 Score**: 70.14%  
- **ROC-AUC**: 73.72%  

#### Classification Report (Random Forest):  
```
               precision    recall  f1-score   support

           1       0.27      0.02      0.03       168
           2       0.79      0.96      0.86     13585
           3       0.35      0.10      0.15      3248
           4       0.14      0.01      0.02       462

    accuracy                           0.76     17463
   macro avg       0.39      0.27      0.27     17463
weighted avg       0.68      0.76      0.70     17463
```  

---

## How to Run  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/your-username/traffic-accident-severity.git  
   cd traffic-accident-severity  

2. Download the dataset and place it in the project directory.  

3. Run the preprocessing and training script:  
   ```bash  
   python train_models.py  
   ```  

4. View the results and reports in the terminal output.  

---

## Future Work  
- Hyperparameter tuning to improve model performance.  
- Integration of deep learning models for severity prediction.  
- Deployment of the model as an API for real-time predictions.  

---

Feel free to contribute or raise issues for improvements.  

**Author**: Arya  
**License**: MIT  

---  
