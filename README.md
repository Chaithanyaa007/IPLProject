# 🏏 IPL Match Winner Prediction

##  Overview
This project predicts the winner of Indian Premier League (IPL) matches using Machine Learning.

The model is trained on historical IPL match data and predicts match outcomes based on team and toss information.

---

##  Problem Statement
To build a classification model that can predict the winning team of an IPL match using historical match features.

---

## Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Jupyter Notebook

---

##  Dataset
The dataset used is `matches.csv`, containing historical IPL match results and key match details.

---

##  Features Used
- Team 1  
- Team 2  
- Toss Winner  
- Toss Decision

---

##  Model Used
Random Forest Classifier

---

##  Model Accuracy
48.62%

---

##  How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Chaithanyaa007/IPLProject.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter Notebook and run all cells.

---

 Core Model Code
```Python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("matches.csv")

# Select features + target
data = data[['team1','team2','toss_winner','toss_decision','winner']].dropna()

# Encode target
le = LabelEncoder()
y = le.fit_transform(data['winner'])

# One-hot encode inputs
X = pd.get_dummies(data.drop('winner', axis=1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Predict & accuracy
y_pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred)*100,2),"%")
```
---

Future Improvements

- Add venue and match statistics
- Improve accuracy with advanced models (XGBoost / LightGBM)
- Deploy the model using Streamlit for interactive predictions
---

**Author:** Chaithanya S

