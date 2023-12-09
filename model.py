import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
import warnings
import pickle

# Load data
data = pd.read_csv('BRAIN_DATA.csv')

# Preprocess data
data.Gender.replace(['M', 'F'], [1, 0], inplace=True)
data.Group.replace(['Demented', 'Nondemented'], [1, 0], inplace=True)
y = np.array(data.Group)
data = data.drop('Group', axis=1)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Classifier
classifier = RandomForestClassifier()
params = {
    'criterion': ['entropy'],
    'n_estimators': [10, 50, 100],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'random_state': [123],
    'n_jobs': [-1]
}

# Grid Search
model1 = GridSearchCV(classifier, param_grid=params, n_jobs=-1)
model1.fit(X_train, y_train)

# Evaluate on the test set
accuracy = model1.score(X_test, y_test)
print(f"Accuracy on the test set: {accuracy:.2f}")

# Save the model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model1, model_file)

# Load the model
loaded_model = pickle.load(open('model.pkl', 'rb'))
