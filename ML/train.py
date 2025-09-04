import pandas as pd
import joblib 
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/bank_churn_dataset.csv')

data['Gender'] = LabelEncoder().fit_transform(data['Gender'])

x = data.drop(columns=['CustomerId', 'Exited'])
y = data['Exited']

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.2)

model = DecisionTreeClassifier(max_depth=5)
model.fit(x_train, y_train)

joblib.dump(model, 'models/model.joblib')