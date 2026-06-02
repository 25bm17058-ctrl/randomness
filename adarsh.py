 
import pandas as pd
data=pd.read_csv("Iris.csv")
print(data.head())
#print(data.shape)
x=data.drop(columns=["Id", "Species"])
y=(data[['Species']])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.shape)


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_predict)
print("Accuracy:", accuracy*100)

import joblib
joblib.dump(model, "model.pkl")