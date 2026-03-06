import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = {
    "claim_amount":[5000,20000,15000,3000,7000,40000,12000,2500],
    "claim_frequency":[1,3,2,1,2,5,2,1],
    "customer_age":[25,45,35,22,40,50,30,28],
    "fraud":[0,1,0,0,0,1,0,0]
}

df = pd.DataFrame(data)

X = df.drop("fraud",axis=1)
y = df["fraud"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestClassifier()

model.fit(X_train,y_train)

joblib.dump(model,"fraud_model.pkl")

print("Model trained and saved")