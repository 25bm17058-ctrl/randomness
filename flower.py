import joblib
model=joblib.load("model.pkl")

n=int(input("Enter the number of flowers: "))

for i in range(1,n+1):
    print(f"Enter the details of flower {i}:")
    user_input=[float(input("Enter sepal length: ")), float(input("Enter sepal width: ")), float(input("Enter petal length: ")), float(input("Enter petal width: "))]
    print(f"                              User input for flower {i}: {user_input}")
    prediction=model.predict([user_input])
    print(f"                              Prediction for flower {i}: {prediction}")


