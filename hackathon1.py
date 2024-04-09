import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv("data.csv")

# Split the data into training and testing sets
train = data.iloc[:int(0.99*len(data)), :]
test = data.iloc[int(0.99*len(data)):, :]

# Get the input from the user
inp1 = input("Enter the demand for which you want to increase: ")
inp = inp1.lower()

# Mapping of input options to their respective features
input_features = {
    "laptop": "Laptop",
    "smartphone": "Smartphone",
    "headphones": "Headphones",
    "tablet": "Tablet",
    "smartwatch": "Smartwatch",
    "bluetooth speaker": "Bluetooth Speaker",
    "charger": "Charger",
    "external hard drive": "External Hard Drive",
    "wireless mouse": "Wireless Mouse",
    "keyboard": "Keyboard",
    "printer": "Printer",
    "monitor": "Monitor"
}

# Check if the input is valid and get the corresponding feature
if inp in input_features:
    features = input_features[inp]
    target = "prediction"
    model = xgb.XGBRegressor()
    model.fit(train[features], train[target])
    predictions = model.predict(test[features])

    # Plot the graph
    plt.plot(np.arange(len(train[features])), train[features], label='Previous Year')
    plt.plot(len(train[features]) + np.arange(len(test[features])), test[features],)

    # Plot next week prediction as a continuous red curve
    next_week = len(train[features]) + len(test[features]) - 1
    plt.plot([next_week, next_week + 1], [test[features].iloc[-1], predictions[-1]], 'r--o', label='Next Week Prediction (Red Curve)')

    plt.legend()
    plt.title(f"Demand Prediction for {inp}")
    plt.xlabel("Week")
    plt.ylabel("Demand")
    plt.show()


    # Table
    table = pd.DataFrame({'Prediction': predictions})
    print(table.to_string(index=False))

else:
    print("Invalid input. Please enter a valid demand option.")






