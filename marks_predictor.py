import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
hours = np.array([1,2,3,4,5,6,7,8]).reshape(-1,1)
marks = np.array([30,40,50,60,70,75,85,95])

# Create model
model = LinearRegression()

# Train model
model.fit(hours, marks)

# Predict marks
study_hours = int(input("Enter study hours: "))
predicted_marks = model.predict([[study_hours]])

print("Predicted Marks:", predicted_marks[0])

# Plot graph
plt.scatter(hours, marks, color="blue")
plt.plot(hours, model.predict(hours), color="red")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()