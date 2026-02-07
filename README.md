# Flight Fare Prediction

## Introduction
The goal of this project is to predict flight fares using machine learning techniques. This can help users get an estimate of the fare before purchasing tickets, potentially saving them money by booking at the right time.

## Dataset
The dataset used for this project contains flight information from various airlines with multiple features that influence the fare. The key input features are:

- **Airline:** Name of the airline
- **Date_of_Journey:** The date on which the journey starts
- **Source:** The starting point of the journey
- **Destination:** The destination point of the journey
- **Route:** The route taken by the flight
- **Dep_Time:** The departure time of the flight
- **Arrival_Time:** The arrival time of the flight
- **Duration:** The total duration of the flight
- **Total_Stops:** The number of stops the flight makes before reaching the destination
- **Additional_Info:** Any additional information about the flight

The output feature is the **Flight Fare**, which is the target variable.

**Dataset Size:** 10,683 flight records  
**Price Range:** ₹1,759 - ₹79,512

## Model Training
The dataset was preprocessed by handling missing values, converting categorical variables using one-hot encoding, and extracting features from dates and times.

## Training Steps
1. Preprocess the dataset
2. Split the data into training and testing sets (80/20 split)
3. Train multiple machine learning models and evaluate their performance

**Models Used:**
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree
- Random Forest
- Gradient Boosting

## Results
The model performs well, providing reliable fare predictions for most flights. 

**Best Model:** Gradient Boosting  
**Accuracy:** 91.07% R² Score  
**Average Error:** ₹783 (approximately 8% error)

This means if the actual price is ₹10,000, the model typically predicts within ±₹783.

Further improvements can be made with more data and hyperparameter tuning.

