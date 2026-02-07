"""
Simple Flight Price Predictions
Shows predicted prices from the model
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from data_preprocessing import FlightDataPreprocessor
from model_training import FlightFarePredictor

def main():
    print("="*80)
    print(" "*25 + "FLIGHT PRICE PREDICTIONS")
    print("="*80)
    
    # Load and preprocess training data
    print("\nLoading training data...")
    preprocessor = FlightDataPreprocessor()
    train_df = preprocessor.load_data('data/Data_Train.xlsx')
    
    # Keep a copy of original data
    original_train = train_df.copy()
    
    # Preprocess
    print("Preprocessing data...")
    processed_df = preprocessor.preprocess(train_df)
    
    # Split into train and test
    print("Splitting data...")
    predictor = FlightFarePredictor(random_state=42)
    X_train, X_test, y_train, y_test = predictor.prepare_data(processed_df, test_size=0.2)
    
    # Load or train model
    print("Loading model...")
    try:
        model = predictor.load_model('models/best_flight_fare_model.pkl')
        print("Model loaded from file!")
    except:
        print("Training model (this may take a minute)...")
        predictor.train_all_models(X_train, X_test, y_train, y_test)
        model = predictor.best_model
        print("Model trained!")
    
    # Make predictions
    print("Making predictions...\n")
    predictions = model.predict(X_test)
    
    # Get original data for the test indices
    test_indices = X_test.index
    original_test_data = original_train.iloc[test_indices].reset_index(drop=True)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Flight_No': range(1, len(predictions) + 1),
        'Airline': original_test_data['Airline'],
        'From': original_test_data['Source'],
        'To': original_test_data['Destination'],
        'Stops': original_test_data['Total_Stops'],
        'Duration': original_test_data['Duration'],
        'Actual_Price': y_test.values,
        'Predicted_Price': predictions.round(2)
    })
    
    # Calculate errors
    results['Difference'] = (results['Actual_Price'] - results['Predicted_Price']).round(2)
    results['Error_%'] = (abs(results['Difference']) / results['Actual_Price'] * 100).round(2)
    results['Accuracy_%'] = (100 - results['Error_%']).round(2)
    
    # Display results
    print("="*80)
    print(" "*20 + "FIRST 25 FLIGHT PREDICTIONS")
    print("="*80)
    
    display_df = results[['Flight_No', 'Airline', 'From', 'To', 'Stops', 
                          'Actual_Price', 'Predicted_Price', 'Error_%']].head(25)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 15)
    
    print("\n" + display_df.to_string(index=False))
    
    # Summary statistics
    print("\n" + "="*80)
    print(" "*28 + "OVERALL STATISTICS")
    print("="*80)
    
    print(f"\nTotal Predictions: {len(predictions):,}")
    print(f"\nPRICES:")
    print(f"   Average Actual Price: ₹{results['Actual_Price'].mean():,.2f}")
    print(f"   Average Predicted Price: ₹{results['Predicted_Price'].mean():,.2f}")
    print(f"   Min Predicted: ₹{results['Predicted_Price'].min():,.2f}")
    print(f"   Max Predicted: ₹{results['Predicted_Price'].max():,.2f}")
    
    print(f"\nACCURACY:")
    print(f"   Average Error: ₹{abs(results['Difference']).mean():,.2f}")
    print(f"   Average Error %: {results['Error_%'].mean():.2f}%")
    print(f"   Model Accuracy: {results['Accuracy_%'].mean():.2f}%")
    print(f"   Best Prediction: {results['Error_%'].min():.2f}% error")
    print(f"   Worst Prediction: {results['Error_%'].max():.2f}% error")
    
    # Accuracy breakdown
    excellent = len(results[results['Error_%'] < 5])
    good = len(results[(results['Error_%'] >= 5) & (results['Error_%'] < 10)])
    fair = len(results[(results['Error_%'] >= 10) & (results['Error_%'] < 20)])
    poor = len(results[results['Error_%'] >= 20])
    
    print(f"\nPREDICTION QUALITY:")
    print(f"   Excellent (<5% error): {excellent} flights ({excellent/len(results)*100:.1f}%)")
    print(f"   Good (5-10% error): {good} flights ({good/len(results)*100:.1f}%)")
    print(f"   Fair (10-20% error): {fair} flights ({fair/len(results)*100:.1f}%)")
    print(f"   Poor (>20% error): {poor} flights ({poor/len(results)*100:.1f}%)")
    
    # Top 5 best predictions
    print("\n" + "="*80)
    print(" "*25 + "TOP 5 BEST PREDICTIONS")
    print("="*80)
    
    best_5 = results.nsmallest(5, 'Error_%')
    print("\n" + best_5[['Flight_No', 'Airline', 'From', 'To', 
                          'Actual_Price', 'Predicted_Price', 'Error_%']].to_string(index=False))
    
    # Top 5 worst predictions
    print("\n" + "="*80)
    print(" "*22 + "TOP 5 LARGEST PREDICTION ERRORS")
    print("="*80)
    
    worst_5 = results.nlargest(5, 'Error_%')
    print("\n" + worst_5[['Flight_No', 'Airline', 'From', 'To', 
                           'Actual_Price', 'Predicted_Price', 'Error_%']].to_string(index=False))
    
    # Predictions by airline
    print("\n" + "="*80)
    print(" "*22 + "AVERAGE PREDICTIONS BY AIRLINE")
    print("="*80)
    
    airline_summary = results.groupby('Airline').agg({
        'Predicted_Price': 'mean',
        'Error_%': 'mean',
        'Airline': 'count'
    }).round(2)
    airline_summary.columns = ['Avg_Predicted_Price', 'Avg_Error_%', 'Num_Flights']
    airline_summary = airline_summary.sort_values('Avg_Predicted_Price', ascending=False)
    print("\n" + airline_summary.to_string())
    
    # Predictions by route
    print("\n" + "="*80)
    print(" "*22 + "TOP 10 ROUTES BY PREDICTED PRICE")
    print("="*80)
    
    route_summary = results.groupby(['From', 'To']).agg({
        'Predicted_Price': 'mean',
        'Airline': 'count'
    }).round(2).reset_index()
    route_summary.columns = ['From', 'To', 'Avg_Predicted_Price', 'Num_Flights']
    route_summary = route_summary.sort_values('Avg_Predicted_Price', ascending=False).head(10)
    print("\n" + route_summary.to_string(index=False))
    
    # Detailed examples
    print("\n" + "="*80)
    print(" "*20 + "DETAILED PREDICTION EXAMPLES")
    print("="*80)
    
    for i in range(min(10, len(results))):
        row = results.iloc[i]
        print(f"\n✈️  Flight #{row['Flight_No']}:")
        print(f"   Route: {row['From']} → {row['To']}")
        print(f"   Airline: {row['Airline']}")
        print(f"   Stops: {row['Stops']}")
        print(f"   Duration: {row['Duration']}")
        print(f"   Actual Price: ₹{row['Actual_Price']:,.2f}")
        print(f"   Predicted Price: ₹{row['Predicted_Price']:,.2f}")
        print(f"   Difference: ₹{abs(row['Difference']):,.2f}")
        print(f"   Error: {row['Error_%']:.2f}%")
        
        if row['Error_%'] < 5:
            print(f"   EXCELLENT prediction!")
        elif row['Error_%'] < 10:
            print(f"   GOOD prediction!")
        elif row['Error_%'] < 15:
            print(f"   FAIR prediction!")
        else:
            print(f"   Higher error")
    
    # Save to CSV
    output_file = 'results/all_predictions.csv'
    results.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print(" "*30 + "FILES SAVED")
    print("="*80)
    print(f"\nAll predictions saved to: {output_file}")
    print(f"   ({len(results):,} predictions)")
    print("\nOpen this file in Excel to see all predictions!")
    
    print("\n" + "="*80)
    print(" "*30 + "COMPLETE!")
    print("="*80)
    print()

if __name__ == "__main__":
    main()
