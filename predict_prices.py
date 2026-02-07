"""
Prediction Script - Show Predicted Flight Prices
Run this to see actual price predictions!
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from data_preprocessing import FlightDataPreprocessor
from model_training import FlightFarePredictor

def main():
    print("="*80)
    print(" "*25 + "FLIGHT FARE PREDICTIONS")
    print("="*80)
    
    # Load preprocessor and trained model
    print("\nLoading trained model...")
    preprocessor = FlightDataPreprocessor()
    predictor = FlightFarePredictor()
    
    # Load the saved model
    model = predictor.load_model('models/best_flight_fare_model.pkl')
    print("Model loaded successfully!")
    
    # Load test data
    print("\nLoading test data...")
    test_df = preprocessor.load_data('data/Test_set.xlsx')
    print(f"Loaded {len(test_df)} test flights")
    
    # Store original data for display
    original_test = test_df.copy()
    
    # Preprocess test data
    print("\nPreprocessing test data...")
    processed_test = preprocessor.preprocess(test_df)
    print("✅ Preprocessing complete!")
    
    # Make predictions
    print("\nMaking predictions...")
    X_test = processed_test.drop('Price', axis=1) if 'Price' in processed_test.columns else processed_test
    
    # Get the feature names from training (stored in model)
    # We need to ensure test data has the same columns as training data
    import joblib
    
    # Load training data to get all features
    print("Aligning features with training data...")
    train_df = preprocessor.load_data('data/Data_Train.xlsx')
    processed_train = preprocessor.preprocess(train_df)
    X_train = processed_train.drop('Price', axis=1)
    
    # Add missing columns to test data with value 0
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    
    # Remove extra columns from test data
    X_test = X_test[X_train.columns]
    
    print("Features aligned!")
    
    predictions = model.predict(X_test)
    print("Predictions complete!")
    
    # Create results dataframe
    results = pd.DataFrame({
        'Airline': original_test['Airline'].values,
        'Source': original_test['Source'].values,
        'Destination': original_test['Destination'].values,
        'Total_Stops': original_test['Total_Stops'].values,
        'Actual_Price': original_test['Price'].values if 'Price' in original_test.columns else ['N/A'] * len(predictions),
        'Predicted_Price': predictions
    })
    
    # Calculate error if actual prices available
    if 'Price' in original_test.columns:
        results['Error'] = results['Actual_Price'] - results['Predicted_Price']
        results['Error_Percentage'] = (abs(results['Error']) / results['Actual_Price'] * 100).round(2)
    
    # Display results
    print("\n" + "="*80)
    print(" "*20 + "PREDICTION RESULTS (First 20 Flights)")
    print("="*80)
    
    # Show first 20 predictions
    display_cols = ['Airline', 'Source', 'Destination', 'Total_Stops', 'Predicted_Price']
    if 'Price' in original_test.columns:
        display_cols.insert(4, 'Actual_Price')
        display_cols.append('Error')
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 15)
    
    print("\n" + results[display_cols].head(20).to_string(index=True))
    
    # Summary statistics
    print("\n" + "="*80)
    print(" "*25 + "PREDICTION SUMMARY")
    print("="*80)
    
    print(f"\n✈️  Total Flights Predicted: {len(predictions)}")
    print(f"Average Predicted Price: ₹{predictions.mean():,.2f}")
    print(f"Price Range: ₹{predictions.min():,.2f} - ₹{predictions.max():,.2f}")
    
    if 'Price' in original_test.columns:
        avg_error = abs(results['Error']).mean()
        avg_error_pct = results['Error_Percentage'].mean()
        print(f"\nAverage Prediction Error: ₹{avg_error:,.2f}")
        print(f"Average Error Percentage: {avg_error_pct:.2f}%")
        print(f"Model Accuracy: {100 - avg_error_pct:.2f}%")
    
    # Save predictions to CSV
    output_file = 'results/flight_predictions.csv'
    results.to_csv(output_file, index=False)
    print(f"\nAll predictions saved to: {output_file}")
    
    # Show sample predictions by route
    print("\n" + "="*80)
    print(" "*20 + "🗺️  SAMPLE PREDICTIONS BY ROUTE")
    print("="*80)
    
    # Group by route and show average predicted price
    route_summary = results.groupby(['Source', 'Destination']).agg({
        'Predicted_Price': 'mean',
        'Airline': 'count'
    }).round(2).reset_index()
    route_summary.columns = ['Source', 'Destination', 'Avg_Predicted_Price', 'Num_Flights']
    route_summary = route_summary.sort_values('Avg_Predicted_Price', ascending=False)
    
    print("\n" + route_summary.head(10).to_string(index=False))
    
    # Interactive mode - predict for specific flights
    print("\n" + "="*80)
    print(" "*15 + "🔍 DETAILED PREDICTIONS (Sample Flights)")
    print("="*80)
    
    for i in range(min(5, len(results))):
        print(f"\n✈️  Flight #{i+1}:")
        print(f"   Route: {results.iloc[i]['Source']} → {results.iloc[i]['Destination']}")
        print(f"   Airline: {results.iloc[i]['Airline']}")
        print(f"   Stops: {results.iloc[i]['Total_Stops']}")
        print(f"   Predicted Price: ₹{results.iloc[i]['Predicted_Price']:,.2f}")
        
        if 'Actual_Price' in results.columns and results.iloc[i]['Actual_Price'] != 'N/A':
            actual = results.iloc[i]['Actual_Price']
            predicted = results.iloc[i]['Predicted_Price']
            error = abs(actual - predicted)
            error_pct = (error / actual * 100)
            
            print(f"   Actual Price: ₹{actual:,.2f}")
            print(f"   Error: ₹{error:,.2f} ({error_pct:.2f}%)")
            
            if error_pct < 5:
                print(f"   Excellent prediction!")
            elif error_pct < 10:
                print(f"   Good prediction!")
            elif error_pct < 15:
                print(f"   Decent prediction!")
            else:
                print(f"   Higher error - consider more features")
    
    print("\n" + "="*80)
    print(" "*25 + "PREDICTIONS COMPLETE!")
    print("="*80)
    print("\nCheck 'results/flight_predictions.csv' for all predictions!")
    print("\n")

if __name__ == "__main__":
    main()
