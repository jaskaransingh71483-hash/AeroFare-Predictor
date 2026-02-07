
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import FlightDataPreprocessor
from model_training import FlightFarePredictor
from visualization import FlightDataVisualizer
import pandas as pd


def main():
    """
    Main function to run the complete flight fare prediction pipeline
    """
    
    print("="*80)
    print(" "*20 + "FLIGHT FARE PREDICTION SYSTEM")
    print(" "*25 + "Complete ML Pipeline")
    print("="*80)
    
    # ==================== STEP 1: LOAD DATA ====================
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING")
    print("="*80)
    
    preprocessor = FlightDataPreprocessor()
    train_df = preprocessor.load_data('data/Data_Train.xlsx')
    
    print(f"\nDataset Overview:")
    print(f"  - Total Records: {train_df.shape[0]:,}")
    print(f"  - Total Features: {train_df.shape[1]}")
    print(f"  - Target Variable: Price")
    print(f"  - Price Range: ₹{train_df['Price'].min():,.0f} - ₹{train_df['Price'].max():,.0f}")
    print(f"  - Average Price: ₹{train_df['Price'].mean():,.2f}")
    
    # ==================== STEP 2: EXPLORATORY DATA ANALYSIS ====================
    print("\n" + "="*80)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    visualizer = FlightDataVisualizer()
    visualizer.create_comprehensive_eda(train_df, output_dir='results')
    
    # ==================== STEP 3: DATA PREPROCESSING ====================
    print("\n" + "="*80)
    print("STEP 3: DATA PREPROCESSING & FEATURE ENGINEERING")
    print("="*80)
    
    processed_df = preprocessor.preprocess(train_df.copy())
    
    print("\nPreprocessing Summary:")
    print(f"  - Original Features: {train_df.shape[1]}")
    print(f"  - Processed Features: {processed_df.shape[1]}")
    print(f"  - Records after cleaning: {processed_df.shape[0]:,}")
    
    # ==================== STEP 4: MODEL TRAINING ====================
    print("\n" + "="*80)
    print("STEP 4: MODEL TRAINING & EVALUATION")
    print("="*80)
    
    # Initialize predictor
    predictor = FlightFarePredictor(random_state=42)
    
    # Prepare data
    X_train, X_test, y_train, y_test = predictor.prepare_data(processed_df)
    
    # Train all models
    results = predictor.train_all_models(X_train, X_test, y_train, y_test)
    
    # ==================== STEP 5: RESULTS & COMPARISON ====================
    print("\n" + "="*80)
    print("STEP 5: MODEL COMPARISON & RESULTS")
    print("="*80)
    
    # Get comparison dataframe
    comparison_df = predictor.get_comparison_dataframe()
    print("\nModel Performance Summary:")
    print(comparison_df.to_string(index=False))
    
    # Save comparison to CSV
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    print("\nModel comparison saved to: results/model_comparison.csv")
    
    # Plot model comparison
    predictor.plot_model_comparison(save_path='results/model_comparison.png')
    
    # ==================== STEP 6: BEST MODEL ANALYSIS ====================
    print("\n" + "="*80)
    print("STEP 6: BEST MODEL ANALYSIS")
    print("="*80)
    
    best_model_name = predictor.best_model_name
    best_results = predictor.results[best_model_name]
    
    print(f"\n Best Model: {best_model_name}")
    print(f"\n Performance Metrics:")
    print(f"  - Test R² Score: {best_results['test_r2']:.4f}")
    print(f"  - Test RMSE: ₹{best_results['test_rmse']:,.2f}")
    print(f"  - Test MAE: ₹{best_results['test_mae']:,.2f}")
    print(f"  - Train R² Score: {best_results['train_r2']:.4f}")
    
    # Plot predictions vs actual
    visualizer.plot_predictions_vs_actual(
        y_test, 
        best_results['predictions'],
        model_name=best_model_name,
        save_path='results/predictions_vs_actual.png'
    )
    
    # ==================== STEP 7: SAVE MODEL ====================
    print("\n" + "="*80)
    print("STEP 7: SAVING MODEL")
    print("="*80)
    
    predictor.save_best_model('models/best_flight_fare_model.pkl')
    
    # ==================== FINAL SUMMARY ====================
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    
    print("\n Generated Files:")
    print("  Models:")
    print("    - models/best_flight_fare_model.pkl")
    print("\n  Visualizations:")
    print("    - results/price_distribution.png")
    print("    - results/airline_analysis.png")
    print("    - results/route_analysis.png")
    print("    - results/stops_analysis.png")
    print("    - results/model_comparison.png")
    print("    - results/predictions_vs_actual.png")
    print("\n  Data:")
    print("    - results/model_comparison.csv")
    
    print("\n" + "="*80)
    print(f" Best Model ({best_model_name}) achieves {best_results['test_r2']:.2%} accuracy!")
    print("="*80)
    
    return predictor, processed_df


if __name__ == "__main__":
    predictor, processed_df = main()
