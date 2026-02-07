"""
Data Preprocessing Module for Flight Fare Prediction
Handles data cleaning, feature engineering, and transformation
"""

import pandas as pd
import numpy as np
from datetime import datetime


class FlightDataPreprocessor:
    """
    Preprocessor for flight fare data
    Handles missing values, feature extraction, and encoding
    """
    
    def __init__(self):
        self.categorical_columns = []
        self.numerical_columns = []
        
    def load_data(self, filepath):
        """Load data from Excel file"""
        print(f"Loading data from {filepath}...")
        df = pd.read_excel(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("\nHandling missing values...")
        print(f"Missing values before:\n{df.isnull().sum()}")
        
        # Drop rows with missing values (only 2 rows affected)
        df = df.dropna()
        
        print(f"Missing values after:\n{df.isnull().sum()}")
        print(f"Shape after handling missing values: {df.shape}")
        return df
    
    def extract_date_features(self, df):
        """Extract features from Date_of_Journey"""
        print("\nExtracting date features...")
        
        # Convert to datetime
        df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y')
        
        # Extract day, month, year
        df['Journey_day'] = df['Date_of_Journey'].dt.day
        df['Journey_month'] = df['Date_of_Journey'].dt.month
        
        # Drop original date column
        df = df.drop('Date_of_Journey', axis=1)
        
        print("Date features extracted: Journey_day, Journey_month")
        return df
    
    def extract_time_features(self, df):
        """Extract features from Dep_Time and Arrival_Time"""
        print("\nExtracting time features...")
        
        # Departure time features
        df['Dep_hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
        df['Dep_min'] = pd.to_datetime(df['Dep_Time']).dt.minute
        df = df.drop('Dep_Time', axis=1)
        
        # Arrival time features
        df['Arrival_hour'] = pd.to_datetime(df['Arrival_Time']).dt.hour
        df['Arrival_min'] = pd.to_datetime(df['Arrival_Time']).dt.minute
        df = df.drop('Arrival_Time', axis=1)
        
        print("Time features extracted: Dep_hour, Dep_min, Arrival_hour, Arrival_min")
        return df
    
    def extract_duration_features(self, df):
        """Convert duration to total minutes"""
        print("\nExtracting duration features...")
        
        duration_list = []
        for duration in df['Duration']:
            if 'h' not in duration:
                # Only minutes
                duration_list.append(int(duration.split('m')[0].strip()))
            elif 'm' not in duration:
                # Only hours
                duration_list.append(int(duration.split('h')[0].strip()) * 60)
            else:
                # Both hours and minutes
                parts = duration.split()
                hours = int(parts[0].replace('h', '').strip())
                minutes = int(parts[1].replace('m', '').strip())
                duration_list.append(hours * 60 + minutes)
        
        df['Duration_minutes'] = duration_list
        df = df.drop('Duration', axis=1)
        
        print(f"Duration converted to minutes. Mean duration: {df['Duration_minutes'].mean():.2f} min")
        return df
    
    def process_total_stops(self, df):
        """Convert Total_Stops to numerical values"""
        print("\nProcessing Total_Stops...")
        
        # Map stops to numbers
        stops_mapping = {
            'non-stop': 0,
            '1 stop': 1,
            '2 stops': 2,
            '3 stops': 3,
            '4 stops': 4
        }
        
        df['Total_Stops'] = df['Total_Stops'].map(stops_mapping)
        
        print(f"Total_Stops distribution:\n{df['Total_Stops'].value_counts().sort_index()}")
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features using one-hot encoding"""
        print("\nEncoding categorical features...")
        
        # Airline - One-Hot Encoding
        airline_dummies = pd.get_dummies(df['Airline'], prefix='Airline', drop_first=True)
        df = pd.concat([df, airline_dummies], axis=1)
        df = df.drop('Airline', axis=1)
        
        # Source - One-Hot Encoding
        source_dummies = pd.get_dummies(df['Source'], prefix='Source', drop_first=True)
        df = pd.concat([df, source_dummies], axis=1)
        df = df.drop('Source', axis=1)
        
        # Destination - One-Hot Encoding
        destination_dummies = pd.get_dummies(df['Destination'], prefix='Destination', drop_first=True)
        df = pd.concat([df, destination_dummies], axis=1)
        df = df.drop('Destination', axis=1)
        
        # Additional_Info - One-Hot Encoding
        # Group rare categories
        additional_info_counts = df['Additional_Info'].value_counts()
        df['Additional_Info'] = df['Additional_Info'].apply(
            lambda x: x if additional_info_counts[x] > 50 else 'Other'
        )
        
        additional_info_dummies = pd.get_dummies(df['Additional_Info'], prefix='Additional_Info', drop_first=True)
        df = pd.concat([df, additional_info_dummies], axis=1)
        df = df.drop('Additional_Info', axis=1)
        
        # Drop Route as it's complex and redundant with Source/Destination
        df = df.drop('Route', axis=1)
        
        print(f"Categorical encoding complete. Final shape: {df.shape}")
        return df
    
    def preprocess(self, df):
        """Run complete preprocessing pipeline"""
        print("="*60)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Extract features from dates
        df = self.extract_date_features(df)
        
        # Extract features from time
        df = self.extract_time_features(df)
        
        # Extract duration features
        df = self.extract_duration_features(df)
        
        # Process total stops
        df = self.process_total_stops(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        print(f"Final dataset shape: {df.shape}")
        print(f"Features: {df.columns.tolist()[:10]}... (showing first 10)")
        
        return df


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = FlightDataPreprocessor()
    train_df = preprocessor.load_data('../data/Data_Train.xlsx')
    processed_df = preprocessor.preprocess(train_df)
    print(f"\nProcessed data head:\n{processed_df.head()}")
