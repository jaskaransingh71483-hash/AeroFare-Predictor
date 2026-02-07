"""
Visualization Module for Flight Fare Prediction
Creates comprehensive EDA visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class FlightDataVisualizer:
    """
    Visualizer for flight fare data
    Creates comprehensive EDA plots
    """
    
    def __init__(self, style='whitegrid'):
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (12, 6)
        
    def plot_price_distribution(self, df, save_path=None):
        """Plot distribution of flight prices"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        axes[0].hist(df['Price'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Price (₹)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Flight Prices')
        axes[0].axvline(df['Price'].mean(), color='red', linestyle='--', 
                        label=f'Mean: ₹{df["Price"].mean():.2f}')
        axes[0].axvline(df['Price'].median(), color='green', linestyle='--', 
                        label=f'Median: ₹{df["Price"].median():.2f}')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Box plot
        axes[1].boxplot(df['Price'], vert=True)
        axes[1].set_ylabel('Price (₹)')
        axes[1].set_title('Box Plot of Flight Prices')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_airline_analysis(self, df, save_path=None):
        """Analyze prices by airline"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Average price by airline
        airline_prices = df.groupby('Airline')['Price'].mean().sort_values(ascending=False)
        axes[0].barh(airline_prices.index, airline_prices.values, color='steelblue', alpha=0.8)
        axes[0].set_xlabel('Average Price (₹)')
        axes[0].set_title('Average Price by Airline')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Flight count by airline
        airline_counts = df['Airline'].value_counts()
        axes[1].bar(airline_counts.index, airline_counts.values, color='coral', alpha=0.8)
        axes[1].set_xlabel('Airline')
        axes[1].set_ylabel('Number of Flights')
        axes[1].set_title('Flight Count by Airline')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_route_analysis(self, df, save_path=None):
        """Analyze prices by source and destination"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Price by Source
        source_prices = df.groupby('Source')['Price'].mean().sort_values(ascending=False)
        axes[0, 0].barh(source_prices.index, source_prices.values, color='green', alpha=0.8)
        axes[0, 0].set_xlabel('Average Price (₹)')
        axes[0, 0].set_title('Average Price by Source City')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # Price by Destination
        dest_prices = df.groupby('Destination')['Price'].mean().sort_values(ascending=False)
        axes[0, 1].barh(dest_prices.index, dest_prices.values, color='orange', alpha=0.8)
        axes[0, 1].set_xlabel('Average Price (₹)')
        axes[0, 1].set_title('Average Price by Destination City')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # Flight count by Source
        source_counts = df['Source'].value_counts()
        axes[1, 0].bar(source_counts.index, source_counts.values, color='purple', alpha=0.8)
        axes[1, 0].set_xlabel('Source City')
        axes[1, 0].set_ylabel('Number of Flights')
        axes[1, 0].set_title('Flight Count by Source')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Flight count by Destination
        dest_counts = df['Destination'].value_counts()
        axes[1, 1].bar(dest_counts.index, dest_counts.values, color='teal', alpha=0.8)
        axes[1, 1].set_xlabel('Destination City')
        axes[1, 1].set_ylabel('Number of Flights')
        axes[1, 1].set_title('Flight Count by Destination')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_stops_analysis(self, df, save_path=None):
        """Analyze impact of stops on price"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Price by number of stops
        stops_price = df.groupby('Total_Stops')['Price'].mean().sort_index()
        axes[0].plot(stops_price.index, stops_price.values, marker='o', 
                     linewidth=2, markersize=10, color='steelblue')
        axes[0].set_xlabel('Number of Stops')
        axes[0].set_ylabel('Average Price (₹)')
        axes[0].set_title('Average Price vs Number of Stops')
        axes[0].grid(alpha=0.3)
        
        # Box plot of price by stops
        df.boxplot(column='Price', by='Total_Stops', ax=axes[1])
        axes[1].set_xlabel('Number of Stops')
        axes[1].set_ylabel('Price (₹)')
        axes[1].set_title('Price Distribution by Number of Stops')
        plt.suptitle('')  # Remove default title
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_heatmap(self, df, save_path=None):
        """Plot correlation heatmap for numerical features"""
        # Select only numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        ax.set_title('Correlation Heatmap of Numerical Features', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_predictions_vs_actual(self, y_true, y_pred, model_name='Model', save_path=None):
        """Plot predicted vs actual prices"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
        axes[0].plot([y_true.min(), y_true.max()], 
                     [y_true.min(), y_true.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Price (₹)')
        axes[0].set_ylabel('Predicted Price (₹)')
        axes[0].set_title(f'{model_name}: Predicted vs Actual Prices')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Price (₹)')
        axes[1].set_ylabel('Residuals (₹)')
        axes[1].set_title(f'{model_name}: Residual Plot')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_comprehensive_eda(self, df, output_dir='../results'):
        """Create all EDA visualizations and save them"""
        print("\n" + "="*60)
        print("CREATING COMPREHENSIVE EDA VISUALIZATIONS")
        print("="*60)
        
        visualizations = []
        
        # Price distribution
        print("\n1. Creating price distribution plot...")
        fig1 = self.plot_price_distribution(df, f'{output_dir}/price_distribution.png')
        visualizations.append(('price_distribution', fig1))
        plt.close()
        
        # Airline analysis
        print("2. Creating airline analysis plot...")
        fig2 = self.plot_airline_analysis(df, f'{output_dir}/airline_analysis.png')
        visualizations.append(('airline_analysis', fig2))
        plt.close()
        
        # Route analysis
        print("3. Creating route analysis plot...")
        fig3 = self.plot_route_analysis(df, f'{output_dir}/route_analysis.png')
        visualizations.append(('route_analysis', fig3))
        plt.close()
        
        # Stops analysis
        print("4. Creating stops analysis plot...")
        fig4 = self.plot_stops_analysis(df, f'{output_dir}/stops_analysis.png')
        visualizations.append(('stops_analysis', fig4))
        plt.close()
        
        print("\n" + "="*60)
        print(f"EDA VISUALIZATIONS COMPLETE - Saved to {output_dir}/")
        print("="*60)
        
        return visualizations


if __name__ == "__main__":
    print("Visualization module loaded successfully!")
