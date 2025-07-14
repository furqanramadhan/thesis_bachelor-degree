import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RainfallHoltWintersForecaster:
    def __init__(self, data_file='preprocessed_rainfall_data.csv'):
        """
        Initialize Holt-Winters Rainfall Forecaster
        
        Parameters:
        -----------
        data_file : str
            Path to preprocessed rainfall data CSV file
        """
        self.data_file = data_file
        self.df = None
        self.ts = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and prepare data for modeling"""
        print("📂 Loading preprocessed rainfall data...")
        
        # Load data
        self.df = pd.read_csv(self.data_file, parse_dates=['Date'], index_col='Date')
        
        # Use RR_log for modeling (better distribution)
        self.ts = self.df['RR_log'].copy()
        
        print(f"✅ Data loaded successfully:")
        print(f"   • Total records: {len(self.df):,}")
        print(f"   • Date range: {self.df.index.min()} to {self.df.index.max()}")
        print(f"   • Missing values: {self.ts.isnull().sum()}")
        
        return self
    
    def time_series_split(self, data, ratio):
        """Split time series data maintaining temporal order"""
        split_idx = int(len(data) * ratio)
        return data.iloc[:split_idx], data.iloc[split_idx:]
    
    def safe_mape(self, actual, forecast):
        """Calculate MAPE avoiding division by zero"""
        # Convert back to original scale
        actual_values = actual.values if hasattr(actual, 'values') else actual
        forecast_values = forecast.values if hasattr(forecast, 'values') else forecast
        
        min_len = min(len(actual_values), len(forecast_values))
        actual_values = actual_values[:min_len]
        forecast_values = forecast_values[:min_len]
        
        # Transform to physical scale (mm)
        actual_orig = np.exp(actual_values) - 1
        forecast_orig = np.exp(forecast_values) - 1
        
        # Handle zero values to avoid division by zero
        non_zero_mask = actual_orig > 0
        if np.sum(non_zero_mask) == 0:
            return 0.0  # All values are zero
        
        # Calculate Absolute Percentage Error
        ape = np.abs((actual_orig[non_zero_mask] - forecast_orig[non_zero_mask]) 
                    / actual_orig[non_zero_mask])
        
        return np.mean(ape) * 100  # Return MAPE in percentage

    def train_holt_winters(self, train_data, seasonal_periods=365):
        """
        Train Holt-Winters model
        
        Parameters:
        -----------
        train_data : pd.Series
            Training time series data
        seasonal_periods : int
            Number of periods in seasonal cycle
        """
        try:
            model = ExponentialSmoothing(
                train_data,
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_periods,
                damped_trend=True,
                use_boxcox=False,  # Already transformed manually
                initialization_method='estimated'
            )
            
            fitted_model = model.fit(
                smoothing_level=None,  # Auto-optimize
                smoothing_trend=None,  # Auto-optimize
                smoothing_seasonal=None,  # Auto-optimize
                damping_trend=None,  # Auto-optimize
                optimized=True,
                remove_bias=True
            )
            
            return fitted_model
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            return None
    
    def evaluate_model(self, model, test_data):
        """Evaluate model performance on test data"""
        if model is None:
            return None
            
       # Generate forecast
        forecast = model.forecast(steps=len(test_data))
        # Ensure forecast length matches test data
        if len(forecast) != len(test_data):
            forecast = forecast[:len(test_data)]  # Trim if longer

        # Validate and adjust forecast length
        if len(forecast) != len(test_data):
            print(f"⚠️  Warning: Forecast length ({len(forecast)}) != test data length ({len(test_data)})")
            min_len = min(len(forecast), len(test_data))
            forecast = forecast[:min_len]
            test_data = test_data.iloc[:min_len] if hasattr(test_data, 'iloc') else test_data[:min_len]
            print(f"   Adjusted to length: {min_len}")
    
        # Calculate metrics in log scale
        mse_log = mean_squared_error(test_data, forecast)
        mae_log = mean_absolute_error(test_data, forecast)
        
        # Convert back to original scale for interpretation
        forecast_orig = np.exp(forecast) - 1
        test_orig = np.exp(test_data) - 1
        # Convert to numpy arrays to avoid pandas alignment issues
        test_orig = test_orig.values if hasattr(test_orig, 'values') else test_orig
        forecast_orig = np.array(forecast_orig) if not isinstance(forecast_orig, np.ndarray) else forecast_orig
        
        # Calculate metrics in original scale
        mse = mean_squared_error(test_orig, forecast_orig)
        mae = mean_absolute_error(test_orig, forecast_orig)
        mape = self.safe_mape(test_data, forecast)
        mad = np.median(np.abs(test_orig - forecast_orig))
        
        # Calculate additional metrics
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((test_orig - forecast_orig)**2) / np.sum((test_orig - np.mean(test_orig))**2))
        
        return {
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape,
            'MAD': mad,
            'RMSE': rmse,
            'R2': r2,
            'MSE_log': mse_log,
            'MAE_log': mae_log,
            'forecast': forecast,
            'forecast_orig': forecast_orig,
            'test_orig': test_orig
        }
    
    def residual_diagnostics(self, model, test_data):
        """Perform residual analysis"""
        if model is None:
            return None
            
        forecast = model.forecast(len(test_data))
        # Ensure same length and convert to compatible types
        if len(forecast) != len(test_data):
            forecast = forecast[:len(test_data)]
        test_values = test_data.values if hasattr(test_data, 'values') else test_data
        residuals = test_values - forecast
        
        # Ljung-Box test for autocorrelation
        try:
            lb_result = acorr_ljungbox(residuals, lags=min(20, len(residuals)//4), return_df=True)
            ljung_box_pvalue = lb_result['lb_pvalue'].iloc[-1]
        except:
            ljung_box_pvalue = np.nan
        
        return {
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'ljung_box_pvalue': ljung_box_pvalue,
            'residuals': residuals
        }
    
    def run_validation(self):
        """Run model validation with multiple train-test splits"""
        print("\n🔄 PHASE 1: Model Validation")
        print("=" * 50)
        
        # Define split ratios
        splits = {
            '70:30': 0.7,
            '80:20': 0.8,
            '90:10': 0.9
        }
        
        self.results = {}
        
        for name, ratio in splits.items():
            print(f"\n📊 Testing {name} split...")
            
            # Split data
            train, test = self.time_series_split(self.ts, ratio)
            
            print(f"   • Training data: {len(train):,} records")
            print(f"   • Test data: {len(test):,} records")
            
            # Train model
            model = self.train_holt_winters(train)
            
            if model is not None:
                # Evaluate model
                metrics = self.evaluate_model(model, test)
                residual_stats = self.residual_diagnostics(model, test)
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'residuals': residual_stats,
                    'train_data': train,
                    'test_data': test
                }
                
                # Print results
                print(f"   ✅ Results:")
                print(f"      MSE: {metrics['MSE']:.2f} mm²")
                print(f"      MAE: {metrics['MAE']:.2f} mm")
                print(f"      MAPE: {metrics['MAPE']:.2f}%")
                print(f"      RMSE: {metrics['RMSE']:.2f} mm")
                print(f"      R²: {metrics['R2']:.3f}")
                
                if residual_stats['ljung_box_pvalue'] > 0.05:
                    print(f"      🟢 Residuals: Independent (p={residual_stats['ljung_box_pvalue']:.3f})")
                else:
                    print(f"      🟡 Residuals: Autocorrelated (p={residual_stats['ljung_box_pvalue']:.3f})")
            else:
                print(f"   ❌ Model training failed")
        
        return self
    
    def select_best_model(self):
        """Select best performing model based on validation results"""
        if not self.results:
            print("❌ No validation results available. Run validation first.")
            return None
        
        print("\n🏆 PHASE 2: Model Selection")
        print("=" * 50)
        
        # Compare models
        comparison = {}
        for name, result in self.results.items():
            if result['metrics'] is not None:
                comparison[name] = {
                    'MAE': result['metrics']['MAE'],
                    'MAPE': result['metrics']['MAPE'],
                    'RMSE': result['metrics']['RMSE'],
                    'R2': result['metrics']['R2']
                }
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison).T
        print("📊 Model Comparison:")
        print(comparison_df.round(3))
        
        # Select best model (lowest MAE as primary criterion)
        best_split = comparison_df['MAE'].idxmin()
        best_model = self.results[best_split]['model']
        
        print(f"\n🎯 Best Model: {best_split} split")
        print(f"   • MAE: {comparison_df.loc[best_split, 'MAE']:.2f} mm")
        print(f"   • MAPE: {comparison_df.loc[best_split, 'MAPE']:.2f}%")
        print(f"   • R²: {comparison_df.loc[best_split, 'R2']:.3f}")
        
        return best_model, best_split
    
    def generate_forecast(self, model, steps=30):
        """Generate forecast for specified number of steps"""
        print(f"\n🔮 PHASE 3: Generating {steps}-day Forecast")
        print("=" * 50)
        
        # Generate forecast
        forecast_log = model.forecast(steps=steps)
        
        # Convert to original scale
        forecast_orig = np.exp(forecast_log) - 1
        
        # Create forecast dates
        last_date = self.df.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'RR_forecast_log': forecast_log,
            'RR_forecast_mm': forecast_orig
        })
        
        # Add confidence intervals (approximation)
        model_residuals = model.resid
        forecast_std = np.std(model_residuals)
        
        forecast_df['RR_forecast_lower'] = np.maximum(0, np.exp(forecast_log - 1.96 * forecast_std) - 1)
        forecast_df['RR_forecast_upper'] = np.exp(forecast_log + 1.96 * forecast_std) - 1
        
        print(f"📅 Forecast Period: {forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[-1].strftime('%Y-%m-%d')}")
        print(f"📊 Forecast Summary:")
        print(f"   • Mean: {forecast_orig.mean():.2f} mm")
        print(f"   • Median: {forecast_orig.median():.2f} mm")
        print(f"   • Min: {forecast_orig.min():.2f} mm")
        print(f"   • Max: {forecast_orig.max():.2f} mm")
        print(f"   • Total: {forecast_orig.sum():.2f} mm")
        
        # Rainfall category analysis
        categories = {
            'No Rain (0mm)': (forecast_orig == 0).sum(),
            'Light Rain (0-20mm)': ((forecast_orig > 0) & (forecast_orig <= 20)).sum(),
            'Moderate Rain (20-50mm)': ((forecast_orig > 20) & (forecast_orig <= 50)).sum(),
            'Heavy Rain (50-100mm)': ((forecast_orig > 50) & (forecast_orig <= 100)).sum(),
            'Very Heavy Rain (>100mm)': (forecast_orig > 100).sum()
        }
        
        print(f"\n📈 Rainfall Distribution:")
        for category, count in categories.items():
            percentage = (count / steps) * 100
            print(f"   • {category}: {count} days ({percentage:.1f}%)")
        
        return forecast_df
    
    def plot_results(self, best_model, best_split, forecast_df):
        """Create comprehensive visualization of results"""
        print("\n📊 PHASE 4: Creating Visualizations")
        print("=" * 50)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('Holt-Winters Rainfall Forecasting Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Training vs Test Performance
        ax1 = axes[0, 0]
        test_data = self.results[best_split]['test_data']
        test_forecast = self.results[best_split]['metrics']['forecast_orig']
        
        # Convert to original scale for plotting
        test_orig = np.exp(test_data) - 1
        
        ax1.plot(test_data.index, test_orig, label='Actual', color='blue', linewidth=2)
        ax1.plot(test_data.index, test_forecast, label='Forecast', color='red', linewidth=2, alpha=0.8)
        ax1.set_title(f'Model Performance - {best_split} Split', fontweight='bold')
        ax1.set_ylabel('Rainfall (mm)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Residuals Analysis
        ax2 = axes[0, 1]
        residuals = self.results[best_split]['residuals']['residuals']
        ax2.scatter(range(len(residuals)), residuals, alpha=0.6, color='green')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax2.set_title('Residuals Analysis', fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Residuals (log scale)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Historical + Forecast
        ax3 = axes[1, 0]
        
        # Last 365 days of historical data
        recent_data = self.df['RR_imputed'].tail(365)
        
        ax3.plot(recent_data.index, recent_data, label='Historical', color='blue', linewidth=2)
        ax3.plot(forecast_df['Date'], forecast_df['RR_forecast_mm'], 
                label='Forecast', color='red', linewidth=2, linestyle='--')
        ax3.fill_between(forecast_df['Date'], 
                        forecast_df['RR_forecast_lower'], 
                        forecast_df['RR_forecast_upper'], 
                        alpha=0.3, color='red', label='95% Confidence')
        
        ax3.set_title('Historical Data + 30-Day Forecast', fontweight='bold')
        ax3.set_ylabel('Rainfall (mm)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Forecast Details
        ax4 = axes[1, 1]
        ax4.bar(forecast_df['Date'], forecast_df['RR_forecast_mm'], 
               color='skyblue', alpha=0.7, edgecolor='navy')
        ax4.set_title('30-Day Rainfall Forecast', fontweight='bold')
        ax4.set_ylabel('Rainfall (mm)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('holt_winters_forecast_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization saved as 'holt_winters_forecast_results.png'")
        
    def run_complete_analysis(self):
        """Run complete Holt-Winters analysis pipeline"""
        print("🚀 STARTING HOLT-WINTERS RAINFALL FORECASTING")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Run validation
        self.run_validation()
        
        # Select best model
        best_model, best_split = self.select_best_model()
        
        if best_model is not None:
            # Generate forecast
            forecast_df = self.generate_forecast(best_model, steps=30)
            
            # Create visualizations
            self.plot_results(best_model, best_split, forecast_df)
            
            # Save forecast results
            forecast_df.to_csv('rainfall_forecast_30days.csv', index=False)
            print(f"\n💾 Forecast saved to 'rainfall_forecast_30days.csv'")
            
            print("\n🎉 ANALYSIS COMPLETE!")
            print("=" * 60)
            
            return forecast_df
        else:
            print("❌ No valid model found. Analysis failed.")
            return None

# Usage Example
if __name__ == "__main__":
    # Create forecaster instance
    forecaster = RainfallHoltWintersForecaster('preprocessed_rainfall_data.csv')
    
    # Run complete analysis
    forecast_results = forecaster.run_complete_analysis()
    
    # Print final summary
    if forecast_results is not None:
        print("\n📋 FINAL FORECAST SUMMARY:")
        print(f"Next 30 days total rainfall: {forecast_results['RR_forecast_mm'].sum():.1f} mm")
        print(f"Average daily rainfall: {forecast_results['RR_forecast_mm'].mean():.1f} mm")
        print(f"Wettest day: {forecast_results.loc[forecast_results['RR_forecast_mm'].idxmax(), 'Date'].strftime('%Y-%m-%d')} ({forecast_results['RR_forecast_mm'].max():.1f} mm)")
        print(f"Driest day: {forecast_results.loc[forecast_results['RR_forecast_mm'].idxmin(), 'Date'].strftime('%Y-%m-%d')} ({forecast_results['RR_forecast_mm'].min():.1f} mm)")