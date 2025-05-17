import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.seasonal import STL
import os
import warnings
warnings.filterwarnings('ignore')

# Plot settings
from pylab import rcParams
rcParams['figure.figsize'] = (12, 5)
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['axes.labelsize'] = 12

# Create output directories if they don't exist
output_dir = './output/plots'
csv_output_dir = './output/csv'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(csv_output_dir, exist_ok=True)

def save_plt(filename):
    """Save the current plot to the output directory."""
    plt.savefig(f'{output_dir}/{filename}.png', bbox_inches='tight', dpi=300)
    plt.close()

# ==============================
# 1. DATA LOADING & PREPROCESSING
# ==============================
print("="*50)
print("LOADING AND PREPROCESSING DATA")
print("="*50)

# Load BMKG data
bmkg_data = pd.read_csv('/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data BMKG/Stasiun Klimatologi Aceh/CSV/BMKG_Data_All.csv', index_col=0, parse_dates=True)
print(f"Data loaded. Shape: {bmkg_data.shape}")
print(f"Date range: {bmkg_data.index.min()} to {bmkg_data.index.max()}")

# Check data types
print("\nData types before conversion:")
print(bmkg_data.dtypes)

# Clean data: convert cols to numeric, replace special values with NaN
cols_to_fix = ['TN', 'TX', 'TAVG', 'RH_AVG', 'RR', 'SS', 'FF_X', 'DDD_X', 'FF_AVG']
for col in cols_to_fix:
    bmkg_data[col] = pd.to_numeric(bmkg_data[col], errors='coerce')

# Handle special missing value codes
for col in bmkg_data.columns:
    if bmkg_data[col].dtype != 'object':  # Only modify numeric columns
        bmkg_data[col] = bmkg_data[col].replace([8888, 9999], np.nan)

print("\nData types after conversion:")
print(bmkg_data.dtypes)

# Check missing values
missing_percentage = bmkg_data.isna().mean() * 100
print("\nMissing values percentage by column:")
print(missing_percentage)

# ==============================
# 2. ADVANCED MISSING VALUE HANDLING WITH STL DECOMPOSITION
# ==============================
print("\n" + "="*50)
print("HANDLING MISSING VALUES WITH STL DECOMPOSITION")
print("="*50)

# Focus on our target variables for rice cultivation
target_vars = ['RR', 'TAVG', 'RH_AVG']
bmkg_for_forecast = bmkg_data[target_vars].copy()

# First apply simple forward/backward fill for initial handling
bmkg_filled = bmkg_for_forecast.fillna(method='ffill').fillna(method='bfill')

def stl_imputation(series, seasonal_period):
    """
    Use STL decomposition to impute missing values with better error handling.
    """
    # Create a copy of the series for manipulation
    imputed_series = series.copy()
    
    # Get indices of missing values
    missing_indices = series[series.isna()].index
    
    # If no missing values, return the original series
    if len(missing_indices) == 0:
        return series
    
    # If too many missing values, use simpler method
    if series.isna().mean() > 0.3:  # If more than 30% missing
        return series.fillna(method='ffill').fillna(method='bfill').fillna(series.mean())
    
    # Fill missing values with a simple method for initial STL
    temp_filled = series.fillna(method='ffill').fillna(method='bfill').fillna(series.mean())
    
    try:
        # Check if we have enough data for the seasonal period
        if len(temp_filled) < 2 * seasonal_period:
            print(f"Not enough data for seasonal_period={seasonal_period}. Using simple imputation.")
            return temp_filled
        
        # Apply STL decomposition with more robust settings
        stl = STL(temp_filled, seasonal=seasonal_period, robust=True, 
                 seasonal_deg=0)  # Lower degree for more stability
        result = stl.fit()
        
        # Extract components
        trend = result.trend
        seasonal = result.seasonal
        residual = result.resid
        
        # Impute missing values using the components
        for idx in missing_indices:
            if idx in trend.index:
                # Reconstruct the value using trend and seasonal component
                imputed_series[idx] = trend[idx] + seasonal[idx]
        
        # For any remaining NaN (e.g., at the edges), use original simple imputation
        imputed_series = imputed_series.fillna(method='ffill').fillna(method='bfill').fillna(series.mean())
        
        return imputed_series
    
    except Exception as e:
        print(f"STL imputation failed: {e}. Using simple imputation.")
        return temp_filled

# Set seasonal periods for our target variables
seasonal_periods = {
    'RR': 365,  # Yearly seasonality for rainfall
    'TAVG': 30,  # Monthly seasonality for temperature
    'RH_AVG': 7   # Weekly seasonality for humidity
}

# Apply STL imputation to each target variable
print("Applying STL imputation for missing values...")
for var in target_vars:
    print(f"Processing {var} with seasonal period {seasonal_periods[var]}...")
    bmkg_filled[var] = stl_imputation(bmkg_for_forecast[var], seasonal_periods[var])

# Check results of imputation
for var in target_vars:
    before_imputation = bmkg_for_forecast[var].isna().sum()
    after_imputation = bmkg_filled[var].isna().sum()
    print(f"{var}: Missing values before: {before_imputation}, after: {after_imputation}")

# ==============================
# 3. GRID SEARCH FOR OPTIMAL PARAMETERS - FIXED VERSION
# ==============================

def hw_grid_search(series, seasonal_periods_list=[7, 30, 365], 
                  trend_types=['add'], seasonal_types=['add', 'mul']):
    """
    Perform grid search with improved error handling
    """
    best_mse = float('inf')
    best_params = {}
    best_model = None
    
    # Split data for training and testing (use last 60 days for testing)
    train_size = len(series) - 60
    if train_size <= 0:
        train_size = int(len(series) * 0.8)
    
    train_data = series.iloc[:train_size]
    test_data = series.iloc[train_size:]
    
    print(f"\nGrid Search for {series.name}")
    
    # Check if we need to force additive seasonality due to zeros or negative values
    force_additive = (series.min() <= 0)
    if force_additive:
        print(f"  Series contains zeros or negative values - forcing additive seasonality")
        seasonal_types = ['add']  # Only use additive seasonality
    
    # Grid search
    for seasonal_period in seasonal_periods_list:
        # Skip if seasonal period is too large for the data
        if seasonal_period >= len(train_data) / 2:
            print(f"  Skipping seasonal_period={seasonal_period} (too large for data)")
            continue
            
        for trend_type in trend_types:
            for seasonal_type in seasonal_types:
                try:
                    # Fit model
                    model = ExponentialSmoothing(
                        train_data,
                        trend=trend_type,
                        seasonal=seasonal_type,
                        seasonal_periods=seasonal_period,
                        use_boxcox=False,
                        initialization_method="estimated"
                    ).fit(optimized=True, remove_bias=True)
                    
                    # Forecast and calculate error
                    forecast = model.forecast(len(test_data))
                    mse = mean_squared_error(test_data, forecast)
                    
                    print(f"  Period={seasonal_period}, Trend={trend_type}, Seasonal={seasonal_type}, MSE={mse:.4f}")
                    
                    # Update best parameters if this is better
                    if mse < best_mse:
                        best_mse = mse
                        best_params = {
                            'seasonal_period': seasonal_period,
                            'trend': trend_type,
                            'seasonal': seasonal_type,
                            'mse': mse
                        }
                        best_model = model
                        
                except Exception as e:
                    print(f"  Error with period={seasonal_period}, trend={trend_type}, seasonal={seasonal_type}: {e}")
    
    if best_model is None:
        print(f"No valid model found. Using safe default parameters.")
        # Always use additive model as fallback
        best_params = {
            'seasonal_period': seasonal_periods[series.name] if series.name in seasonal_periods else 7,
            'trend': 'add',
            'seasonal': 'add',  # Changed to additive as safe default
            'mse': float('inf')
        }
        
    print(f"\nBest parameters for {series.name}:")
    print(f"  Seasonal Period: {best_params['seasonal_period']}")
    print(f"  Trend Type: {best_params['trend']}")
    print(f"  Seasonal Type: {best_params['seasonal']}")
    print(f"  MSE: {best_params['mse']:.4f}")
    
    return {'params': best_params, 'model': best_model}

# Specify potential parameters for grid search
seasonal_periods_options = {
    'RR': [365, 183, 90],    # Annual, semi-annual, quarterly
    'TAVG': [365, 30, 15],   # Annual, monthly, half-monthly
    'RH_AVG': [30, 14, 7]    # Monthly, bi-weekly, weekly
}

# Results container
grid_search_results = {}

# FIXED: Perform grid search only once for each variable with both seasonal options
for var in target_vars:
    print(f"\nPerforming grid search for {var}...")
    grid_search_results[var] = hw_grid_search(
        bmkg_filled[var], 
        seasonal_periods_list=seasonal_periods_options[var],
        trend_types=['add'],  # Additive trend is safer
        seasonal_types=['add', 'mul']  # Consider both options - will be filtered if needed
    )

# Extract optimal parameters
optimal_params = {var: results['params'] for var, results in grid_search_results.items()}

# ==============================
# 4. FORECASTING USING OPTIMAL PARAMETERS - FIXED VERSION
# ==============================

# Forecast horizon (120 days)
forecast_horizon = 120

# Create forecast models using optimal parameters
forecast_models = {}
forecasts = {}

for var in target_vars:
    print(f"\nFitting {var} model with optimal parameters...")
    
    best_period = optimal_params[var]['seasonal_period']
    best_trend = optimal_params[var]['trend']
    best_seasonal = optimal_params[var]['seasonal']
    
    # FIXED: More robust safety check for multiplicative models
    if best_seasonal == 'mul' and bmkg_filled[var].min() <= 0:
        print(f"Forcing additive seasonal for {var} due to zero or negative values")
        best_seasonal = 'add'
    
    try:
        # Create model with optimal parameters
        model = ExponentialSmoothing(
            bmkg_filled[var],
            trend=best_trend,
            seasonal=best_seasonal,
            seasonal_periods=best_period,
            initialization_method="estimated"
        ).fit(optimized=True, remove_bias=True)
        
        # Verify forecasts contain valid values and not all NaN
        test_forecast = model.forecast(5)
        if test_forecast.isnull().all():
            raise ValueError("Model produced all NaN forecasts")
        
        # Store model and generate forecast
        forecast_models[var] = model
        forecasts[var] = model.forecast(forecast_horizon)
        
        # Log success and parameters
        print(f"Successfully fitted model for {var}")
        print(f"Model parameters:")
        print(f"  Alpha (level): {model.params['smoothing_level']:.4f}")
        print(f"  Beta (trend): {model.params.get('smoothing_trend', 'N/A')}")
        print(f"  Gamma (seasonal): {model.params.get('smoothing_seasonal', 'N/A')}")
        
    except Exception as e:
        print(f"Error fitting model for {var}: {e}")
        print("Using simpler model as fallback...")
        
        # Fallback to simpler model if optimal one fails
        try:
            model = ExponentialSmoothing(
                bmkg_filled[var],
                trend='add',
                seasonal='add',  # Always additive as safe fallback
                seasonal_periods=seasonal_periods[var],
                initialization_method="estimated"
            ).fit(optimized=True, remove_bias=True)
            
            forecast_models[var] = model
            forecasts[var] = model.forecast(forecast_horizon)
            print(f"Fallback model successful for {var}")
            
        except Exception as e2:
            print(f"Fallback model also failed: {e2}")
            print("Using naive forecast...")
            
            # FIXED: Improved naive forecast implementation
            # Get last date from data
            # last_date = bmkg_filled.index[-1]
            # forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), 
            #                               periods=forecast_horizon, freq='D')
            
            # Pastikan forecast_index adalah DatetimeIndex
            last_date = bmkg_filled.index[-1]
            forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                          periods=forecast_horizon,freq='D')
            forecast_df = pd.DataFrame(index=forecast_index)  # Inisialisasi dengan indeks yang benar
            
            # FIXED: More robust approach to generate naive forecast
            # Get monthly-day averages with error handling
            monthly_day_means = bmkg_filled[var].groupby([bmkg_filled.index.month, 
                                                        bmkg_filled.index.day]).mean()
            
            # Create naive forecast
            naive_forecast = []
            for date in forecast_index:
                key = (date.month, date.day)
                if key in monthly_day_means.index and not pd.isna(monthly_day_means.loc[key]):
                    naive_forecast.append(monthly_day_means.loc[key])
                else:
                    # If no data for this month-day, use monthly average
                    month_mean = bmkg_filled[var][bmkg_filled.index.month == date.month].mean()
                    if pd.isna(month_mean):
                        naive_forecast.append(bmkg_filled[var].mean())  # Global mean as last resort
                    else:
                        naive_forecast.append(month_mean)
            
            # forecast_models[var] = None  # No actual model
            # forecasts[var] = pd.Series(naive_forecast, index=forecast_index)

            # Ganti kode naive forecast dengan:
            naive_forecast = bmkg_filled[var].ffill().iloc[-forecast_horizon:].values
            forecasts[var] = pd.Series(naive_forecast, index=forecast_index)
            print(f"Naive forecast created for {var}")

# ==============================
# 5. VISUALIZING FORECASTS
# ==============================
print("\n" + "="*50)
print("VISUALIZING FORECASTS")
print("="*50)

# Create forecast index (dates)
last_date = bmkg_filled.index[-1]
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')

# FIXED: Verify forecasts before visualization
for var in target_vars:
    print(f"Forecast for {var}: {len(forecasts[var])} values, {forecasts[var].isna().sum()} NaNs")
    if len(forecasts[var]) > 0:
        print(f"  First 5 values: {forecasts[var].head().tolist()}")

# Visualize forecasts for each variable
for var in target_vars:
    # Skip if forecast is empty or all NaN
    if len(forecasts[var]) == 0 or forecasts[var].isna().all():
        print(f"Skipping visualization for {var} - no valid forecast")
        continue
    
    # Historical data (last 365 days for context)
    plt.figure(figsize=(14, 7))
    recent_data = bmkg_filled[var].iloc[-365:]
    
    # Plot historical data
    plt.plot(recent_data.index, recent_data, label='Historical Data', color='blue', alpha=0.6)
    
    # Plot forecast
    plt.plot(forecast_index, forecasts[var], label='Forecast', color='red', linewidth=2)
    
    # Calculate confidence intervals
    if forecast_models[var] is not None:
        try:
            # FIXED: Handle potential model issues when calculating RMSE
            actual_fitted = recent_data.loc[recent_data.index.intersection(forecast_models[var].fittedvalues.index)]
            model_fitted = forecast_models[var].fittedvalues.loc[recent_data.index.intersection(forecast_models[var].fittedvalues.index)]
            
            if len(actual_fitted) > 0:
                in_sample_rmse = np.sqrt(mean_squared_error(actual_fitted, model_fitted))
                
                # Plot confidence intervals
                plt.fill_between(
                    forecast_index,
                    forecasts[var] - 1.96 * in_sample_rmse,
                    forecasts[var] + 1.96 * in_sample_rmse,
                    color='red', alpha=0.2, label='95% Confidence Interval'
                )
        except Exception as e:
            print(f"Error calculating confidence intervals for {var}: {e}")
    
    # Add threshold reference lines based on agricultural criteria
    if var == 'RR':
        plt.axhline(y=2, color='orange', linestyle='--', alpha=0.7, label='Lower Threshold (2 mm)')
        plt.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='Upper Threshold (15 mm)')
    elif var == 'TAVG':
        plt.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='Lower Threshold (20°C)')
        plt.axhline(y=35, color='red', linestyle='--', alpha=0.7, label='Upper Threshold (35°C)')
    elif var == 'RH_AVG':
        plt.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Lower Threshold (60%)')
        plt.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Upper Threshold (90%)')
    
    plt.grid(True, alpha=0.3)
    plt.title(f'{var} - Forecast for {forecast_horizon} Days')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(loc='best')
    plt.tight_layout()
    save_plt(f'forecast_{var}')

# ==============================
# 6. CLASSIFICATION AND DECISION SYSTEM
# ==============================
print("\n" + "="*50)
print("SETTING UP CLASSIFICATION AND DECISION SYSTEM")
print("="*50)

# Create DataFrame to store forecasts
forecast_df = pd.DataFrame(index=forecast_index)

# FIXED: Verify forecasts are properly assigned to DataFrame
# for var in target_vars:
#     if len(forecasts[var]) > 0:
#         # Make sure indexes align
#         forecast_df[var] = forecasts[var]
#         print(f"Added {var} to forecast_df: {forecast_df[var].notna().sum()} non-null values")
#     else:
#         print(f"Warning: No valid forecast for {var}")
#         forecast_df[var] = np.nan

for var in target_vars:
    if len(forecasts[var]) > 0:
        # Pastikan forecast memiliki indeks yang sama dengan forecast_df
        forecast_df[var] = forecasts[var].reindex(forecast_df.index)
        print(f"Added {var} to forecast_df: {forecast_df[var].notna().sum()} non-null values")
    else:
        print(f"Warning: No valid forecast for {var}")
        forecast_df[var] = np.nan

# Classification functions based on agricultural thresholds
def classify_rr(value):
    """Classify rainfall values into agricultural risk categories."""
    if pd.isna(value):
        return "Tidak Ada Data"
    elif value < 2:
        return "Kering (Risiko)"
    elif 2 <= value <= 15:
        return "Optimal"
    else:
        return "Banjir (Risiko)"

def classify_tavg(value):
    """Classify temperature values into agricultural risk categories."""
    if pd.isna(value):
        return "Tidak Ada Data"
    elif value < 20:
        return "Dingin (Risiko)"
    elif 20 <= value <= 35:
        return "Optimal"
    else:
        return "Panas (Risiko)"

def classify_rh(value):
    """Classify humidity values into agricultural risk categories."""
    if pd.isna(value):
        return "Tidak Ada Data"
    elif value < 60:
        return "Kering (Risiko)"
    elif 60 <= value <= 90:
        return "Optimal"
    else:
        return "Lembab Ekstrem (Risiko)"

# Apply classifications to forecast data
forecast_df['RR_Status'] = forecast_df['RR'].apply(classify_rr)
forecast_df['TAVG_Status'] = forecast_df['TAVG'].apply(classify_tavg)
forecast_df['RH_AVG_Status'] = forecast_df['RH_AVG'].apply(classify_rh)

# Combine status into a single category column
forecast_df['Kategori'] = forecast_df['RR_Status'] + ' / ' + forecast_df['TAVG_Status'] + ' / ' + forecast_df['RH_AVG_Status']

# Calculate weighted score
def calculate_score(row):
    """Calculate weighted score based on variable classifications."""
    score = 0
    
    # Check if we have valid data
    if pd.isna(row['RR']) or pd.isna(row['TAVG']) or pd.isna(row['RH_AVG']):
        return 0
        
    # Assign weights: RR (40%), TAVG (40%), RH_AVG (20%)
    if 'Optimal' in row['RR_Status']:
        score += 40
    if 'Optimal' in row['TAVG_Status']:
        score += 40
    if 'Optimal' in row['RH_AVG_Status']:
        score += 20
        
    return score

# Apply scoring function
forecast_df['Skor'] = forecast_df.apply(calculate_score, axis=1)

# Decision logic
def make_decision(row):
    """Determine planting recommendation based on scores and status."""
    # Handle missing data
    if pd.isna(row['RR']) or pd.isna(row['TAVG']) or pd.isna(row['RH_AVG']):
        return "Bera"
        
    # Prioritize extreme conditions
    if 'Banjir' in row['RR_Status'] or 'Panas' in row['TAVG_Status']:
        return "Bera (Risiko Tinggi)"
        
    # Score-based decisions
    if row['Skor'] >= 70:
        return "Tanam"
    elif 50 <= row['Skor'] < 70:
        return "Tanam (Waspada)"
    else:
        return "Bera"

# Apply decision function
forecast_df['Keputusan'] = forecast_df.apply(make_decision, axis=1)

# ==============================
# 7. HARVEST CALENDAR INTEGRATION
# ==============================
print("\n" + "="*50)
print("INTEGRATING WITH HARVEST CALENDAR")
print("="*50)

# Set planting date for simulation
planting_date = pd.Timestamp('2025-01-11')
print(f"Simulating planting on: {planting_date}")

# Create rice growth window (typically ~100 days from planting to harvest)
growth_duration = 100  # days
harvest_date = planting_date + pd.Timedelta(days=growth_duration)
print(f"Expected harvest date: {harvest_date}")

# Check if harvest date is within our forecast window
if harvest_date in forecast_df.index:
    # Check rainfall conditions 7 days before harvest
    pre_harvest_window = pd.date_range(end=harvest_date, periods=7, freq='D')
    pre_harvest_rain = forecast_df.loc[forecast_df.index.isin(pre_harvest_window), 'RR']
    
    # Determine if there's a rainfall risk (>10 mm/day)
    harvest_risk = (pre_harvest_rain > 10).any()
    
    if harvest_risk:
        harvest_recommendation = "Percepat Panen (Risiko Hujan)"
        print(f"Recommendation: {harvest_recommendation}")
    else:
        harvest_recommendation = "Panen Sesuai Jadwal"
        print(f"Recommendation: {harvest_recommendation}")
        
    # Add harvest recommendation to forecast data
    for date in pre_harvest_window:
        if date in forecast_df.index:
            forecast_df.loc[date, 'Keputusan'] = harvest_recommendation
else:
    print("Harvest date is outside the forecast window.")

# ==============================
# 8. FORECAST EVALUATION AND VALIDATION - FIXED VERSION
# ==============================
print("\n" + "="*50)
print("EVALUATING FORECAST PERFORMANCE")
print("="*50)

# Function to calculate forecast metrics
def calculate_forecast_metrics(actual, forecast):
    """Calculate common forecast accuracy metrics with proper alignment."""
    # First ensure the indices are aligned
    common_index = actual.index.intersection(forecast.index)
    
    # Check if we have any common indices
    if len(common_index) == 0:
        print("Warning: No common indices between actual and forecast data")
        return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'r2': np.nan}
    
    # Filter both series to use only the common indices
    actual = actual.loc[common_index]
    forecast = forecast.loc[common_index]
    
    # Handle NaN values
    valid = ~np.isnan(actual) & ~np.isnan(forecast)
    if sum(valid) == 0:
        return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'r2': np.nan}
    
    actual = actual[valid]
    forecast = forecast[valid]
    
    # Avoid division by zero in MAPE
    mape_valid = actual != 0
    mape = np.mean(np.abs((actual[mape_valid] - forecast[mape_valid]) / actual[mape_valid])) * 100 if sum(mape_valid) > 0 else np.nan
    
    return {
        'mse': mean_squared_error(actual, forecast),
        'rmse': np.sqrt(mean_squared_error(actual, forecast)),
        'mae': mean_absolute_error(actual, forecast),
        'mape': mape,
        'r2': r2_score(actual, forecast)
    }

# Calculate in-sample metrics for each variable
metrics = {}
for var in target_vars:
    # Skip if no model was created for this variable
    if forecast_models[var] is None:
        print(f"\nSkipping metrics for {var} - no model available")
        metrics[var] = {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'r2': np.nan}
        continue
    
    try:
        # # Get fitted values from the model
        # fitted_values = forecast_models[var].fittedvalues
        
        # # Get actual values for the same period
        # actual = bmkg_filled[var].loc[fitted_values.index]

        # fitted_values = model.fittedvalues.reset_index(drop=True)
        # actual = bmkg_filled[var].iloc[:len(fitted_values)].reset_index(drop=True)

        fitted_values = forecast_models[var].fittedvalues  # Sudah memiliki indeks tanggal
        actual = bmkg_filled[var].loc[fitted_values.index]  # Ambil data aktual berdasarkan indeks yang sama

        
        # Calculate metrics
        metrics[var] = calculate_forecast_metrics(actual, fitted_values)
        
        print(f"\nPerformance metrics for {var}:")
        print(f"  MSE: {metrics[var]['mse']:.4f}")
        print(f"  RMSE: {metrics[var]['rmse']:.4f}")
        print(f"  MAE: {metrics[var]['mae']:.4f}")
        print(f"  MAPE: {metrics[var]['mape']:.2f}%")
        print(f"  R-squared: {metrics[var]['r2']:.4f}")
    except Exception as e:
        print(f"\nError calculating metrics for {var}: {e}")
        metrics[var] = {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'r2': np.nan}

# ==============================
# 9. SAVE RESULTS
# ==============================
print("\n" + "="*50)
print("SAVING RESULTS")
print("="*50)

# Prepare final forecast dataframe for output
output_df = forecast_df.reset_index().rename(columns={'index': 'Tanggal'})

# Select and reorder columns for final output
final_columns = ['Tanggal', 'RR', 'TAVG', 'RH_AVG', 'Skor', 'Kategori', 'Keputusan']
output_df = output_df[final_columns]

# Save to CSV
output_path = f"{csv_output_dir}/rice_planting_forecast_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
output_df.to_csv(output_path, index=False)
print(f"Forecast saved to: {output_path}")

# Create summary visualization
plt.figure(figsize=(14, 10))

# Plot 1: Stacked bar for decision counts
plt.subplot(2, 1, 1)
decision_counts = output_df['Keputusan'].value_counts()
decision_counts.plot(kind='bar', color=['green', 'orange', 'red', 'gray'])
plt.title('Distribution of Planting Decisions')
plt.xlabel('Decision')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# Plot 2: Timeline of decisions
plt.subplot(2, 1, 2)
# Create numeric mapping for decisions for colormap
decision_map = {
    'Tanam': 3,
    'Tanam (Waspada)': 2,
    'Bera': 1,
    'Bera (Risiko Tinggi)': 0,
    'Percepat Panen (Risiko Hujan)': 4
}
output_df['Decision_Code'] = output_df['Keputusan'].map(decision_map)

# Plot decision timeline
plt.scatter(output_df['Tanggal'], output_df['Decision_Code'], c=output_df['Decision_Code'], 
            cmap='RdYlGn', alpha=0.8, s=50)
plt.yticks([0, 1, 2, 3, 4], 
           ['Bera (Risiko Tinggi)', 'Bera', 'Tanam (Waspada)', 'Tanam', 'Percepat Panen'])
plt.title('Timeline of Planting Decisions')
plt.xlabel('Date')
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_plt('decision_summary')

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)
print(f"Results saved to {output_path}")
print(f"Plots saved to {output_dir}")
