import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prophet import Prophet  # Tambahkan import ini
from prophet.diagnostics import performance_metrics  # Tambahkan import ini
from prophet.diagnostics import cross_validation  # Tambahkan import ini
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
# 1. DATA LOADING (preprocessing is done externally)
# ==============================
print("="*50)
print("LOADING DATA")
print("="*50)

# Load BMKG data (assume it's already cleaned and preprocessed)
bmkg_data = pd.read_csv('/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data BMKG/Stasiun Klimatologi Aceh/CSV CLEANED/BMKG_Data_Cleaned.csv', index_col=0, parse_dates=True)
print(f"Data loaded. Shape: {bmkg_data.shape}")
print(f"Date range: {bmkg_data.index.min()} to {bmkg_data.index.max()}")

# Check data types (expecting all necessary preprocessing already done)
print("\nData types:")
print(bmkg_data.dtypes)

# ==============================
# 2. PREPARE DATA FOR FORECASTING (MODIFIED: Remove Season references)
# ==============================
print("\n" + "="*50)
print("PREPARING DATA FOR FORECASTING")
print("="*50)

target_vars = ['RR', 'TAVG', 'RH_AVG']

# Create a copy to work with
bmkg_for_forecast = bmkg_data[target_vars].copy()

# Tambahkan transformasi log1p untuk RR
bmkg_for_forecast['RR_log'] = np.log1p(bmkg_for_forecast['RR'])

# Add missing rainfall indicator if not already present
if 'is_RR_missing' not in bmkg_data.columns:
    bmkg_for_forecast['is_RR_missing'] = bmkg_for_forecast['RR'].isna().astype(int)
else:
    bmkg_for_forecast['is_RR_missing'] = bmkg_data['is_RR_missing']

# Check if data is already imputed, otherwise use simple imputation for remaining missing values
missing_percentage = bmkg_for_forecast[target_vars].isna().mean() * 100
print("\nMissing values percentage by column:")
print(missing_percentage)

# Simple imputation for any remaining missing values
for var in target_vars:
    if bmkg_for_forecast[var].isna().sum() > 0:
        print(f"Applying simple imputation for remaining missing values in {var}")
        bmkg_for_forecast[var] = bmkg_for_forecast[var].fillna(method='ffill').fillna(method='bfill')

# Create a modified RR variable that incorporates is_RR_missing information
bmkg_for_forecast['RR_adjusted'] = bmkg_for_forecast['RR'].copy()
# When is_RR_missing is 1, multiply RR by a small factor to encode this information
bmkg_for_forecast.loc[bmkg_for_forecast['is_RR_missing'] == 1, 'RR_adjusted'] *= 0.1

# Examine distribution of original vs adjusted rainfall data
print("\nOriginal RR statistics:")
print(bmkg_for_forecast['RR'].describe())
print("\nAdjusted RR statistics:")
print(bmkg_for_forecast['RR_adjusted'].describe())

# ==============================
# 3. GRID SEARCH FOR OPTIMAL PARAMETERS (MODIFIED: Update seasonal periods)
# ==============================
print("\n" + "="*50)
print("GRID SEARCH FOR OPTIMAL PARAMETERS")
print("="*50)

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
        # Use reasonable defaults based on the variable
        default_period = 30 if series.name in ['TAVG', 'RH_AVG'] else 365  # default seasonal periods
        best_params = {
            'seasonal_period': default_period,
            'trend': 'add',
            'seasonal': 'add',
            'mse': float('inf')
        }
        
    print(f"\nBest parameters for {series.name}:")
    print(f"  Seasonal Period: {best_params['seasonal_period']}")
    print(f"  Trend Type: {best_params['trend']}")
    print(f"  Seasonal Type: {best_params['seasonal']}")
    print(f"  MSE: {best_params['mse']:.4f}")
    
    return {'params': best_params, 'model': best_model}

# Specify potential parameters for grid search based on domain knowledge
seasonal_periods_options = {
    'RR': [365, 183, 90],    # Annual, semi-annual, quarterly
    'RR_adjusted': [365, 183, 90],  # Same as RR
    'TAVG': [365, 30, 15],   # Annual, monthly, half-monthly
    'RH_AVG': [30, 14, 7]    # Monthly, bi-weekly, weekly
}

# Results container
grid_search_results = {}

# Perform grid search only once for each variable with both seasonal options
for var in target_vars + ['RR_adjusted']:
    if var == 'RR' and 'RR_adjusted' in bmkg_for_forecast.columns:
        # Skip original RR if we're using the adjusted version
        continue
        
    print(f"\nPerforming grid search for {var}...")
    grid_search_results[var] = hw_grid_search(
        bmkg_for_forecast[var], 
        seasonal_periods_list=seasonal_periods_options.get(var, [365, 30, 7]),
        trend_types=['add'],  # Additive trend is safer
        seasonal_types=['add', 'mul']  # Consider both options - will be filtered if needed
    )

# Extract optimal parameters
optimal_params = {var: results['params'] for var, results in grid_search_results.items()}

# ==============================
# 4. FORECASTING USING OPTIMAL PARAMETERS
# ==============================
print("\n" + "="*50)
print("FORECASTING USING OPTIMAL PARAMETERS")
print("="*50)

# Forecast horizon (120 days)
forecast_horizon = 120

# Create forecast models using optimal parameters
forecast_models = {}
forecasts = {}

last_date = bmkg_for_forecast.index[-1]
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')

rr_df = bmkg_for_forecast[['RR_log']].copy()
rr_df['ds'] = rr_df.index
rr_df.rename(columns={'RR_log': 'y'}, inplace=True)
rr_df = rr_df[['ds', 'y']]

# Build Prophet model
prophet_model = Prophet(
    daily_seasonality=False,  # Usually not needed for rainfall
    weekly_seasonality=True,  # Weekly patterns are often important
    yearly_seasonality=True,
    seasonality_prior_scale=0.1,
    changepoint_prior_scale=0.05
)
prophet_model.fit(rr_df)
# Forecast horizon
future = prophet_model.make_future_dataframe(periods=forecast_horizon)
forecast_rr = prophet_model.predict(future)
# Get only predicted yhat for forecast horizon
forecast_rr_vals = forecast_rr.set_index("ds").loc[forecast_index, "yhat"]
# Inverse log transform
forecast_rr_expm1 = np.expm1(forecast_rr_vals)
# Store in forecasts
forecasts['RR'] = forecast_rr_expm1
forecast_models['RR'] = prophet_model

for var in target_vars + ['RR_adjusted']:
    if var == 'RR' and 'RR_adjusted' in optimal_params:
        # Skip original RR if we're using the adjusted version
        continue
        
    print(f"\nFitting {var} model with optimal parameters...")
    
    if var not in optimal_params:
        print(f"No optimal parameters found for {var}. Skipping.")
        continue
    
    best_period = optimal_params[var]['seasonal_period']
    best_trend = optimal_params[var]['trend']
    best_seasonal = optimal_params[var]['seasonal']
    
    # Safety check for multiplicative models
    if best_seasonal == 'mul' and bmkg_for_forecast[var].min() <= 0:
        print(f"Forcing additive seasonal for {var} due to zero or negative values")
        best_seasonal = 'add'
    
    try:
        # Create model with optimal parameters
        model = ExponentialSmoothing(
            bmkg_for_forecast[var],
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
        forecast_vals = model.forecast(forecast_horizon)
        forecast_vals.index = forecast_index  # <- FIX: Assign datetime index to the forecast
        forecasts[var] = forecast_vals
        
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
            default_period = 30 if var in ['TAVG', 'RH_AVG'] else 365
            model = ExponentialSmoothing(
                bmkg_for_forecast[var],
                trend='add',
                seasonal='add',  # Always additive as safe fallback
                seasonal_periods=default_period,
                initialization_method="estimated"
            ).fit(optimized=True, remove_bias=True)
            
            forecast_models[var] = model
            forecasts[var] = model.forecast(forecast_horizon)
            print(f"Fallback model successful for {var}")
            
        except Exception as e2:
            print(f"Fallback model also failed: {e2}")
            print("Using naive forecast...")
            
            # Improved naive forecast implementation
            last_date = bmkg_for_forecast.index[-1]
            forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                          periods=forecast_horizon, freq='D')
            
            # Create naive forecast based on monthly patterns
            naive_forecast = []
            for date in forecast_index:
                # Get the month for this date
                month = date.month
                
                # Get month average
                month_avg = bmkg_for_forecast[bmkg_for_forecast.index.month == month][var].mean()
                
                if pd.isna(month_avg):
                    # If no month average, use overall average
                    naive_forecast.append(bmkg_for_forecast[var].mean())
                else:
                    naive_forecast.append(month_avg)
            
            forecast_models[var] = None  # No actual model
            forecasts[var] = pd.Series(naive_forecast, index=forecast_index)
            print(f"Monthly-aware naive forecast created for {var}")

# ==============================
# 5. FORECAST FUTURE VALUES (MODIFIED: No more Season-based approach)
# ==============================
print("\n" + "="*50)
print("GENERATING FUTURE DATA")
print("="*50)

# Create forecast index (dates)
last_date = bmkg_for_forecast.index[-1]
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')

# Create DataFrame to store forecasts
forecast_df = pd.DataFrame(index=forecast_index)

for var in target_vars:
    var_to_use = 'RR_adjusted' if var == 'RR' and 'RR_adjusted' in forecasts else var
    
    if var_to_use in forecasts and len(forecasts[var_to_use]) > 0:
        if var == 'RR' and var_to_use == 'RR_adjusted':
            # For rainfall, reverse the adjustment to get original values
            forecast_vals = forecasts[var_to_use].copy()
            # Flag values that appear to come from is_RR_missing=1 (they're much smaller than usual)
            threshold = bmkg_for_forecast.loc[bmkg_for_forecast['is_RR_missing'] == 0, 'RR'].mean() * 0.2
            missing_mask = forecast_vals < threshold
            
            # Convert back to original scale
            forecast_vals[missing_mask] = forecast_vals[missing_mask] / 0.1
            
            # Store both RR and is_RR_missing
            forecast_df[var] = forecast_vals.reindex(forecast_df.index)
            forecast_df['is_RR_missing'] = missing_mask.astype(int)
        else:
            forecast_df[var] = forecasts[var_to_use].reindex(forecast_df.index)
        
        print(f"Added {var} to forecast_df: {forecast_df[var].notna().sum()} non-null values")
    else:
        print(f"Warning: No valid forecast for {var}")
        forecast_df[var] = np.nan

# If is_RR_missing wasn't set yet, initialize it
if 'is_RR_missing' not in forecast_df.columns:
    # For forecasts where RR is very low (< 0.1), mark as potentially missing
    forecast_df['is_RR_missing'] = 0  # Default is not missing
    forecast_df.loc[forecast_df['RR'] < 0.1, 'is_RR_missing'] = 1

# ==============================
# 6. VISUALIZING FORECASTS
# ==============================
print("\n" + "="*50)
print("VISUALIZING FORECASTS")
print("="*50)

for var in target_vars:
    var_to_viz = 'RR_adjusted' if var == 'RR' and 'RR_adjusted' in forecasts else var
    
    # Skip if forecast is empty or all NaN
    if var_to_viz not in forecasts or len(forecasts[var_to_viz]) == 0 or forecasts[var_to_viz].isna().all():
        print(f"Skipping visualization for {var} - no valid forecast")
        continue

    if 'RR' in forecast_models and isinstance(forecast_models['RR'], Prophet):
        fig = forecast_models['RR'].plot_components(forecast_rr)
        fig.savefig(f'{output_dir}/prophet_components_RR.png', bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    # Historical data (last 365 days for context)
    plt.figure(figsize=(14, 7))
    recent_data = bmkg_for_forecast[var].iloc[-365:]
    
    # Plot historical data
    plt.plot(recent_data.index, recent_data, label='Historical Data', color='blue', alpha=0.6)
    
    # Plot forecast
    if var == 'RR' and 'RR_adjusted' in forecasts:
        # Adjust values back for visualization
        plot_forecast = forecasts['RR_adjusted'].copy()
    else:
        plot_forecast = forecasts[var_to_viz]
        
    plt.plot(forecast_index, plot_forecast, label='Forecast', color='red', linewidth=2)
    
    # Calculate confidence intervals
    if var_to_viz in forecast_models and forecast_models[var_to_viz] is not None:
        try:
            # Handle potential model issues when calculating RMSE
            actual_fitted = recent_data.loc[recent_data.index.intersection(forecast_models[var_to_viz].fittedvalues.index)]
            model_fitted = forecast_models[var_to_viz].fittedvalues.loc[recent_data.index.intersection(forecast_models[var_to_viz].fittedvalues.index)]
            
            if len(actual_fitted) > 0:
                in_sample_rmse = np.sqrt(mean_squared_error(actual_fitted, model_fitted))
                
                # Plot confidence intervals
                plt.fill_between(
                    forecast_index,
                    plot_forecast - 1.96 * in_sample_rmse,
                    plot_forecast + 1.96 * in_sample_rmse,
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
    
    # Plot is_RR_missing indicators for RR forecasts
    if var == 'RR':
        # For historical data
        # First, ensure we only use valid points for scatter plot
        missing_mask = (bmkg_for_forecast['is_RR_missing'] == 1) & (~bmkg_for_forecast['RR'].isna())
        missing_dates = bmkg_for_forecast.loc[missing_mask].index
        missing_values = bmkg_for_forecast.loc[missing_mask, 'RR']
        
        # Make sure x and y have the same length before plotting
        if len(missing_dates) > 0 and len(missing_dates) == len(missing_values):
            plt.scatter(missing_dates, missing_values, 
                       color='orange', marker='x', s=50, label='Historical Missing Data Points')
        else:
            print(f"Skipping historical missing data scatter plot: x-size={len(missing_dates)}, y-size={len(missing_values)}")
        
        # For forecast data, also ensure x and y have same length
        forecast_missing_mask = (forecast_df['is_RR_missing'] == 1) & (~forecast_df['RR'].isna())
        forecast_missing_dates = forecast_df.loc[forecast_missing_mask].index
        forecast_missing_values = forecast_df.loc[forecast_missing_mask, 'RR']
        
        if len(forecast_missing_dates) > 0 and len(forecast_missing_dates) == len(forecast_missing_values):
            plt.scatter(forecast_missing_dates, forecast_missing_values, 
                       color='purple', marker='x', s=50, label='Forecast Potential Missing Data')
        else:
            print(f"Skipping forecast missing data scatter plot: x-size={len(forecast_missing_dates)}, y-size={len(forecast_missing_values)}")
    
    plt.grid(True, alpha=0.3)
    plt.title(f'{var} - Forecast for {forecast_horizon} Days')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(loc='best')
    plt.tight_layout()
    save_plt(f'forecast_{var}_with_missing_data')

# ==============================
# 7. ENHANCED CLASSIFICATION AND DECISION SYSTEM (MODIFIED)
# ==============================
print("\n" + "="*50)
print("SETTING UP ENHANCED CLASSIFICATION AND DECISION SYSTEM")
print("="*50)

# Classification functions based on agricultural thresholds
def classify_rr(value, is_missing=0):
    """Classify rainfall values into agricultural risk categories."""
    if pd.isna(value):
        return "Tidak Ada Data"
    elif is_missing == 1:
        return "Data Hujan Tidak Tersedia"
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
forecast_df['RR_Status'] = forecast_df.apply(lambda row: classify_rr(row['RR'], row['is_RR_missing']), axis=1)
forecast_df['TAVG_Status'] = forecast_df['TAVG'].apply(classify_tavg)
forecast_df['RH_AVG_Status'] = forecast_df['RH_AVG'].apply(classify_rh)

# Combine status into a single category column
forecast_df['Kategori'] = forecast_df['RR_Status'] + ' / ' + forecast_df['TAVG_Status'] + ' / ' + forecast_df['RH_AVG_Status']

# Enhanced scoring function incorporating is_RR_missing directly
def calculate_enhanced_score(row):
    """Calculate weighted score based on variable classifications and missing data."""
    score = 0
    
    # Check if we have valid data
    if pd.isna(row['RR']) or pd.isna(row['TAVG']) or pd.isna(row['RH_AVG']):
        return 0
        
    # Base scoring: RR (40%), TAVG (40%), RH_AVG (20%)
    if 'Optimal' in row['RR_Status']:
        score += 40
    if 'Optimal' in row['TAVG_Status']:
        score += 40
    if 'Optimal' in row['RH_AVG_Status']:
        score += 20
    
    # Missing rainfall adjustment (higher impact)
    if row['is_RR_missing'] == 1:
        score -= 25  # Significant penalty for missing rainfall data
        
    # Monthly adjustments for seasonal patterns without referencing 'Season'
    month = row.name.month
    # Wet months (Dec-Mar)
    if month in [12, 1, 2, 3]:
        if 'Banjir' in row['RR_Status']:
            score -= 15  # Higher penalty for flooding in wet months
    # Dry months (Jul-Sep)
    elif month in [7, 8, 9]:
        if 'Kering' in row['RR_Status']:
            score -= 15  # Higher penalty for dry conditions in dry months
        elif 'Optimal' in row['RR_Status']:
            score += 10  # Bonus for optimal rainfall in dry months
            
    return max(0, min(100, score))  # Ensure score is between 0 and 100

# Apply enhanced scoring function
forecast_df['Skor'] = forecast_df.apply(calculate_enhanced_score, axis=1)

# Enhanced decision logic that focuses on is_RR_missing
def make_enhanced_decision(row):
    """Determine planting recommendation based on scores, status, and missing data."""
    # Handle missing data
    if pd.isna(row['RR']) or pd.isna(row['TAVG']) or pd.isna(row['RH_AVG']):
        return "Bera (Data Tidak Lengkap)"
        
    # Prioritize extreme conditions
    if 'Banjir' in row['RR_Status']:
        return "Bera (Risiko Banjir)"
    
    if 'Panas' in row['TAVG_Status']:
        return "Bera (Risiko Suhu Tinggi)"
    
    # Missing rainfall data is critical
    if row['is_RR_missing'] == 1:
        month = row.name.month
        # During typical dry months, missing data is more critical
        if month in [7, 8, 9]:
            return "Bera (Data Hujan Tidak Tersedia di Musim Kering)"
        else:
            return "Tanam Dengan Caution (Data Hujan Tidak Lengkap)"
    
    # Month-specific decisions without referencing 'Season'
    month = row.name.month
    
    # Wet months (Dec-Mar)
    if month in [12, 1, 2, 3]:
        if row['Skor'] >= 75:
            return "Tanam (Kondisi Optimal)"
        elif row['Skor'] >= 60:
            return "Tanam Dengan Caution"
        else:
            return "Bera (Kondisi Tidak Optimal)"
    
    # Dry months (Jul-Sep)
    elif month in [7, 8, 9]:
        if row['Skor'] >= 85:  # Higher threshold for dry months
            return "Tanam (Kondisi Kering Optimal)"
        elif row['Skor'] >= 70:
            return "Tanam Terbatas"
        else:
            return "Bera (Musim Kering Risiko Tinggi)"
    
    # Transition months
    else:  # Apr-Jun or Oct-Nov
        if row['Skor'] >= 80:
            return "Tanam (Periode Transisi Optimal)"
        elif row['Skor'] >= 65:
            return "Tanam Dengan Caution"
        else:
            return "Bera (Periode Transisi Risiko)"

# Apply enhanced decision function
forecast_df['Keputusan'] = forecast_df.apply(make_enhanced_decision, axis=1)

# ==============================
# 8. HARVEST PLANNING WITH MONTHLY AWARENESS
# ==============================
print("\n" + "="*50)
print("INTEGRATING MONTHLY-AWARE HARVEST CALENDAR")
print("="*50)

# Set planting date for simulation
planting_date = pd.Timestamp('2025-01-11')
print(f"Simulating planting on: {planting_date}")

# Create rice growth window with month-aware duration instead of season-aware
def calculate_harvest_date(planting_date, forecast_df):
    """Calculate harvest date considering monthly factors."""
    # Get month of planting
    planting_month = planting_date.month
    
    # Adjust growth duration based on month rather than season
    if planting_month in [12, 1, 2, 3]:  # Wet months
        growth_duration = 105  # Slightly longer due to more cloud cover
    elif planting_month in [7, 8, 9]:  # Dry months
        growth_duration = 95   # Slightly shorter due to more sunlight
    else:  # Transition months
        growth_duration = 100  # Standard duration
    
    harvest_date = planting_date + pd.Timedelta(days=growth_duration)
    
    print(f"Planting in month {planting_month}")
    print(f"Adjusted growth duration: {growth_duration} days")
    print(f"Expected harvest date: {harvest_date}")
    
    return harvest_date, growth_duration

# Calculate month-aware harvest date
harvest_date, growth_duration = calculate_harvest_date(planting_date, forecast_df)

# Check if harvest date is within our forecast window
if harvest_date in forecast_df.index:
    # Check conditions around harvest time
    pre_harvest_window = pd.date_range(end=harvest_date, periods=7, freq='D')
    pre_harvest_data = forecast_df.loc[forecast_df.index.isin(pre_harvest_window)]
    
    # Check for rainfall risk (>10 mm/day)
    rain_risk = (pre_harvest_data['RR'] > 10).any()
    
    # Determine harvest recommendation based on month and rainfall
    harvest_month = harvest_date.month
    
    if rain_risk and harvest_month == 'Wet':  # Wet months
        harvest_recommendation = "Percepat Panen 3-5 Hari (Risiko Hujan Tinggi)"
    elif rain_risk:
        harvest_recommendation = "Percepat Panen 1-2 Hari (Risiko Hujan)"
    elif harvest_month == 'Dry':  # Dry months
        harvest_recommendation = "Panen Sesuai Jadwal (Kondisi Kering Optimal)"
    else:
        harvest_recommendation = "Panen Sesuai Jadwal"
    
    print(f"Harvest recommendation: {harvest_recommendation}")
    
    # Add harvest recommendation to forecast data
    for date in pre_harvest_window:
        if date in forecast_df.index:
            forecast_df.loc[date, 'Keputusan'] = harvest_recommendation
else:
    print("Harvest date is outside the forecast window.")


# ==============================
# 9. FORECAST EVALUATION AND VALIDATION
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
metrics = {}
# Prophet evaluation should be outside the function
if 'RR' in target_vars and 'RR' in forecast_models:
    print(f"\nEvaluating RR with Prophet model...")
    actual_log = bmkg_for_forecast['RR_log']
    prophet_forecast = prophet_model.predict(rr_df)
    
    # Set the ds column as index for proper alignment
    fitted_log = prophet_forecast.set_index('ds')['yhat']
    
    # Align indices
    common_index = actual_log.index.intersection(fitted_log.index)
    actual = np.expm1(actual_log.loc[common_index])
    fitted = np.expm1(fitted_log.loc[common_index])
    
    metrics['RR'] = calculate_forecast_metrics(actual, fitted)

for var in target_vars:
    var_to_use = 'RR_adjusted' if var == 'RR' and 'RR_adjusted' in forecast_models else var

    model = forecast_models.get(var_to_use)
    if model is None:
        print(f"\nSkipping metrics for {var} - no model available")
        metrics[var] = {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'r2': np.nan}
        continue

    try:
        fitted_values = model.fittedvalues
        actual_series = bmkg_for_forecast[var_to_use]

        # Align both to common index
        common_index = actual_series.index.intersection(fitted_values.index)
        actual = actual_series.loc[common_index]
        forecast = fitted_values.loc[common_index]

        # Calculate metrics
        metrics[var] = calculate_forecast_metrics(actual, forecast)

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