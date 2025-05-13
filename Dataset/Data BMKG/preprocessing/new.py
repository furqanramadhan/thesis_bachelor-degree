import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from datetime import datetime, timedelta

# Create an output directory if it doesn't exist
output_dir = 'rice_planting_dss_output'
os.makedirs(output_dir, exist_ok=True)

def save_plt(filename):
    """Save the current plot to the output directory."""
    plt.savefig(f'{output_dir}/{filename}.png', bbox_inches='tight', dpi=300)
    plt.close()

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Load preprocessed BMKG data
def load_data(file_path):
    """Load and prepare BMKG data."""
    try:
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print(f"Successfully loaded data with shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

# Function to handle missing values
def preprocess_data(data):
    """Clean and preprocess the data."""
    # List columns that need to be fixed
    cols_to_fix = ['TN', 'TX', 'TAVG', 'RH_AVG', 'RR', 'SS', 'FF_X', 'DDD_X', 'FF_AVG']
    
    # Clean: replace '-' with NaN, then convert to numeric
    for col in cols_to_fix:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Handle 8888 and 9999 values (missing value codes)
    for col in data.columns:
        if data[col].dtype != 'object':  # Only change numeric columns
            data[col] = data[col].replace([8888, 9999], np.nan)
    
    # Check missing values percentage
    missing_percentage = data.isna().mean() * 100
    print("Missing Values Percentage by Column:")
    for col, pct in missing_percentage.items():
        print(f"{col}: {pct:.2f}%")
    
    # Fill missing values for forecasting
    data_filled = data.fillna(method='ffill').fillna(method='bfill')
    
    return data, data_filled

# Function to perform Holt-Winters forecasting
def forecast_variable(data, variable, period, forecast_days=120):
    """
    Apply Holt-Winters forecasting to a specific variable.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input data with the variable to forecast
    variable : str
        The column name to forecast
    period : int
        Seasonal period (7=weekly, 30=monthly, 365=yearly)
    forecast_days : int
        Number of days to forecast
    
    Returns:
    --------
    tuple
        (model, forecast values, fitted values)
    """
    try:
        print(f"\nForecasting {variable} with seasonal period: {period}")
        
        # Check if we have enough data points for the given period
        if len(data) < 2 * period:
            print(f"Warning: Data length ({len(data)}) is less than twice the seasonal period ({period})")
        
        # Create and fit the model
        model = ExponentialSmoothing(
            data[variable],
            trend='add',
            seasonal='mul',
            seasonal_periods=period
        ).fit(optimized=True)
        
        # Generate forecast
        forecast = model.forecast(forecast_days)
        
        # Get fitted values
        fitted = model.fittedvalues
        
        # Calculate metrics
        mse = mean_squared_error(data[variable].iloc[period:], fitted.iloc[period:])
        rmse = np.sqrt(mse)
        
        print(f"Model parameters:")
        print(f"  Alpha (level): {model.params['smoothing_level']:.4f}")
        print(f"  Beta (trend): {model.params['smoothing_trend']:.4f}")
        print(f"  Gamma (seasonal): {model.params['smoothing_seasonal']:.4f}")
        print(f"Model performance:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        
        return model, forecast, fitted
        
    except Exception as e:
        print(f"Error forecasting {variable}: {str(e)}")
        return None, None, None

# Classification functions for the three main variables
def classify_rr(value):
    """Classify rainfall values."""
    if value < 2:
        return 'Kering (Risiko)'
    elif 2 <= value <= 15:
        return 'Optimal'
    else:
        return 'Banjir (Risiko)'

def classify_tavg(value):
    """Classify average temperature values."""
    if value < 20:
        return 'Dingin (Risiko)'
    elif 20 <= value <= 35:
        return 'Optimal'
    else:
        return 'Panas (Risiko)'

def classify_rh(value):
    """Classify relative humidity values."""
    if value < 60:
        return 'Kering (Risiko)'
    elif 60 <= value <= 90:
        return 'Optimal'
    else:
        return 'Lembab Ekstrem (Risiko)'

# Calculate decision score and recommendation
def calculate_decision(rr_value, tavg_value, rh_value, ss_value=None):
    """
    Calculate decision score and recommendation based on weather parameters.
    
    Parameters:
    -----------
    rr_value : float
        Rainfall value in mm
    tavg_value : float
        Average temperature in °C
    rh_value : float
        Relative humidity in %
    ss_value : float, optional
        Sunshine duration in hours
    
    Returns:
    --------
    tuple
        (score, category, decision)
    """
    # Get classifications
    rr_status = classify_rr(rr_value)
    tavg_status = classify_tavg(tavg_value)
    rh_status = classify_rh(rh_value)
    
    # Initialize score
    score = 0
    
    # Apply weighted scoring: RR (40%), TAVG (40%), RH_AVG (20%)
    if 'Optimal' in rr_status:
        score += 40
    elif 'Kering' in rr_status and rr_value >= 1:  # Some rain is better than none
        score += 20
    
    if 'Optimal' in tavg_status:
        score += 40
    elif tavg_value >= 18 and tavg_value < 20:  # Close to optimal
        score += 30
    elif tavg_value > 35 and tavg_value <= 37:  # Close to optimal
        score += 30
    
    if 'Optimal' in rh_status:
        score += 20
    elif rh_value > 90 and rh_value <= 95:  # Slightly high but manageable
        score += 10
    
    # Build category description
    category_parts = []
    if 'Optimal' not in rr_status:
        category_parts.append(rr_status)
    if 'Optimal' not in tavg_status:
        category_parts.append(tavg_status)
    if 'Optimal' not in rh_status:
        category_parts.append(rh_status)
    
    if category_parts:
        category = ', '.join(category_parts)
    else:
        category = 'Optimal'
    
    # Determine decision
    if 'Banjir' in rr_status or 'Panas' in tavg_status or ('Lembab Ekstrem' in rh_status and rr_value > 10):
        decision = 'Bera'  # Don't plant due to high risk
    elif score >= 70:
        decision = 'Tanam'  # Optimal conditions for planting
    elif 50 <= score < 70:
        decision = 'Tanam (Waspada)'  # Plant with caution
    else:
        decision = 'Bera'  # Don't plant due to suboptimal conditions
    
    return score, category, decision

# Function to check harvesting conditions
def check_harvest_conditions(forecast_df, planting_date):
    """
    Check conditions for harvesting based on planting date.
    
    Parameters:
    -----------
    forecast_df : pandas.DataFrame
        DataFrame with forecasted values
    planting_date : str
        Planting date in YYYY-MM-DD format
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with harvesting recommendations
    """
    # Convert planting date to datetime
    plant_date = pd.to_datetime(planting_date)
    
    # Define harvest window (90-100 days after planting)
    harvest_start = plant_date + pd.Timedelta(days=90)
    harvest_end = plant_date + pd.Timedelta(days=100)
    
    # Get forecast for harvest window plus a few days before and after
    buffer_days = 7
    window_start = harvest_start - pd.Timedelta(days=buffer_days)
    window_end = harvest_end + pd.Timedelta(days=buffer_days)
    
    # Extract relevant forecast period
    harvest_forecast = forecast_df[(forecast_df.index >= window_start) & 
                                  (forecast_df.index <= window_end)].copy()
    
    # Check for risky conditions during harvest
    harvest_forecast['Rainfall_Risk'] = harvest_forecast['RR'] > 10
    harvest_forecast['Humidity_Risk'] = harvest_forecast['RH_AVG'] > 90
    
    # Add recommendation column
    def get_harvest_recommendation(row):
        if row['Rainfall_Risk']:
            if row.name >= harvest_start and row.name <= harvest_end:
                return 'Percepat Panen (Risiko Hujan)'
            else:
                return 'Pantau Cuaca'
        elif row['Humidity_Risk']:
            if row.name >= harvest_start and row.name <= harvest_end:
                return 'Pertimbangkan Panen (Kelembaban Tinggi)'
            else:
                return 'Pantau Kelembaban'
        elif row.name >= harvest_start and row.name <= harvest_end:
            return 'Panen Sesuai Jadwal'
        else:
            return 'Belum Waktunya Panen'
    
    harvest_forecast['Rekomendasi'] = harvest_forecast.apply(get_harvest_recommendation, axis=1)
    
    return harvest_forecast

# Visualize forecasting results
def visualize_forecasts(original_data, forecast_df, variable, period):
    """Create visualization for forecasting results."""
    plt.figure(figsize=(12, 6))
    
    # Plot historical data (last 365 days)
    historical_data = original_data[variable].iloc[-365:]
    plt.plot(historical_data.index, historical_data, 
             label='Data Historis', color='gray', alpha=0.7)
    
    # Plot forecast
    plt.plot(forecast_df.index, forecast_df[variable], 
             label=f'Forecast (period={period})', 
             color='blue', linewidth=2)
    
    # Add confidence intervals
    plt.fill_between(forecast_df.index, 
                     forecast_df[f'{variable}_Lower'], 
                     forecast_df[f'{variable}_Upper'], 
                     color='blue', alpha=0.2)
    
    plt.title(f'Forecast {variable} dengan Seasonal Period {period} Hari')
    plt.xlabel('Tanggal')
    plt.ylabel(variable)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_plt(f'forecast_{variable}_period_{period}')

# Visualize decision support results
def visualize_decisions(decision_df):
    """Create visualization for the decision support results."""
    # Plot decisions timeline
    plt.figure(figsize=(14, 8))
    
    # Convert decision to numeric for plotting
    decision_map = {'Bera': 0, 'Tanam (Waspada)': 1, 'Tanam': 2}
    decision_df['Decision_Value'] = decision_df['Keputusan'].map(
        lambda x: decision_map.get(x, 0) if 'Panen' not in x else 3
    )
    
    # Plot score
    plt.subplot(2, 1, 1)
    plt.plot(decision_df.index, decision_df['Skor'], 
             marker='o', linestyle='-', markersize=4)
    plt.fill_between(decision_df.index, 0, decision_df['Skor'], alpha=0.3)
    plt.axhline(y=70, color='green', linestyle='--', alpha=0.7, label='Batas Optimal (70%)')
    plt.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Batas Waspada (50%)')
    plt.title('Skor Keputusan untuk Penanaman Padi')
    plt.ylabel('Skor (%)')
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot decisions
    plt.subplot(2, 1, 2)
    colors = ['red', 'orange', 'green', 'blue']
    decision_colors = [colors[val] for val in decision_df['Decision_Value']]
    
    plt.scatter(decision_df.index, decision_df['Decision_Value'], 
                c=decision_colors, s=50)
    plt.yticks([0, 1, 2, 3], ['Bera', 'Tanam (Waspada)', 'Tanam', 'Panen'])
    plt.title('Rekomendasi Penanaman Padi')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plt('rice_planting_decisions_timeline')
    
    # Plot weather parameters distribution by decision
    plt.figure(figsize=(16, 12))
    
    # Rainfall by decision
    plt.subplot(2, 2, 1)
    sns.boxplot(x='Keputusan', y='RR', data=decision_df)
    plt.title('Curah Hujan Berdasarkan Keputusan')
    plt.axhline(y=2, color='orange', linestyle='--', label='Batas Kering (2mm)')
    plt.axhline(y=15, color='red', linestyle='--', label='Batas Banjir (15mm)')
    plt.legend()
    
    # Temperature by decision
    plt.subplot(2, 2, 2)
    sns.boxplot(x='Keputusan', y='TAVG', data=decision_df)
    plt.title('Suhu Rata-rata Berdasarkan Keputusan')
    plt.axhline(y=20, color='blue', linestyle='--', label='Batas Dingin (20°C)')
    plt.axhline(y=35, color='red', linestyle='--', label='Batas Panas (35°C)')
    plt.legend()
    
    # Humidity by decision
    plt.subplot(2, 2, 3)
    sns.boxplot(x='Keputusan', y='RH_AVG', data=decision_df)
    plt.title('Kelembaban Relatif Berdasarkan Keputusan')
    plt.axhline(y=60, color='orange', linestyle='--', label='Batas Kering (60%)')
    plt.axhline(y=90, color='blue', linestyle='--', label='Batas Lembab (90%)')
    plt.legend()
    
    # Score by decision
    plt.subplot(2, 2, 4)
    sns.boxplot(x='Keputusan', y='Skor', data=decision_df)
    plt.title('Distribusi Skor Berdasarkan Keputusan')
    
    plt.tight_layout()
    save_plt('decision_parameters_distribution')

def main():
    """Main function to run the Rice Planting Decision Support System."""
    print("=" * 80)
    print("SISTEM PENDUKUNG KEPUTUSAN PENANAMAN PADI BERBASIS FORECASTING HOLT-WINTERS")
    print("=" * 80)
    
    # 1. Load and preprocess data
    print("\nSTEP 1: Loading and preprocessing data...")
    data_path = input("Enter the path to the BMKG data CSV file: ")
    data = load_data(data_path)
    
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    raw_data, filled_data = preprocess_data(data)
    
    # 2. Perform Holt-Winters forecasting for each selected variable
    print("\nSTEP 2: Forecasting weather variables...")
    
    # Variables and their seasonal periods
    forecast_variables = {
        'RR': 365,      # Rainfall (yearly seasonality)
        'TAVG': 30,     # Average temperature (monthly seasonality)
        'RH_AVG': 7     # Relative humidity (weekly seasonality)
    }
    
    # Prepare forecasting
    forecast_days = 120  # 4 months forecast
    forecast_start_date = filled_data.index[-1] + pd.Timedelta(days=1)
    forecast_dates = pd.date_range(start=forecast_start_date, periods=forecast_days, freq='D')
    
    # Initialize forecast dataframe
    forecast_results = pd.DataFrame(index=forecast_dates)
    
    # Run forecasts for each variable
    models = {}
    for var, period in forecast_variables.items():
        if var in filled_data.columns:
            model, forecast, fitted = forecast_variable(filled_data, var, period, forecast_days)
            
            if model is not None:
                # Store in the forecast results
                forecast_results[var] = forecast
                
                # Calculate confidence intervals (95%)
                mse = mean_squared_error(filled_data[var].iloc[period:], fitted.iloc[period:])
                rmse = np.sqrt(mse)
                
                forecast_results[f'{var}_Lower'] = forecast - 1.96 * rmse
                forecast_results[f'{var}_Upper'] = forecast + 1.96 * rmse
                
                # Store model for later use
                models[var] = model
                
                # Visualize forecast
                visualize_forecasts(filled_data, forecast_results, var, period)
        else:
            print(f"Warning: Variable {var} not found in the dataset.")
    
    # 3. Check if sunshine duration (SS) is available
    if 'SS' in filled_data.columns:
        # Use a simpler method for SS forecasting (e.g., monthly seasonality)
        model_ss, forecast_ss, fitted_ss = forecast_variable(filled_data, 'SS', 30, forecast_days)
        if model_ss is not None:
            forecast_results['SS'] = forecast_ss
            
            # Calculate confidence intervals
            mse_ss = mean_squared_error(filled_data['SS'].iloc[30:], fitted_ss.iloc[30:])
            rmse_ss = np.sqrt(mse_ss)
            
            forecast_results['SS_Lower'] = forecast_ss - 1.96 * rmse_ss
            forecast_results['SS_Upper'] = forecast_ss + 1.96 * rmse_ss
            
            # Visualize forecast
            visualize_forecasts(filled_data, forecast_results, 'SS', 30)
    
    # 4. Apply decision support logic
    print("\nSTEP 3: Applying decision support logic...")
    
    # Create a copy of the forecast results for decision making
    decision_df = forecast_results.copy()
    
    # Set negative values to 0 (can't have negative rainfall, etc.)
    for var in forecast_variables.keys():
        decision_df[var] = decision_df[var].clip(lower=0)
    
    # Apply the decision logic for each day in the forecast
    decision_df['Skor'] = 0
    decision_df['Kategori'] = ''
    decision_df['Keputusan'] = ''
    
    for idx, row in decision_df.iterrows():
        try:
            # Check if we have all required variables
            if all(var in row.index for var in forecast_variables.keys()):
                # Calculate decision
                score, category, decision = calculate_decision(
                    row['RR'], row['TAVG'], row['RH_AVG'], 
                    row.get('SS', None)  # SS is optional
                )
                
                # Store results
                decision_df.at[idx, 'Skor'] = score
                decision_df.at[idx, 'Kategori'] = category
                decision_df.at[idx, 'Keputusan'] = decision
        except Exception as e:
            print(f"Error calculating decision for {idx}: {str(e)}")
    
    # 5. Check harvesting conditions
    print("\nSTEP 4: Checking harvesting conditions...")
    planting_date = input("Enter a reference planting date (YYYY-MM-DD) or press Enter to skip: ")
    
    if planting_date:
        try:
            harvest_forecast = check_harvest_conditions(decision_df, planting_date)
            
            # Apply harvesting decisions to main decision dataframe
            for idx, row in harvest_forecast.iterrows():
                if 'Panen' in row['Rekomendasi']:
                    decision_df.at[idx, 'Keputusan'] = row['Rekomendasi']
        except Exception as e:
            print(f"Error checking harvest conditions: {str(e)}")
    
    # 6. Visualize and save results
    print("\nSTEP 5: Visualizing and saving results...")
    visualize_decisions(decision_df)
    
    # 7. Save results to CSV
    output_file = f'{output_dir}/rice_planting_decisions.csv'
    decision_columns = ['RR', 'TAVG', 'RH_AVG', 'Skor', 'Kategori', 'Keputusan']
    if 'SS' in decision_df.columns:
        decision_columns.insert(3, 'SS')
    
    decision_df[decision_columns].to_csv(output_file)
    print(f"\nDecision support results saved to {output_file}")
    
    # Display sample of results
    print("\nSample of Decision Support Results:")
    print(decision_df[decision_columns].head(10))
    
    print("\n" + "=" * 80)
    print("RICE PLANTING DECISION SUPPORT SYSTEM COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()