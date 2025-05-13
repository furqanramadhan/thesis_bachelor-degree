import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

def load_cleaned_data(location_code, file_path):
    """
    Load cleaned data with proper frequency handling.
    """
    try:
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return None
        
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_index().asfreq('D')  # Force daily frequency
        
        # Forward fill missing values (adjust based on data nature)
        df = df.ffill().bfill()  # Simple handling for demonstration
        
        print(f"Data loaded successfully for {location_code} (Frequency: {df.index.freq})")
        return df
    except Exception as e:
        print(f"Error loading data for {location_code}: {e}")
        return None

def train_hw_model(data: pd.DataFrame, variable: str, seasonal_periods: int = 365, train_ratio: float = 0.8):
    """
    Train Holt-Winters model with improved initialization and error handling.
    """
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if variable not in data.columns:
        raise ValueError(f"Variable '{variable}' not found in data")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    # Clean and prepare data
    data_cleaned = data[[variable]].dropna().copy()
    if len(data_cleaned) < 2:
        raise ValueError("Insufficient data after cleaning")

    # Check seasonal period length
    if len(data_cleaned) < 2 * seasonal_periods:
        print(f"Warning: Data length ({len(data_cleaned)}) is less than 2 complete seasonal cycles ({2*seasonal_periods})")
        seasonal_periods = None

    # Split data
    train_size = int(len(data_cleaned) * train_ratio)
    train_data = data_cleaned.iloc[:train_size]
    test_data = data_cleaned.iloc[train_size:]

    # Configure model with use_boxcox at initialization
    model = ExponentialSmoothing(
        train_data[variable],
        trend='add',
        seasonal='add' if seasonal_periods else None,
        seasonal_periods=seasonal_periods,
        damped_trend=True,
        initialization_method="estimated",
        use_boxcox=False  # Moved here from fit()
    )

    # Fit model with optimized parameters
    try:
        model_fit = model.fit(
            optimized=True,
            remove_bias=True,
            method='L-BFGS-B',
            minimize_kwargs={
                'bounds': [
                    (0, 1),    # smoothing_level
                    (0, 1),    # smoothing_trend
                    (0, 1),    # smoothing_seasonal
                    (0.8, 0.98)  # damping_trend
                ]
            }
        )
    except Exception as e:
        print(f"Optimization failed, using default parameters. Error: {e}")
        # Fallback to default parameters
        model_fit = model.fit(
            smoothing_level=0.2,
            smoothing_trend=0.05,
            smoothing_seasonal=0.15 if seasonal_periods else None,
            damping_trend=0.98,
            optimized=False
        )

    print(f"Model fitted successfully. Training size: {len(train_data)}, Test size: {len(test_data)}")
    return model_fit, train_data, test_data

def evaluate_model(model_fit, test_data, variable):
    """
    Evaluate model with index alignment.
    """
    forecast_horizon = len(test_data)
    
    # Generate index-aligned forecast
    forecast = model_fit.forecast(forecast_horizon)
    forecast = pd.Series(forecast, index=test_data.index)
    
    # Calculate metrics
    mae = mean_absolute_error(test_data[variable], forecast)
    rmse = np.sqrt(mean_squared_error(test_data[variable], forecast))
    mape = np.mean(np.abs((test_data[variable] - forecast) / test_data[variable])) * 100
    
    print(f"Evaluation metrics for {variable}:")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape}, forecast

def plot_forecasts(train_data, test_data, forecast, variable, location_code, output_dir):
    """
    Plot the forecasts against actual data.
    """
    plt.figure(figsize=(15, 7))
    
    # Handle different index types
    if isinstance(train_data.index, pd.DatetimeIndex):
        x_train = train_data.index
        x_test = test_data.index
        x_forecast = forecast.index if hasattr(forecast, 'index') else test_data.index
    else:
        # Create arbitrary dates for plotting
        x_train = np.arange(len(train_data))
        x_test = np.arange(len(train_data), len(train_data) + len(test_data))
        x_forecast = x_test
    
    # Plot data
    plt.plot(x_train, train_data[variable], label='Training Data')
    plt.plot(x_test, test_data[variable], label='Test Data')
    plt.plot(x_forecast, forecast, label='Forecast', color='red')
    
    plt.title(f'{variable} Forecast for {location_code}')
    plt.xlabel('Date')
    
    if variable == 'SST':
        plt.ylabel('Sea Surface Temperature (°C)')
    elif variable == 'SWRad':
        plt.ylabel('Shortwave Radiation (W/m²)')
    elif variable == 'WSPD':
        plt.ylabel('Wind Speed (m/s)')
    
    plt.legend()
    plt.grid(True)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{location_code}_{variable}_forecast.png'))
    plt.close()

def define_thresholds():
    """
    Define thresholds for each variable for rice planting suitability.
    
    Returns:
    --------
    dict
        Thresholds for each variable
    """
    # These thresholds should be adjusted based on research and local conditions
    thresholds = {
        'SST': {
            'min': 27.0,  # Minimum SST in °C for favorable conditions
            'max': 30.0,  # Maximum SST in °C for favorable conditions
            'weight': 0.3  # Relative importance weight
        },
        'SWRad': {
            'min': 150.0,  # Minimum radiation in W/m² for adequate photosynthesis
            'max': None,   # No upper limit for radiation (more is generally better)
            'weight': 0.4  # Relative importance weight
        },
        'WSPD': {
            'min': 1.0,    # Minimum wind speed in m/s (too little means stagnant air)
            'max': 6.0,    # Maximum wind speed in m/s (too much can damage plants)
            'weight': 0.3  # Relative importance weight
        }
    }
    
    return thresholds

def calculate_monthly_suitability(forecasts, thresholds):
    """
    Calculate monthly suitability scores based on forecasted variables.
    
    Parameters:
    -----------
    forecasts : dict
        Dictionary with forecasted values for each variable
    thresholds : dict
        Thresholds for each variable
        
    Returns:
    --------
    DataFrame
        Monthly suitability scores and decisions
    """
    # Create a DataFrame with forecasted values
    forecast_df = pd.DataFrame()
    
    # Add each variable's forecast to the DataFrame
    for var, values in forecasts.items():
        forecast_df[var] = values
    
    # Resample to monthly averages
    monthly_df = forecast_df.resample('MS').mean()
    
    # Initialize suitability scores
    monthly_df['SST_suitable'] = 0.0
    monthly_df['SWRad_suitable'] = 0.0
    monthly_df['WSPD_suitable'] = 0.0
    
    # Calculate suitability scores for each variable
    for var in ['SST', 'SWRad', 'WSPD']:
        if var in monthly_df.columns:
            # Check minimum threshold
            if thresholds[var]['min'] is not None:
                monthly_df[f'{var}_suitable'] = np.where(
                    monthly_df[var] >= thresholds[var]['min'],
                    monthly_df[f'{var}_suitable'] + 0.5,
                    monthly_df[f'{var}_suitable']
                )
            
            # Check maximum threshold
            if thresholds[var]['max'] is not None:
                monthly_df[f'{var}_suitable'] = np.where(
                    monthly_df[var] <= thresholds[var]['max'],
                    monthly_df[f'{var}_suitable'] + 0.5,
                    monthly_df[f'{var}_suitable']
                )
    
    # Calculate weighted overall suitability
    monthly_df['overall_suitability'] = (
        monthly_df['SST_suitable'] * thresholds['SST']['weight'] +
        monthly_df['SWRad_suitable'] * thresholds['SWRad']['weight'] +
        monthly_df['WSPD_suitable'] * thresholds['WSPD']['weight']
    )
    
    # Normalize overall suitability to 0-1 range
    max_possible = (
        1.0 * thresholds['SST']['weight'] +
        1.0 * thresholds['SWRad']['weight'] +
        1.0 * thresholds['WSPD']['weight']
    )
    monthly_df['overall_suitability'] = monthly_df['overall_suitability'] / max_possible
    
    return monthly_df

def make_planting_decisions(monthly_suitability, suitability_threshold=0.7, grow_days=95):
    """
    Make rice planting decisions based on suitability scores.
    
    Parameters:
    -----------
    monthly_suitability : DataFrame
        Monthly suitability scores
    suitability_threshold : float
        Minimum suitability score to recommend planting (0-1)
    grow_days : int
        Number of days required for rice to grow and be harvested
        
    Returns:
    --------
    DataFrame
        Monthly planting decisions
    """
    # Create copy to avoid modifying original
    decision_df = monthly_suitability.copy()
    
    # Initialize decisions
    decision_df['decision'] = 'Tidak Sesuai'  # Default to "Not Suitable"
    
    # Mark suitable months for planting
    decision_df.loc[decision_df['overall_suitability'] >= suitability_threshold, 'decision'] = 'Sesuai'
    
    # Convert to list for easier processing
    months = decision_df.index.tolist()
    decisions = decision_df['decision'].tolist()
    
    # Account for growth period (harvesting after grow_days)
    # Assume approximately 3 months for rice growth
    grow_months = grow_days // 30
    
    final_decisions = []
    for i in range(len(decisions)):
        if decisions[i] == 'Sesuai':
            # Check future months for harvesting suitability
            if i + grow_months < len(decisions):
                if decisions[i + grow_months] == 'Sesuai':
                    final_decisions.append('Tanam')  # Plant
                else:
                    final_decisions.append('Tidak Sesuai')  # Not suitable due to harvest conditions
            else:
                final_decisions.append('Tidak Sesuai')  # Not enough time in forecast for full growing cycle
        else:
            # Check if this is a harvest month for a previous planting
            is_harvest_month = False
            for j in range(1, grow_months + 1):
                if i - j >= 0 and final_decisions[i - j] == 'Tanam':
                    is_harvest_month = True
                    break
            
            if is_harvest_month:
                final_decisions.append('Panen')  # Harvest
            else:
                final_decisions.append('Bera')  # Fallow
    
    # Add final decisions to DataFrame
    decision_df['final_decision'] = final_decisions
    
    return decision_df

def generate_planting_calendar(decision_df, location_code, output_dir):
    """
    Generate planting calendar visualization.
    
    Parameters:
    -----------
    decision_df : DataFrame
        Monthly planting decisions
    location_code : str
        Location identifier
    output_dir : str
        Directory for saving outputs
    """
    # Prepare data for plotting
    months = decision_df.index.strftime('%Y-%m')
    decisions = decision_df['final_decision']
    
    # Define colors for different decisions
    colors = {
        'Tanam': 'green',
        'Panen': 'gold',
        'Bera': 'brown',
        'Tidak Sesuai': 'red'
    }
    
    # Map decisions to colors
    bar_colors = [colors[d] for d in decisions]
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Plot horizontal bars
    y_pos = np.arange(len(months))
    plt.barh(y_pos, 1, color=bar_colors)
    
    # Add labels
    plt.yticks(y_pos, months)
    plt.xlabel('Decision')
    plt.title(f'Rice Planting Calendar for {location_code}')
    
    # Add legend
    for decision, color in colors.items():
        plt.bar(0, 0, color=color, label=decision)
    plt.legend(loc='best')
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{location_code}_planting_calendar.png'))
    plt.close()
    
    # Also save as CSV
    decision_df.to_csv(os.path.join(output_dir, f'{location_code}_planting_calendar.csv'))

def export_monthly_decisions(decision_df, location_code, output_dir):
    """
    Export monthly decisions in a simple format.
    
    Parameters:
    -----------
    decision_df : DataFrame
        Monthly planting decisions
    location_code : str
        Location identifier
    output_dir : str
        Directory for saving outputs
    """
    # Format for export
    export_df = decision_df.copy()
    export_df.index = export_df.index.strftime('%Y-%m')
    
    # Keep only essential columns
    export_df = export_df[['overall_suitability', 'final_decision']]
    
    # Export to CSV
    os.makedirs(output_dir, exist_ok=True)
    export_path = os.path.join(output_dir, f'{location_code}_monthly_decisions.csv')
    export_df.to_csv(export_path)
    
    print(f"Exported monthly decisions to {export_path}")
    
    # Create a simplified text version
    with open(os.path.join(output_dir, f'{location_code}_decisions.txt'), 'w') as f:
        f.write(';'.join([d for d in export_df['final_decision']]))
    
    return export_df

def main():
    """
    Main execution function for rice planting decision support system.
    """
    # Define location codes and directories
    location_codes = ['0N90E', '4N90E', '8N90E']
    data_dirs = {
        '0N90E': '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data Buoys/0N90E/CSV CLEANED/0N90E_combined_clean.csv',
        '4N90E': '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data Buoys/4N90E/CSV CLEANED/4N90E_combined_clean.csv',
        '8N90E': '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data Buoys/8N90E/CSV CLEANED/8N90E_combined_clean.csv'
    }
    output_dir = '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data Buoys/results'

    # Process each location
    for location_code in location_codes:
        print(f"\n{'='*50}")
        print(f"Processing location: {location_code}")
        print(f"{'='*50}")

        file_path = data_dirs.get(location_code)
        if not file_path or not os.path.exists(file_path):
            print(f"Error: File not found for {location_code}")
            continue

        # Load data with enforced daily frequency
        df = load_cleaned_data(location_code, file_path)
        if df is None:
            continue

        variables = ['SST', 'SWRad', 'WSPD']
        seasonality = {'SST': 365, 'SWRad': 365, 'WSPD': 365}
        thresholds = define_thresholds()
        
        forecasts = {}
        test_forecasts = {}  # Store test period forecasts

        for var in variables:
            if var not in df.columns:
                print(f"Skipping {var} - not in dataset")
                continue

            print(f"\nForecasting {var} for {location_code}")
            
            # Train model
            model_fit, train_data, test_data = train_hw_model(
                df, var, seasonal_periods=seasonality[var]
            )

            # Evaluate on test data
            var_metrics, test_forecast = evaluate_model(model_fit, test_data, var)
            
            # Generate future forecast (365 days after training)
            forecast_horizon = 365
            last_train_date = train_data.index[-1]
            future_dates = pd.date_range(
                start=last_train_date + pd.Timedelta(days=1),
                periods=forecast_horizon,
                freq=train_data.index.freq  # Use same frequency
            )
            future_forecast = model_fit.forecast(forecast_horizon)
            future_forecast = pd.Series(future_forecast, index=future_dates)

            # Plot test period forecast and actuals
            plot_forecasts(
                train_data, 
                test_data, 
                test_forecast,  # Plot predictions for test period
                var, 
                location_code, 
                output_dir
            )

            # Store forecasts
            forecasts[var] = future_forecast
            test_forecasts[var] = test_forecast  # For debugging

        # Calculate monthly suitability using future forecasts
        monthly_suitability = calculate_monthly_suitability(forecasts, thresholds)
        
        # Generate planting calendar
        if not monthly_suitability.empty:
            decision_df = make_planting_decisions(monthly_suitability)
            generate_planting_calendar(decision_df, location_code, output_dir)
            export_monthly_decisions(decision_df, location_code, output_dir)

        print(f"\nCompleted processing for {location_code}")

    print("\nAll locations processed successfully!")

if __name__ == "__main__":
    main()