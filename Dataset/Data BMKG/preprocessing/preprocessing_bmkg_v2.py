import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL
from scipy.interpolate import CubicSpline
from datetime import datetime
import math
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Ignore future warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def plot_rr_distribution(df):
        imputed_mask = df['is_RR_missing'] == 1
        known_mask = df['is_RR_missing'] == 0
        plt.figure(figsize=(14, 6))
    # Histogram RR diketahui
        plt.hist(df.loc[known_mask, 'RR'], bins=50, alpha=0.6, label='Original RR (Known)', color='blue', density=True)

    # Histogram RR hasil imputasi
        plt.hist(df.loc[imputed_mask, 'RR'], bins=50, alpha=0.6, label='Imputed RR (Missing)', color='orange', density=True)

        plt.title("Distribusi Curah Hujan (RR): Asli vs Hasil Imputasi", fontsize=14)
        plt.xlabel("RR (mm)", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("RR_Distribution_Before_After_Imputation.png", dpi=300)
        print("📊 Histogram RR saved to 'RR_Distribution_Before_After_Imputation.png'")

def preprocess_bmkg_data(input_file, output_file, location="Aceh"):
    print("Loading data...")
    # Load data
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded data with shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Convert Date to datetime format
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        # Create Date from Year, Month, Day if Date column doesn't exist
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        df.set_index('Date', inplace=True)
    
    # Sort data by date
    df = df.sort_index()
    
    # Define missing value codes
    missing_codes = [8888.0, 9999.0]
    
    # IMPROVEMENT 1: Location-specific season definition
    # Define seasons based on location (customize for different regions in Indonesia)
    season_definitions = {
        "Aceh": {
            # Based on Aceh's specific climate pattern (may need adjustment)
            "wet": [9, 10, 11, 12, 1, 2, 3],  # Sep-Mar
            "dry": [4, 5, 6, 7, 8]            # Apr-Aug
        },
    }
    
    # Use appropriate season definition
    if location in season_definitions:
        wet_months = season_definitions[location]["wet"]
        df['Season'] = df.index.month.map(lambda m: 'Wet' if m in wet_months else 'Dry')
        print(f"Using {location}-specific season definition")
    else:
        # Default definition
        df['Season'] = df.index.month.map(lambda m: 'Wet' if m >= 10 or m <= 4 else 'Dry')
        print("Using default season definition")
    
    # Add month name for easier analysis
    df['Month'] = df.index.month_name()
    
    # Display initial information
    print("\nInitial data info:")
    print(f"Dataset shape: {df.shape}")
    
    # Replace missing value codes with NaN
    for col in df.select_dtypes(include=[np.number]).columns:
        for code in missing_codes:
            mask = df[col] == code
            if mask.sum() > 0:
                print(f"Replaced {mask.sum()} instances of {code} with NaN in column {col}")
                df.loc[mask, col] = np.nan
    
    # IMPROVEMENT 2: Detect and handle outliers before imputation
    print("\nDetecting and handling outliers...")
    
    # Define valid ranges for each variable (based on climate norms)
    valid_ranges = {
        'TX': (-5, 45),    # Max temperature in °C
        'TN': (-10, 35),   # Min temperature in °C
        'TAVG': (-5, 40),  # Avg temperature in °C
        'RH_AVG': (0, 100), # Relative humidity in %
        'RR': (0, 500),    # Rainfall in mm (extreme events can exceed 300mm)
        'FF_X': (0, 50),   # Max wind speed in m/s
        'FF_AVG': (0, 30), # Avg wind speed in m/s
        'SS': (0, 14)      # Sunshine duration in hours
    }
    
    # Detect and handle outliers
    for col, (min_val, max_val) in valid_ranges.items():
        if col in df.columns:
            # Flag values outside valid range
            invalid_mask = (df[col] < min_val) | (df[col] > max_val)
            if invalid_mask.sum() > 0:
                print(f"Found {invalid_mask.sum()} outliers in {col}")
                # Set outliers to NaN for later imputation
                df.loc[invalid_mask, col] = np.nan
    
    # Display missing values before preprocessing
    missing_before = df.isnull().sum() / len(df) * 100
    print("\nMissing values percentage before preprocessing:")
    print(missing_before)
    
    # ----- IMPROVED RAINFALL PREPROCESSING -----
    print("\n🔧 [RR] Processing Rainfall (RR) using Probabilistic Approach...")
    
    # Create flag for missing rainfall (preserve this from original code)
    df['is_RR_missing'] = df['RR'].isna().astype(int)
    
    # Pre-process key predictors to avoid NaNs in the model
    # We need to do this before building the rainfall model
    # Temperature (TX, TN, TAVG) & Humidity (RH_AVG) preprocessing
    print("Pre-processing temperature and humidity for rainfall prediction...")
    
    for col in ['TX', 'TN', 'TAVG']:
        if col in df.columns and df[col].isna().any():
            # Try to maintain the relationship between daily min, max, and average temps
            if col == 'TAVG' and not df['TX'].isna().all() and not df['TN'].isna().all():
                # Calculate TAVG from TX and TN where both are available
                calc_mask = df['TAVG'].isna() & df['TX'].notna() & df['TN'].notna()
                df.loc[calc_mask, 'TAVG'] = (df.loc[calc_mask, 'TX'] + df.loc[calc_mask, 'TN']) / 2
            
            elif col == 'TX' and not df['TAVG'].isna().all() and not df['TN'].isna().all():
                # Use relationship TX ≈ TAVG + (TAVG - TN)
                calc_mask = df['TX'].isna() & df['TAVG'].notna() & df['TN'].notna()
                df.loc[calc_mask, 'TX'] = 2 * df.loc[calc_mask, 'TAVG'] - df.loc[calc_mask, 'TN']
            
            elif col == 'TN' and not df['TAVG'].isna().all() and not df['TX'].isna().all():
                # Use relationship TN ≈ TAVG - (TX - TAVG)
                calc_mask = df['TN'].isna() & df['TAVG'].notna() & df['TX'].notna()
                df.loc[calc_mask, 'TN'] = 2 * df.loc[calc_mask, 'TAVG'] - df.loc[calc_mask, 'TX']
            
            # For remaining missing values, use time-based interpolation
            df[col] = df[col].interpolate(method='time', limit_direction='both')
    
    # Handle humidity with awareness of temperature
    if 'RH_AVG' in df.columns and df['RH_AVG'].isna().any():
        df['RH_AVG'] = df['RH_AVG'].interpolate(method='time', limit_direction='both')
    
    # Pre-process Sunshine Duration for rainfall prediction
    if 'SS' in df.columns and df['SS'].isna().any():
        df['SS'] = df['SS'].interpolate(method='time', limit_direction='both')
    
    # Check if we need month and season as additional features
    df['Month_Num'] = df.index.month
    
    # Create training data for rain occurrence model (only using complete records)
    predictors = ['TAVG', 'RH_AVG', 'SS', 'TX', 'TN', 'Month_Num']
    predictors = [p for p in predictors if p in df.columns]
    
    print(f"Using predictors for rainfall model: {predictors}")
    
    # Create training set where RR is known
    train_data = df.dropna(subset=['RR'] + predictors)
    
    # Create binary target: did it rain or not?
    train_data['rain_occurred'] = (train_data['RR'] > 0).astype(int)
    
    # Check if we have enough data for training
    if len(train_data) < 100:
        print("⚠️ Not enough complete data for rainfall model, using simple methods")
        # Fallback to simple method
        df['RR'] = df['RR'].interpolate(method='time', limit_direction='both')
    else:
        print(f"Training rainfall model with {len(train_data)} complete records")
        
        # Split for validation
        X = train_data[predictors]
        y = train_data['rain_occurred']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train rain occurrence classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluate classifier
        from sklearn.metrics import accuracy_score, f1_score
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"📊 Rain occurrence model: Accuracy={accuracy:.4f}, F1-score={f1:.4f}")
        
        # Feature importance
        importances = clf.feature_importances_
        feature_imp = pd.DataFrame({'Feature': predictors, 'Importance': importances})
        feature_imp = feature_imp.sort_values('Importance', ascending=False)
        print("Top predictors of rainfall:")
        print(feature_imp)
        
        # Identify rows with missing RR values
        missing_rr_mask = df['RR'].isna()
        missing_rr_count = missing_rr_mask.sum()
        
        # Check if we need to impute missing values
        if missing_rr_count > 0:
            print(f"Imputing {missing_rr_mask.sum()} missing rainfall values...")

            # Prepare features for prediction
            X_missing = df.loc[missing_rr_mask, predictors].copy()

            # Fill any remaining NaNs in predictors with median values
            for col in X_missing.columns:
                if X_missing[col].isna().any():
                    X_missing[col] = X_missing[col].fillna(df[col].median())

            # Predict rain occurrence
            rain_predicted = clf.predict(X_missing)

            # IMPORTANT FIX: Create properly aligned boolean masks
            # Get the indices where RR is missing
            missing_indices = df.index[missing_rr_mask]

            # For dry days (no rain), set RR to 0
            dry_indices = missing_indices[rain_predicted == 0]
            df.loc[dry_indices, 'RR'] = 0.0
            print(f"✓ Set {len(dry_indices)} days to zero rainfall")

            # For rainy days, sample from appropriate distribution
            rainy_indices = missing_indices[rain_predicted == 1]
            print(f"Need to sample values for {len(rainy_indices)} rainy days")

            # For each season and month, create distribution of non-zero rainfall
            rainy_day_distributions = {}

            # Group data by month for more precise sampling
            for month in range(1, 13):
                month_rain = df[(df.index.month == month) & (df['RR'] > 0) & (~df['RR'].isna())]['RR']
                if len(month_rain) > 5:  # Need enough samples
                    rainy_day_distributions[month] = month_rain

            # Also group by season as a fallback
            for season in df['Season'].unique():
                if pd.isna(season):
                    continue
                season_rain = df[(df['Season'] == season) & (df['RR'] > 0) & (~df['RR'].isna())]['RR']
                if len(season_rain) > 5:
                    rainy_day_distributions[season] = season_rain

            # Overall distribution as last resort
            overall_rain = df[(df['RR'] > 0) & (~df['RR'].isna())]['RR']
            if len(overall_rain) > 0:
                rainy_day_distributions['overall'] = overall_rain

            # Sample values for rainy days
            for idx in rainy_indices:
                month = idx.month
                season = df.loc[idx, 'Season']
                
                # Try month-specific distribution first (most accurate)
                if month in rainy_day_distributions:
                    sample_source = rainy_day_distributions[month]
                    dist_type = "month"
                # Then try season-specific 
                elif season in rainy_day_distributions:
                    sample_source = rainy_day_distributions[season]
                    dist_type = "season"
                # Fallback to overall distribution
                elif 'overall' in rainy_day_distributions:
                    sample_source = rainy_day_distributions['overall']
                    dist_type = "overall"
                else:
                    # If all else fails, use median of all rainfall values
                    df.loc[idx, 'RR'] = df[(df['RR'] > 0) & (~df['RR'].isna())]['RR'].median()
                    continue
                
                # Sample from the appropriate distribution
                df.loc[idx, 'RR'] = sample_source.sample(1).values[0]
            print(f"✓ Sampled values for {len(rainy_indices)} rainy days")            
    
        # Check for any remaining missing values
        still_missing = df['RR'].isna().sum()
        if still_missing > 0:
            print(f"Filling {still_missing} remaining missing values...")
            df['RR'] = df['RR'].interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
    
    # Ensure no negative values for rainfall
    df['RR'] = df['RR'].clip(lower=0)
    
    # Calculate improved metrics for rainfall
    if 'is_RR_missing' in df.columns:
        originally_missing = df['is_RR_missing'] == 1
        # Can't evaluate imputation accuracy directly, but can show distribution stats
        if originally_missing.sum() > 0:
            print("\nRainfall imputation statistics:")
            print(f"Total imputed values: {originally_missing.sum()}")
            print(f"Imputed zero values: {(df.loc[originally_missing, 'RR'] == 0).sum()}")
            print(f"Imputed rainy values: {(df.loc[originally_missing, 'RR'] > 0).sum()}")
            print(f"Mean of imputed values: {df.loc[originally_missing, 'RR'].mean():.2f}")
            print(f"Max of imputed values: {df.loc[originally_missing, 'RR'].max():.2f}")    
    plot_rr_distribution(df)
    # Remove any temporary columns used for rainfall prediction
    if 'Month_Num' in df.columns:
        df.drop('Month_Num', axis=1, inplace=True)


    # ----- END OF IMPROVED RAINFALL PREPROCESSING -----
    
    # 2. Wind Direction at Maximum Speed (DDD_X)
    print("Processing Wind Direction (DDD_X)...")
    
    # IMPROVEMENT 5: Improved circular mean calculation for wind direction
    def circular_mean(directions):
        """
        Calculate circular mean for wind directions (in degrees)
        with proper handling of the circular nature of angles
        """
        if len(directions) == 0 or directions.isna().all():
            return np.nan
            
        # Convert to radians
        directions_rad = np.radians(directions.astype(float))
        
        # Calculate rectangular coordinates
        x_coords = np.cos(directions_rad)
        y_coords = np.sin(directions_rad)
        
        # Calculate means of coordinates
        x_mean = np.nanmean(x_coords)
        y_mean = np.nanmean(y_coords)
        
        # Convert back to angle
        mean_direction = np.degrees(np.arctan2(y_mean, x_mean))
        
        # Normalize to [0, 360)
        return (mean_direction + 360) % 360
    
    # IMPROVEMENT 6: Calculate confidence measure for circular mean
    def circular_consistency(directions):
        """
        Calculate consistency measure (0-1) for circular data
        0 = completely scattered, 1 = perfectly aligned
        """
        if len(directions) == 0 or directions.isna().all():
            return 0
            
        # Convert to radians
        directions_rad = np.radians(directions.astype(float))
        
        # Calculate rectangular coordinates
        x_coords = np.cos(directions_rad)
        y_coords = np.sin(directions_rad)
        
        # Calculate means of coordinates
        x_mean = np.nanmean(x_coords)
        y_mean = np.nanmean(y_coords)
        
        # Calculate resultant vector length (consistency measure)
        r = np.sqrt(x_mean**2 + y_mean**2)
        
        return r
    
    # For each month, calculate mean direction and consistency
    monthly_wind_data = {}
    for month in range(1, 13):
        month_data = df[df.index.month == month]['DDD_X'].dropna()
        if not month_data.empty:
            direction = circular_mean(month_data)
            consistency = circular_consistency(month_data)
            monthly_wind_data[month] = {'direction': direction, 'consistency': consistency}
    
    # For each season, calculate mean direction and consistency
    seasonal_wind_data = {}
    for season in df['Season'].unique():
        if pd.isna(season):
            continue
        season_data = df[df['Season'] == season]['DDD_X'].dropna()
        if not season_data.empty:
            direction = circular_mean(season_data)
            consistency = circular_consistency(season_data)
            seasonal_wind_data[season] = {'direction': direction, 'consistency': consistency}
    
    # Fill missing values based on consistency of monthly/seasonal patterns
    for idx in df.index[df['DDD_X'].isna()]:
        month = idx.month
        season = df.loc[idx, 'Season']
        
        # Try using monthly mean if consistency is good
        if month in monthly_wind_data and monthly_wind_data[month]['consistency'] > 0.5:
            df.loc[idx, 'DDD_X'] = monthly_wind_data[month]['direction']
        # Otherwise try seasonal mean if consistency is good
        elif season in seasonal_wind_data and seasonal_wind_data[season]['consistency'] > 0.5:
            df.loc[idx, 'DDD_X'] = seasonal_wind_data[season]['direction']
        # Otherwise, try nearest-neighbor approach
        else:
            # Find closest date with valid wind direction (within 7 days)
            closest_idx = None
            min_diff = 7 * 24 * 3600  # 7 days in seconds
            
            for other_idx in df.index[~df['DDD_X'].isna()]:
                diff = abs((other_idx - idx).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = other_idx
            
            if closest_idx is not None:
                df.loc[idx, 'DDD_X'] = df.loc[closest_idx, 'DDD_X']
    
    # For any remaining NaNs, use overall mean direction
    if df['DDD_X'].isna().any():
        overall_mean = circular_mean(df['DDD_X'].dropna())
        df.loc[df['DDD_X'].isna(), 'DDD_X'] = overall_mean
    
    # 3. Average Wind Speed (FF_AVG)
    print("Processing Average Wind Speed (FF_AVG)...")
    
    # IMPROVEMENT 7: More robust correlation threshold and validation
    # Check correlation between FF_X and FF_AVG
    valid_mask = ~df['FF_X'].isna() & ~df['FF_AVG'].isna()
    correlation = df.loc[valid_mask, ['FF_X', 'FF_AVG']].corr().iloc[0, 1]
    print(f"Correlation between FF_X and FF_AVG: {correlation:.4f}")
    
    # For missing FF_AVG where FF_X is available, use linear regression with validation
    if correlation > 0.7:  # IMPROVEMENT: Higher threshold for more robust relationship
        X = df.loc[valid_mask, 'FF_X'].values.reshape(-1, 1)
        y = df.loc[valid_mask, 'FF_AVG'].values
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression().fit(X_train, y_train)
        
        # Validate model
        from sklearn.metrics import r2_score, mean_squared_error
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Linear model: FF_AVG = {model.coef_[0]:.4f} * FF_X + {model.intercept_:.4f}")
        print(f"Validation R² = {r2:.4f}, RMSE = {rmse:.4f}")
        
        # Apply model only if validation metrics are good
        if r2 > 0.65:  # Good R² score
            ff_x_valid_mask = df['FF_X'].notna() & df['FF_AVG'].isna()
            df.loc[ff_x_valid_mask, 'FF_AVG'] = model.predict(df.loc[ff_x_valid_mask, 'FF_X'].values.reshape(-1, 1))
        else:
            print("Linear model validation failed, using alternative methods")
    
    # For remaining missing values, use weather pattern-aware imputation
    # Group similar weather days based on temperature and relative humidity
    if df['FF_AVG'].isna().any():
        try:
            # Group days into weather patterns using temperature and humidity
            from sklearn.cluster import KMeans
            
            # Select features for clustering
            features = df[['TAVG', 'RH_AVG']].copy()
            
            # Handle missing values in features
            features = features.fillna(features.mean())
            
            # Standardize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Determine optimal number of clusters (2-5)
            from sklearn.metrics import silhouette_score
            
            best_score = -1
            best_n_clusters = 2
            
            for n_clusters in range(2, 6):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(features_scaled)
                
                # Skip if only one sample in any cluster
                if np.min(np.bincount(cluster_labels)) < 2:
                    continue
                
                score = silhouette_score(features_scaled, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
            
            # Cluster days into weather patterns
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
            df['weather_pattern'] = kmeans.fit_predict(features_scaled)
            
            print(f"Clustered days into {best_n_clusters} weather patterns for imputation")
            
            # For each missing FF_AVG, use the median value from the same weather pattern
            for pattern in range(best_n_clusters):
                pattern_median = df[df['weather_pattern'] == pattern]['FF_AVG'].median()
                if not np.isnan(pattern_median):
                    pattern_mask = (df['weather_pattern'] == pattern) & df['FF_AVG'].isna()
                    df.loc[pattern_mask, 'FF_AVG'] = pattern_median
            
            # Remove temporary column
            df.drop('weather_pattern', axis=1, inplace=True)
        except Exception as e:
            print(f"Weather pattern clustering failed: {e}")
    
    # For any remaining missing values, use interpolation
    if df['FF_AVG'].isna().any():
        df['FF_AVG'] = df['FF_AVG'].interpolate(method='time', limit_direction='both')
    
    # 4. Temperature (TX, TN, TAVG) & Humidity (RH_AVG)
    print("Processing Temperature and Humidity...")
    
    # IMPROVEMENT 8: More sophisticated temperature imputation
    for col in ['TX', 'TN', 'TAVG']:
        if col in df.columns and df[col].isna().any():
            # Try to maintain the relationship between daily min, max, and average temps
            if col == 'TAVG' and not df['TX'].isna().all() and not df['TN'].isna().all():
                # Calculate TAVG from TX and TN where both are available
                calc_mask = df['TAVG'].isna() & df['TX'].notna() & df['TN'].notna()
                df.loc[calc_mask, 'TAVG'] = (df.loc[calc_mask, 'TX'] + df.loc[calc_mask, 'TN']) / 2
            
            elif col == 'TX' and not df['TAVG'].isna().all() and not df['TN'].isna().all():
                # Use relationship TX ≈ TAVG + (TAVG - TN)
                calc_mask = df['TX'].isna() & df['TAVG'].notna() & df['TN'].notna()
                df.loc[calc_mask, 'TX'] = 2 * df.loc[calc_mask, 'TAVG'] - df.loc[calc_mask, 'TN']
            
            elif col == 'TN' and not df['TAVG'].isna().all() and not df['TX'].isna().all():
                # Use relationship TN ≈ TAVG - (TX - TAVG)
                calc_mask = df['TN'].isna() & df['TAVG'].notna() & df['TX'].notna()
                df.loc[calc_mask, 'TN'] = 2 * df.loc[calc_mask, 'TAVG'] - df.loc[calc_mask, 'TX']
            
            # Apply constraints (TX ≥ TAVG ≥ TN)
            if 'TX' in df.columns and 'TAVG' in df.columns:
                invalid_mask = df['TAVG'] > df['TX']
                if invalid_mask.sum() > 0:
                    df.loc[invalid_mask, 'TAVG'] = df.loc[invalid_mask, 'TX']
            
            if 'TN' in df.columns and 'TAVG' in df.columns:
                invalid_mask = df['TAVG'] < df['TN']
                if invalid_mask.sum() > 0:
                    df.loc[invalid_mask, 'TAVG'] = df.loc[invalid_mask, 'TN']
            
            # For remaining missing values, use time-based interpolation
            df[col] = df[col].interpolate(method='time', limit_direction='both')
    
    # Handle humidity with awareness of temperature
    if 'RH_AVG' in df.columns and df['RH_AVG'].isna().any():
        # Calculate dewpoint where available
        mask = df['RH_AVG'].notna() & df['TAVG'].notna()
        
        if mask.sum() > 50:  # Need enough data to establish relationship
            # Calculate dewpoint temperature
            df.loc[mask, 'dewpoint'] = df.loc[mask, 'TAVG'] - ((100 - df.loc[mask, 'RH_AVG']) / 5)
            
            # Interpolate dewpoint
            df['dewpoint'] = df['dewpoint'].interpolate(method='time', limit_direction='both')
            
            # Calculate humidity from dewpoint where missing
            rh_missing = df['RH_AVG'].isna() & df['TAVG'].notna() & df['dewpoint'].notna()
            df.loc[rh_missing, 'RH_AVG'] = 100 - 5 * (df.loc[rh_missing, 'TAVG'] - df.loc[rh_missing, 'dewpoint'])
            
            # Apply physical constraints
            df['RH_AVG'] = df['RH_AVG'].clip(0, 100)
            
            # Remove temporary column
            df.drop('dewpoint', axis=1, inplace=True)
    
    # For any remaining missing values in RH_AVG, use interpolation
    if 'RH_AVG' in df.columns and df['RH_AVG'].isna().any():
        df['RH_AVG'] = df['RH_AVG'].interpolate(method='time', limit_direction='both')
    
    # 5. Sunshine Duration (SS)
    print("Processing Sunshine Duration (SS)...")
    
    # IMPROVEMENT 9: Location-aware daylight hours
    # More accurate estimate of max daylight hours based on latitude
    latitudes = {
        "Aceh": 5.5,      # North Sumatra approx 5.5°N
    }
    
    latitude = latitudes.get(location, 0)  # Default to equator if unknown
    
    # Calculate approximate max daylight hours (Cooper equation)
    def max_daylight_hours(day_of_year, latitude):
        """
        Calculate maximum possible daylight hours based on day of year and latitude
        Cooper equation (1969)
        """
        # Convert latitude to radians
        lat_rad = np.radians(latitude)
        
        # Calculate solar declination
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        declination_rad = np.radians(declination)
        
        # Calculate day length in hours
        cos_hour_angle = -np.tan(lat_rad) * np.tan(declination_rad)
        
        # Handle special cases near poles
        cos_hour_angle = np.clip(cos_hour_angle, -1, 1)
        
        # Calculate sunrise hour angle in radians
        hour_angle = np.arccos(cos_hour_angle)
        
        # Convert hour angle to hours of daylight
        day_length = 2 * hour_angle * 24 / (2 * np.pi)
        
        return day_length
    
    # Calculate day of year for each date
    df['day_of_year'] = df.index.dayofyear
    
    # Calculate max daylight hours for each day
    df['max_daylight'] = df['day_of_year'].apply(lambda d: max_daylight_hours(d, latitude))
    
    # Interpolate sunshine duration
    df['SS'] = df['SS'].interpolate(method='time', limit_direction='both')
    
    # Apply physical constraints (sunshine hours cannot exceed daylight hours)
    df['SS'] = df[['SS', 'max_daylight']].min(axis=1)
    
    # Ensure no negative values
    df['SS'] = df['SS'].clip(lower=0)
    
    # 6. Most Common Wind Direction (DDD_CAR)
    print("Processing Cardinal Wind Direction (DDD_CAR)...")
    
    # IMPROVEMENT 10: Better handling of categorical wind direction data
    # Convert to string type to handle categorical data properly
    df['DDD_CAR'] = df['DDD_CAR'].astype(str)
    
    # Replace 'nan' string with np.nan
    df.loc[df['DDD_CAR'] == 'nan', 'DDD_CAR'] = np.nan
    
    # Check for valid cardinal directions
    valid_directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                         'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    
    # Convert invalid values to NaN
    invalid_mask = ~df['DDD_CAR'].isin(valid_directions) & ~df['DDD_CAR'].isna()
    if invalid_mask.sum() > 0:
        print(f"Found {invalid_mask.sum()} invalid cardinal directions")
        df.loc[invalid_mask, 'DDD_CAR'] = np.nan
    
    # Check DDD_X against DDD_CAR for consistency
    if 'DDD_X' in df.columns:
        # Function to convert degrees to cardinal direction
        def degrees_to_cardinal(degrees):
            if np.isnan(degrees):
                return np.nan
                
            # Define direction ranges in degrees
            dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                    'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
            
            # Each direction covers 22.5 degrees
            idx = round(degrees / 22.5) % 16
            return dirs[idx]
        
        # Where DDD_CAR is missing but DDD_X is available, use DDD_X
        missing_car_mask = df['DDD_CAR'].isna() & df['DDD_X'].notna()
        if missing_car_mask.sum() > 0:
            df.loc[missing_car_mask, 'DDD_CAR'] = df.loc[missing_car_mask, 'DDD_X'].apply(degrees_to_cardinal)
    
    # For remaining missing values, use mode from same month
    for month in range(1, 13):
        # Get mode for this month
        month_data = df[df.index.month == month]['DDD_CAR'].dropna()
        if not month_data.empty:
            month_mode = month_data.mode()[0]
            
            # Fill missing values for this month
            month_mask = (df.index.month == month) & df['DDD_CAR'].isna()
            df.loc[month_mask, 'DDD_CAR'] = month_mode
    
    # For any remaining missing values, use mode from same season
    for season in df['Season'].unique():
        if pd.isna(season):
            continue
            
        # Get mode for this season
        season_data = df[df['Season'] == season]['DDD_CAR'].dropna()
        if not season_data.empty:
            season_mode = season_data.mode()[0]
            
            # Fill missing values for this season
            season_mask = (df['Season'] == season) & df['DDD_CAR'].isna()
            df.loc[season_mask, 'DDD_CAR'] = season_mode
    
    # Any remaining NaNs, fill with the overall most common direction
    if df['DDD_CAR'].isna().any():
        overall_mode = df['DDD_CAR'].value_counts().idxmax() if not df['DDD_CAR'].dropna().empty else 'N'
        df['DDD_CAR'] = df['DDD_CAR'].fillna(overall_mode)

        # Final check for any remaining missing values
    missing_after = df.isnull().sum() / len(df) * 100
    print("\nMissing values percentage after preprocessing:")
    print(missing_after)
    
    # Drop the temporary column
    if 'max_daylight' in df.columns:
        df.drop('max_daylight', axis=1, inplace=True)
    
    # Save preprocessed data
    df = df.drop(columns=['day_of_year'], errors='ignore')
    df.to_csv(output_file)
    print(f"\nPreprocessed data saved to {output_file}")
    
    return df

def create_diagnostic_plots(processed_df, output_file='BMKG_Data_Diagnostic_Plots.png'):
    """
    Create improved diagnostic plots for the processed BMKG data
    """
    # Create figure with more space
    fig = plt.figure(figsize=(20, 16))
    
    # Set larger spacing between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Plot 1: Rainfall data
    ax1 = plt.subplot(2, 2, 1)
    processed_df['RR'].plot(ax=ax1, color='blue', alpha=0.7)
    
    # Add a rolling average to see the trend better
    processed_df['RR'].rolling(window=30).mean().plot(
        ax=ax1, color='red', linewidth=2, 
        label='30-day Moving Average'
    )
    
    # Set reasonable y-axis limits for rainfall (adjust based on your data)
    y_max_rain = min(processed_df['RR'].quantile(0.99) * 1.5, processed_df['RR'].max())
    ax1.set_ylim(0, y_max_rain)
    
    ax1.set_title('Rainfall After Preprocessing', fontsize=14, pad=20)
    ax1.set_ylabel('Rainfall (mm)', fontsize=12)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Temperature data
    ax2 = plt.subplot(2, 2, 2)
    processed_df['TX'].plot(ax=ax2, color='red', alpha=0.5, label='Max Temp')
    processed_df['TAVG'].plot(ax=ax2, color='green', alpha=0.7, label='Avg Temp')
    processed_df['TN'].plot(ax=ax2, color='blue', alpha=0.5, label='Min Temp')
    
    ax2.set_title('Temperature After Preprocessing', fontsize=14, pad=20)
    ax2.set_ylabel('Temperature (°C)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    
    # Move legend to a better position
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)
    
    # Plot 3: Humidity
    ax3 = plt.subplot(2, 2, 3)
    processed_df['RH_AVG'].plot(ax=ax3, color='blue', alpha=0.6)
    processed_df['RH_AVG'].rolling(window=30).mean().plot(
        ax=ax3, color='darkblue', linewidth=2, 
        label='30-day Moving Average'
    )
    
    ax3.set_title('Relative Humidity After Preprocessing', fontsize=14, pad=20)
    ax3.set_ylabel('Humidity (%)', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylim(40, 100)  # Reasonable range for humidity
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Plot 4: Sunshine hours
    ax4 = plt.subplot(2, 2, 4)
    processed_df['SS'].plot(ax=ax4, color='orange', alpha=0.6)
    processed_df['SS'].rolling(window=30).mean().plot(
        ax=ax4, color='red', linewidth=2, 
        label='30-day Moving Average'
    )
    
    ax4.set_title('Sunshine Hours After Preprocessing', fontsize=14, pad=20)
    ax4.set_ylabel('Sunshine Duration (hours)', fontsize=12)
    ax4.set_xlabel('Date', fontsize=12)
    ax4.set_ylim(0, processed_df['SS'].quantile(0.99) * 1.2)  # Set reasonable y-limit
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # Add a main title
    plt.suptitle('BMKG Data Diagnostic Plots', fontsize=18, y=0.98)
    
    # Apply date formatting to all x-axes
    for ax in [ax1, ax2, ax3, ax4]:
        # Format x-axis dates nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))  # Show every 2 years
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Save the improved plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Improved diagnostic plots saved to '{output_file}'")
    
    # Optional: Create additional plots for more detailed analysis
    create_seasonal_plots(processed_df, 'BMKG_Data_Seasonal_Analysis.png')


def create_seasonal_plots(processed_df, output_file='BMKG_Data_Seasonal_Analysis.png'):
    """
    Create additional plots showing seasonal patterns
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Monthly average rainfall boxplot
    ax1 = plt.subplot(2, 2, 1)
    processed_df.reset_index().boxplot(
        column='RR', by='Month', ax=ax1, 
        grid=False, showfliers=False  # Hide outliers for clarity
    )
    ax1.set_title('Monthly Rainfall Distribution', fontsize=14)
    ax1.set_ylabel('Rainfall (mm)', fontsize=12)
    ax1.set_xlabel('')
    plt.suptitle('')  # Remove default boxplot title
    
    # Plot 2: Monthly average temperature
    ax2 = plt.subplot(2, 2, 2)
    by_month = processed_df.groupby(processed_df.index.month)
    by_month['TX'].mean().plot(ax=ax2, marker='o', color='red', label='Max Temp')
    by_month['TAVG'].mean().plot(ax=ax2, marker='o', color='green', label='Avg Temp')
    by_month['TN'].mean().plot(ax=ax2, marker='o', color='blue', label='Min Temp')
    
    ax2.set_title('Monthly Average Temperature', fontsize=14)
    ax2.set_ylabel('Temperature (°C)', fontsize=12)
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    # Plot 3: Wet vs Dry season comparison
    ax3 = plt.subplot(2, 2, 3)
    processed_df.groupby('Season')['RR'].mean().plot(
        kind='bar', ax=ax3, color=['blue', 'orange']
    )
    ax3.set_title('Average Rainfall by Season', fontsize=14)
    ax3.set_ylabel('Average Rainfall (mm)', fontsize=12)
    ax3.grid(alpha=0.3, axis='y')
    
    # Plot 4: Annual rainfall trend - FIXED
    ax4 = plt.subplot(2, 2, 4)
    annual_rain = processed_df.resample('Y')['RR'].sum()
    annual_rain.plot(ax=ax4, marker='o', linestyle='-', color='blue')
    
    # Add trend line - FIXED to avoid converter warning
    # Get years as float values for trend calculation
    years = np.array([d.year + d.dayofyear/365.25 for d in annual_rain.index])
    y_values = annual_rain.values
    
    # Fit the trend line using years as float
    z = np.polyfit(years, y_values, 1)
    p = np.poly1d(z)
    
    # Create a separate line artist and add it to the plot
    trend_line = plt.Line2D(annual_rain.index, p(years), 
                           color='r', linestyle='--', linewidth=2)
    ax4.add_artist(trend_line)
    
    # Add the trend label separately
    ax4.plot([], [], 'r--', linewidth=2, label=f'Trend: {z[0]:.2f} mm/year')
    
    ax4.set_title('Annual Rainfall Trend', fontsize=14)
    ax4.set_ylabel('Total Annual Rainfall (mm)', fontsize=12)
    ax4.set_xlabel('Year', fontsize=12)
    ax4.grid(alpha=0.3)
    ax4.legend()
    
    # Add a main title
    plt.suptitle('BMKG Data Seasonal Analysis', fontsize=18, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the seasonal analysis plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Seasonal analysis plots saved to '{output_file}'")

if __name__ == "__main__":
    input_file = "/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data BMKG/Stasiun Klimatologi Aceh/CSV/BMKG_Data_All.csv"
    output_file = "/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data BMKG/Stasiun Klimatologi Aceh/CSV CLEANED/BMKG_Data_Cleaned.csv"
    
    try:
        processed_df = preprocess_bmkg_data(input_file, output_file)
        
        if processed_df is not None:
        # Generate improved diagnostic plots
            create_diagnostic_plots(processed_df)

        else:
            print("Preprocessing failed, cannot generate diagnostic plots.")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()