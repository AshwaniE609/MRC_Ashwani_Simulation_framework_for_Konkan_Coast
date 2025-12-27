import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import numpy as np
import re

# --- CONFIGURATION ---
data_folder = "Data/Modis CHL reading/*.nc" 

# Economic Zones (Lat/Lon)
zones = {
    "Mumbai":      {"lat": slice(19.5, 18.5), "lon": slice(72.0, 73.0)},
    "Ratnagiri":   {"lat": slice(17.5, 16.5), "lon": slice(72.0, 73.5)},
    "Sindhudurg":  {"lat": slice(16.5, 15.5), "lon": slice(72.0, 73.5)}
}

# --- MODEL PARAMETERS (Based on Research Note) ---
# Baseline: Healthy post-monsoon Chl-a concentration (mg/m^3)
# Source: Typical values for Eastern Arabian Sea during Oct-Nov bloom
BASELINE_CHL = 2.0

# Sensitivity Factor: How much fisheries yield drops per unit drop in productivity
# 0.8 means a 10% drop in algae leads to an 8% drop in fish biomass (trophic transfer efficiency proxy)
TROPHIC_SENSITIVITY = 0.8 

# --- STEP 1: LOAD DATA (Sequential Method) ---
print("Loading satellite data for prediction model...")
if os.path.isdir(data_folder):
    data_folder = os.path.join(data_folder, "*.nc")
file_list = sorted(glob.glob(data_folder))
file_list = [f for f in file_list if os.path.isfile(f)]

if not file_list:
    print("Error: No files found.")
    exit()

datasets = []
for f in file_list:
    try:
        ds_i = xr.open_dataset(f, engine='netcdf4')
        filename = os.path.basename(f)
        match = re.search(r'(2025)(\d{2})(\d{2})', filename) 
        match_julian = re.search(r'(2025)(\d{3})', filename)
        
        timestamp = None
        if match:
            timestamp = pd.to_datetime(match.group(0), format='%Y%m%d')
        elif match_julian:
            timestamp = pd.to_datetime(int(match_julian.group(1)) * 1000 + int(match_julian.group(2)), format='%Y%j')
        
        if timestamp:
            ds_i = ds_i.expand_dims(time=[timestamp])
            datasets.append(ds_i)
    except:
        continue

ds = xr.concat(datasets, dim='time').sortby('time')
# Resample to weekly to smooth noise
ds_weekly = ds['chlor_a'].resample(time='1W').mean(skipna=True)

# --- STEP 2: RUN PREDICTION MODEL ---
print("\n--- RUNNING FISHERIES PREDICTION SIMULATION ---")
print(f"Model Baseline (Healthy Ecosystem): {BASELINE_CHL} mg/m³")
print(f"Trophic Sensitivity Factor: {TROPHIC_SENSITIVITY}")

fig, ax = plt.subplots(figsize=(12, 6))

colors = {'Mumbai': 'blue', 'Ratnagiri': 'orange', 'Sindhudurg': 'green'}

prediction_report = []

for zone_name, bounds in zones.items():
    # 1. Extract Zone Data
    zone_data = ds_weekly.sel(lat=bounds['lat'], lon=bounds['lon'])
    ts = zone_data.mean(dim=['lat', 'lon'], skipna=True)
    
    # 2. Isolate the Critical "Bloom Period" (Oct-Nov) for Prediction
    # We only care about the bloom failure for the forecast
    bloom_period = ts.sel(time=slice('2025-10-01', '2025-11-30'))
    
    if len(bloom_period) > 0:
        avg_observed_chl = bloom_period.mean().values
    else:
        avg_observed_chl = 0.0 # No data in bloom period
        
    # 3. Calculate Primary Productivity Index (PPI) Anomaly
    # PPI Anomaly = (Observed - Baseline) / Baseline
    ppi_anomaly = (avg_observed_chl - BASELINE_CHL) / BASELINE_CHL
    
    # 4. Predict Fisheries Yield Gap
    # If PPI is negative (drop in food), fisheries yield drops
    # Formula: Predicted_Decline = PPI_Anomaly * Sensitivity
    yield_gap_percent = ppi_anomaly * TROPHIC_SENSITIVITY * 100
    
    # Cap positive growth predictions (ecosystems take time to recover, but crash quickly)
    if yield_gap_percent > 0:
        yield_gap_percent = yield_gap_percent * 0.2 # Dampen optimism
        
    # Store for Report
    prediction_report.append({
        "Zone": zone_name,
        "Observed_Chl": avg_observed_chl,
        "PPI_Anomaly": ppi_anomaly * 100,
        "Yield_Forecast": yield_gap_percent
    })

    # 5. Plotting
    ax.plot(ts.time, ts, label=f"{zone_name} (Observed)", color=colors[zone_name], marker='o', markersize=4)

# --- STEP 3: VISUALIZATION ---
# Plot Baseline
ax.axhline(y=BASELINE_CHL, color='red', linestyle='--', linewidth=2, label=f'Healthy Baseline ({BASELINE_CHL} mg/m³)')
ax.text(ds_weekly.time[0].values, BASELINE_CHL + 0.1, " Target Productivity", color='red', fontsize=10)

ax.set_title("Eco-Economic Prediction: Productivity Gap Analysis (Konkan 2025)", fontsize=14)
ax.set_ylabel("Chlorophyll-a (mg/m³)", fontsize=12)
ax.set_xlabel("Date", fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Add "Risk Zone" shading
ax.fill_between(ds_weekly.time, 0, 1.0, color='red', alpha=0.1, label='Critical Failure Zone')

plt.tight_layout()
plt.show()

# --- STEP 4: GENERATE TEXT REPORT ---
print("\n" + "="*50)
print("PREDICTIVE GOVERNANCE REPORT: Q1 2026 FORECAST")
print("="*50)
print(f"{'ZONE':<15} | {'OBSERVED (mg/m³)':<18} | {'ECO-HEALTH (PPI)':<18} | {'FISH CATCH FORECAST':<20}")
print("-" * 75)

for row in prediction_report:
    # Formatting status
    obs = row['Observed_Chl']
    ppi = row['PPI_Anomaly']
    forecast = row['Yield_Forecast']
    
    status = "STABLE"
    if ppi < -20: status = "STRESSED"
    if ppi < -50: status = "CRITICAL"
    
    forecast_str = f"{forecast:.1f}% Decline"
    if forecast > 0: forecast_str = f"+{forecast:.1f}% Growth"
    
    print(f"{row['Zone']:<15} | {obs:<18.2f} | {ppi:>6.1f}% ({status})   | {forecast_str:<20}")

print("-" * 75)
print("INTERPRETATION:")
print("1. PPI Anomaly: % deviation from healthy ecosystem baseline.")
print("2. Catch Forecast: Estimated impact on fisheries volume for next season.")
print("   (Based on trophic transfer sensitivity of 0.8)")
print("="*50)