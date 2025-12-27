import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import numpy as np
import re

# --- CONFIGURATION ---
# 1. Path to your folder containing the 43 .nc files
# CHANGE THIS to your actual folder path. 
# If you provide a folder path like ".../Data", the script will now automatically look for .nc files inside it.
data_folder = "Data/*.nc" 

# 2. Define your Economic Zones (Lat/Lon boundaries)
zones = {
    "Mumbai":      {"lat": slice(19.5, 18.5), "lon": slice(72.0, 73.0)}, # Note: Slice is North -> South
    "Ratnagiri":   {"lat": slice(17.5, 16.5), "lon": slice(72.0, 73.5)},
    "Sindhudurg":  {"lat": slice(16.5, 15.5), "lon": slice(72.0, 73.5)}
}

# --- STEP 1: LOAD AND MERGE DATA ---
print("Loading files...")

# Logic to handle if user provided a folder path instead of a glob pattern
if os.path.isdir(data_folder):
    print(f"Note: '{data_folder}' is a directory. Automatically searching for .nc files inside it.")
    data_folder = os.path.join(data_folder, "*.nc")

file_list = sorted(glob.glob(data_folder))

# Filter to ensure we only try to open actual files, not directories
file_list = [f for f in file_list if os.path.isfile(f)]

if not file_list:
    print(f"Error: No .nc files found matching: {data_folder}")
    print("Tip: Check your path and ensure files end in .nc")
    exit()

try:
    # Alternative loading method: Sequential load to avoid 'dask' dependency error
    print(f"Found {len(file_list)} files. Loading sequentially...")
    datasets = []
    
    for f in file_list:
        try:
            # Open individual file (loads into memory, avoiding dask chunks)
            ds_i = xr.open_dataset(f, engine='netcdf4')
            
            # Extract time from filename
            filename = os.path.basename(f)
            
            # Try to find YYYYMMDD
            match = re.search(r'(2025)(\d{2})(\d{2})', filename) 
            # Fallback for Julian Day format YYYYDDD
            match_julian = re.search(r'(2025)(\d{3})', filename)
            
            timestamp = None
            if match:
                date_str = match.group(0)
                timestamp = pd.to_datetime(date_str, format='%Y%m%d')
            elif match_julian:
                year = int(match_julian.group(1))
                day_of_year = int(match_julian.group(2))
                timestamp = pd.to_datetime(year * 1000 + day_of_year, format='%Y%j')
            
            if timestamp is not None:
                # Add time dimension to this single file
                ds_i = ds_i.expand_dims(time=[timestamp])
                datasets.append(ds_i)
            else:
                print(f"Warning: Could not parse date from {filename}, skipping.")
                
        except Exception as e:
            print(f"Skipping {f} due to error: {e}")
            continue

    if not datasets:
        print("Error: No valid datasets could be loaded.")
        exit()

    # Concatenate all loaded files into one dataset
    ds = xr.concat(datasets, dim='time')
    ds = ds.sortby('time') # Ensure chronological order
    
    print(f"Successfully compiled dataset with {len(ds.time)} time steps.")

except Exception as e:
    print(f"Critical Error loading files: {e}")
    print("Tip: Ensure files are uniform and filenames have dates.")
    exit()

# --- STEP 2: RESAMPLING (Handling Clouds) ---
# Resample to Weekly averages to fill gaps
print("Aggregating data to Weekly averages (filling cloud gaps)...")
ds_weekly = ds['chlor_a'].resample(time='1W').mean(skipna=True)

# --- STEP 3: EXTRACT ZONAL DATA ---
print("Extracting data for Economic Zones...")
results = {}

plt.figure(figsize=(12, 6))

for zone_name, bounds in zones.items():
    # Select the specific lat/lon box
    zone_data = ds_weekly.sel(lat=bounds['lat'], lon=bounds['lon'])
    
    # Calculate the spatial average for that zone over time
    # skipna=True is crucial here to ignore cloud pixels
    time_series = zone_data.mean(dim=['lat', 'lon'], skipna=True)
    
    results[zone_name] = time_series
    
    # Plotting
    plt.plot(time_series.time, time_series, marker='o', label=zone_name)

# --- STEP 4: VISUALIZATION ---
plt.title("Weekly Chlorophyll-a Concentration (Konkan Coast 2025)\nProxy for Marine Productivity", fontsize=14)
plt.ylabel("Chlorophyll-a (mg/m³)", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Highlight the "Post-Monsoon Bloom" expectation
plt.axvspan(pd.Timestamp('2025-10-01'), pd.Timestamp('2025-11-30'), color='green', alpha=0.1, label='Expected Bloom Period')

plt.tight_layout()
plt.show()

# --- STEP 5: BASIC INTERPRETATION ---
print("\n--- ANALYSIS REPORT ---")
for zone, data in results.items():
    # Calculate peak value
    peak_val = data.max().values
    peak_date = data.idxmax().values
    mean_val = data.mean().values
    
    print(f"\nZONE: {zone}")
    print(f"  - Average Productivity: {mean_val:.2f} mg/m³")
    if pd.notnull(peak_val):
        print(f"  - Peak Productivity: {peak_val:.2f} mg/m³ (on {str(peak_date)[:10]})")
        
        if peak_val < 1.0:
            print("  - STATUS: LOW PRODUCTIVITY (Potential Fisheries Risk)")
        elif peak_val > 5.0:
            print("  - STATUS: VERY HIGH (Potential Algal Bloom Risk)")
        else:
            print("  - STATUS: HEALTHY/NORMAL")
    else:
        print("  - STATUS: NO DATA (Check cloud cover or file selection)")

print("\nProcessing Complete.")