import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
# Folder where you put the 3 CMEMS .nc files
data_folder = "biogeochem_data" 

# Define filenames (make sure these match exactly)
files = {
    "bio": "konkan_bgc_chl_phyc.nc",
    "nut": "konkan_bgc_no3.nc",
    "prod": "konkan_bgc_nppv_o2.nc"
}

# Define Ratnagiri Zone for Analysis
lat_slice = slice(16.5, 17.5)
lon_slice = slice(72.0, 73.5)

# --- STEP 1: LOAD AND MERGE DATASETS ---
print("Loading 3D Biogeochemical Data...")

datasets = []
for key, fname in files.items():
    path = os.path.join(data_folder, fname)
    if not os.path.exists(path):
        print(f"Error: File {fname} not found in {data_folder}")
        exit()
    
    # Open dataset
    ds = xr.open_dataset(path)
    datasets.append(ds)

# Merge into one big dataset
# They should align automatically on time/depth/lat/lon
try:
    ds_merged = xr.merge(datasets)
    print("Successfully merged 3D datasets.")
except Exception as e:
    print(f"Merge error: {e}")
    exit()

# Extract Ratnagiri Zone
print("Extracting Ratnagiri Zone Data...")
ratnagiri = ds_merged.sel(latitude=lat_slice, longitude=lon_slice)

# --- STEP 2: CALCULATE 3D METRICS ---
print("Calculating Integrated Water Column Metrics...")

# 2a. Total Integrated Production (0m to 100m)
# We select depth up to ~100m (Photic Zone) and sum it up
# Note: Check unit conversion if needed (mg/m3 -> mg/m2), here we do a simple sum proxy
npp_integrated = ratnagiri['nppv'].sel(depth=slice(0, 100)).sum(dim='depth').mean(dim=['latitude', 'longitude'])

# 2b. Surface vs Deep Nitrate (Upwelling Check)
no3_surface = ratnagiri['no3'].sel(depth=0.49, method='nearest').mean(dim=['latitude', 'longitude'])
no3_deep = ratnagiri['no3'].sel(depth=100, method='nearest').mean(dim=['latitude', 'longitude'])

# 2c. Deep Oxygen (Hypoxia Check)
# Check Oxygen at 30m-50m (common fish habitat)
o2_deep = ratnagiri['o2'].sel(depth=50, method='nearest').mean(dim=['latitude', 'longitude'])

# --- STEP 3: VISUALIZATION ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Plot 1: The "True" Food Supply (Integrated Production)
ax1.plot(npp_integrated.time, npp_integrated, color='green', linewidth=2, label='Total Water Column Production (0-100m)')
ax1.set_title("1. True Fish Carrying Capacity (Integrated NPP)", fontweight='bold')
ax1.set_ylabel("Production Proxy (Sum mg/m3)", fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: The Nutrient Engine (Surface vs Deep)
ax2.plot(no3_deep.time, no3_deep, color='purple', linestyle='--', label='Deep Nitrate (Reserve)')
ax2.plot(no3_surface.time, no3_surface, color='blue', label='Surface Nitrate (Available)')
ax2.set_title("2. Upwelling Diagnostics (Nutrient Pump)", fontweight='bold')
ax2.set_ylabel("Nitrate (mmol/m3)", fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: The Oxygen Limit
ax3.plot(o2_deep.time, o2_deep, color='red', label='Oxygen at 50m')
# Hypoxia threshold is approx 60 mmol/m3 (depends on species, approx 2 mg/L)
ax3.axhline(y=60, color='black', linestyle=':', label='Hypoxia Threshold')
ax3.set_title("3. Habitat Viability (Oxygen)", fontweight='bold')
ax3.set_ylabel("Dissolved O2 (mmol/m3)", fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.show()

# --- STEP 4: INTERPRETATION & FORECAST ---
print("\n" + "="*50)
print("3D BIO-MODEL FORECAST REPORT (Q1 2026)")
print("="*50)

# Check Bloom Status (Oct-Nov)
bloom_prod = npp_integrated.sel(time=slice('2025-10-01', '2025-11-30')).mean().values
avg_prod = npp_integrated.mean().values # Yearly average

# Check Nutrient Status
bloom_no3 = no3_surface.sel(time=slice('2025-10-01', '2025-11-30')).mean().values

# Check Oxygen Status
min_o2 = o2_deep.min().values

print(f"1. Integrated Production (Bloom Period): {bloom_prod:.2f} (Year Avg: {avg_prod:.2f})")
if bloom_prod < avg_prod:
    print("   -> STATUS: BLOOM FAILURE CONFIRMED in 3D Model.")
    print("   -> Implication: The failure is not just at the surface; the whole water column is empty.")
else:
    print("   -> STATUS: Hidden Sub-surface Bloom Detected!")
    print("   -> Implication: Satellite was wrong! Fish food exists deep down.")

print(f"\n2. Nutrient Diagnostic (Surface NO3): {bloom_no3:.4f} mmol/m3")
if bloom_no3 < 0.5:
    print("   -> CAUSE: NUTRIENT LIMITATION. Upwelling failed to breach the surface.")
else:
    print("   -> CAUSE: Nutrients available. Failure due to other factor (Temp/Light).")

print(f"\n3. Oxygen Safety: Minimum O2 at 50m was {min_o2:.2f}")
if min_o2 < 60:
    print("   -> WARNING: HYPOXIC EVENT DETECTED. Fish likely fled the zone.")
else:
    print("   -> STATUS: Oxygen levels safe.")

print("="*50)


