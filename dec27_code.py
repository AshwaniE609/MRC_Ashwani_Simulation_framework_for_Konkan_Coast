import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os


# --- CONFIGURATION ---
data_folder = "biogeochem_data" 
lat_slice = slice(16.5, 17.5)
lon_slice = slice(72.0, 73.5) 


files = {
    "prod": "konkan_bgc_nppv_o2.nc",  # Contains NPP and O2
    "nut":  "konkan_bgc_no3.nc",      # Contains Nitrate
    "bio":  "konkan_bgc_chl_phyc.nc"  # Contains Chlorophyll
}


print("="*70)
print("ROBUST 3D BIOGEOCHEMICAL ANALYSIS - RATNAGIRI ZONE")
print("="*70)


# --- HELPER FUNCTION (IMPROVED) ---
def extract_and_process(fname, var_name, label):
    """
    Load and validate a variable from NetCDF file.
    Returns: (data_array, dataset) or (None, None) if failed
    """
    path = os.path.join(data_folder, fname)
    if not os.path.exists(path):
        print(f"❌ Missing {fname}")
        return None, None

    try:
        ds = xr.open_dataset(path)
        
        # 1. Find the variable (exact match or partial match)
        target_var = None
        if var_name in ds.data_vars:
            target_var = var_name
        else:
            # Try case-insensitive partial match
            for v in ds.data_vars:
                if var_name.lower() in v.lower():
                    target_var = v
                    print(f"   ℹ️  Using '{v}' for requested '{var_name}'")
                    break
        
        if target_var is None:
            print(f"⚠️  Variable '{var_name}' not found in {fname}")
            print(f"   Available: {list(ds.data_vars)}")
            return None, None

        # 2. Extract spatial subset
        subset = ds[target_var].sel(latitude=lat_slice, longitude=lon_slice)
        
        # 3. Validate data (check for all-NaN)
        # Use a sample time point to check
        sample = subset.isel(time=0) if 'time' in subset.dims else subset
        valid_count = int((~sample.isnull()).sum())
        
        if valid_count == 0:
            print(f"❌ {label}: Zone contains only NaN (likely all land)")
            print(f"   → Try adjusting lat/lon bounds")
            print(f"   → Current: Lat {lat_slice}, Lon {lon_slice}")
            return None, None
        
        print(f"✅ {label}: Loaded successfully ({valid_count} valid grid cells)")
        return subset, ds

    except Exception as e:
        print(f"❌ Error processing {fname}: {e}")
        return None, None


# --- STEP 1: PROCESS PRODUCTION (NPP) ---
print("\n" + "─"*70)
print("1. PROCESSING PRODUCTIVITY (NPP)")
print("─"*70)
npp_subset, ds_prod = extract_and_process(files['prod'], 'nppv', 'NPP')

integrated_npp = None
if npp_subset is not None:
    try:
        # Select 0-100m depth range
        npp_100m = npp_subset.sel(depth=slice(0, 100))
        
        # Check if data has variation (not all zeros/NaN)
        if npp_100m.mean().values == 0 or np.isnan(npp_100m.mean().values):
            print("   ⚠️  NPP data is all zeros/NaN - trying alternative...")
            # Fallback: try using chlorophyll instead
            chl_subset, _ = extract_and_process(files['bio'], 'chl', 'Chlorophyll')
            if chl_subset is not None:
                print("   ✅ Using chlorophyll as productivity proxy")
                npp_100m = chl_subset.sel(depth=slice(0, 100))
        
        # Integrate over depth (trapezoidal rule)
        integrated_npp = npp_100m.integrate(coord='depth')
        
        # Average over space
        integrated_npp = integrated_npp.mean(dim=['latitude', 'longitude'], skipna=True)
        
        mean_val = float(integrated_npp.mean().values)
        print(f"   ✅ Integrated NPP calculated")
        print(f"      Mean: {mean_val:.2f}")
        print(f"      Range: {float(integrated_npp.min().values):.2f} to {float(integrated_npp.max().values):.2f}")
        
    except Exception as e:
        print(f"   ❌ Integration failed: {e}")
        integrated_npp = None


# --- STEP 2: PROCESS NITRATE (NO3) ---
print("\n" + "─"*70)
print("2. PROCESSING NUTRIENTS (NO3)")
print("─"*70)
no3_subset, ds_nut = extract_and_process(files['nut'], 'no3', 'Nitrate')

surf_no3 = None
deep_no3 = None

if no3_subset is not None:
    try:
        # Get surface depth (closest to 0)
        min_depth = float(no3_subset.depth.min().values)
        max_depth = float(no3_subset.depth.max().values)
        
        # Surface (nearest to minimum depth)
        surf_no3 = no3_subset.sel(depth=min_depth, method='nearest').mean(
            dim=['latitude', 'longitude'], skipna=True)
        
        # Deep (100m or maximum available)
        deep_depth = min(100, max_depth)
        deep_no3 = no3_subset.sel(depth=deep_depth, method='nearest').mean(
            dim=['latitude', 'longitude'], skipna=True)
        
        print(f"   ✅ Extracted nitrate profiles")
        print(f"      Surface depth: {min_depth:.1f}m")
        print(f"      Deep depth: {deep_depth:.1f}m")
        
    except Exception as e:
        print(f"   ❌ Nitrate extraction failed: {e}")


# --- STEP 3: PROCESS OXYGEN (O2) ---
print("\n" + "─"*70)
print("3. PROCESSING OXYGEN (O2)")
print("─"*70)
o2_subset, _ = extract_and_process(files['prod'], 'o2', 'Oxygen')

deep_o2 = None
if o2_subset is not None:
    try:
        # Get oxygen at 50m (typical fish habitat depth)
        available_depth = min(50, float(o2_subset.depth.max().values))
        deep_o2 = o2_subset.sel(depth=available_depth, method='nearest').mean(
            dim=['latitude', 'longitude'], skipna=True)
        
        print(f"   ✅ Extracted oxygen at {available_depth:.0f}m depth")
        print(f"      Min: {float(deep_o2.min().values):.2f} mmol/m³")
        print(f"      Mean: {float(deep_o2.mean().values):.2f} mmol/m³")
        
    except Exception as e:
        print(f"   ❌ Oxygen extraction failed: {e}")


# --- STEP 4: PLOTTING ---
print("\n" + "─"*70)
print("GENERATING VISUALIZATION")
print("─"*70)

if integrated_npp is None and surf_no3 is None and deep_o2 is None:
    print("❌ CRITICAL: No data extracted. Cannot generate plot.")
    print("   Check your lat/lon bounds or data files.")
    exit()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Plot 1: NPP
if integrated_npp is not None:
    ax1.plot(integrated_npp.time, integrated_npp, color='darkgreen', 
             linewidth=2.5, label='Integrated NPP (0-100m)')
    ax1.axhline(y=integrated_npp.mean(), color='gray', linestyle='--', 
                alpha=0.5, label='Annual Mean')
    ax1.set_title("Water Column Productivity (Integrated NPP)", fontweight='bold', fontsize=13)
    ax1.set_ylabel("NPP (mg C m⁻² day⁻¹)", fontsize=11)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
else:
    ax1.text(0.5, 0.5, "⚠️  NO NPP DATA\n(Check land mask or try different coordinates)", 
             ha='center', va='center', transform=ax1.transAxes, fontsize=12, color='red')
    ax1.set_title("Water Column Productivity - DATA UNAVAILABLE", fontweight='bold')

# Plot 2: Nitrate
if surf_no3 is not None and deep_no3 is not None:
    ax2.plot(deep_no3.time, deep_no3, color='navy', linewidth=2.5, 
             linestyle='--', label='Deep NO₃ (100m)')
    ax2.plot(surf_no3.time, surf_no3, color='cyan', linewidth=2.5, 
             label='Surface NO₃')
    ax2.fill_between(deep_no3.time, surf_no3, deep_no3, alpha=0.2, 
                      color='blue', label='Upwelling Potential')
    ax2.set_title("Upwelling Diagnostics (Nutrient Gradient)", fontweight='bold', fontsize=13)
    ax2.set_ylabel("Nitrate (mmol m⁻³)", fontsize=11)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
else:
    ax2.text(0.5, 0.5, "⚠️  NO NITRATE DATA", ha='center', va='center', 
             transform=ax2.transAxes, fontsize=12, color='red')
    ax2.set_title("Upwelling Diagnostics - DATA UNAVAILABLE", fontweight='bold')

# Plot 3: Oxygen
if deep_o2 is not None:
    ax3.plot(deep_o2.time, deep_o2, color='crimson', linewidth=2.5, 
             label='O₂ at 50m')
    ax3.axhline(y=60, color='darkred', linestyle=':', linewidth=2, 
                label='Hypoxia Threshold')
    ax3.axhline(y=120, color='orange', linestyle=':', linewidth=1.5, 
                label='Stress Level')
    ax3.fill_between(deep_o2.time, 0, deep_o2, where=(deep_o2 < 60), 
                      alpha=0.3, color='red')
    ax3.set_title("Habitat Oxygen Levels (50m Depth)", fontweight='bold', fontsize=13)
    ax3.set_ylabel("Dissolved O₂ (mmol m⁻³)", fontsize=11)
    ax3.set_xlabel("Date", fontsize=11)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
else:
    ax3.text(0.5, 0.5, "⚠️  NO OXYGEN DATA", ha='center', va='center', 
             transform=ax3.transAxes, fontsize=12, color='red')
    ax3.set_title("Habitat Oxygen - DATA UNAVAILABLE", fontweight='bold')

plt.tight_layout()
plt.savefig('ratnagiri_biogeochem_robust.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved: ratnagiri_biogeochem_robust.png")
plt.show()


# --- STEP 5: QUANTITATIVE REPORT ---
print("\n" + "="*70)
print("BIOGEOCHEMICAL ANALYSIS REPORT - RATNAGIRI ZONE (2025)")
print("="*70)

# 1. Productivity Analysis
if integrated_npp is not None:
    print("\n" + "─"*70)
    print("1. PRIMARY PRODUCTION ASSESSMENT")
    print("─"*70)
    
    try:
        bloom = integrated_npp.sel(time=slice('2025-10-01', '2025-11-30')).mean().values
        annual = integrated_npp.mean().values
        anomaly = ((bloom - annual) / annual) * 100
        
        print(f"   Bloom Period (Oct-Nov):  {bloom:.2f} mg C m⁻² day⁻¹")
        print(f"   Annual Mean:             {annual:.2f} mg C m⁻² day⁻¹")
        print(f"   Bloom Anomaly:           {anomaly:+.1f}%")
        
        if anomaly > 20:
            print("\n   ✅ STATUS: STRONG BLOOM")
            print("      → Favorable for fish recruitment")
        elif anomaly > 0:
            print("\n   ℹ️  STATUS: NORMAL BLOOM")
        else:
            print("\n   ⚠️  STATUS: BLOOM FAILURE")
            print("      → Below-average productivity")
            
    except Exception as e:
        print(f"   ⚠️  Could not analyze bloom period: {e}")
else:
    print("\n1. PRIMARY PRODUCTION: No data available")

# 2. Nutrient Analysis
if surf_no3 is not None and deep_no3 is not None:
    print("\n" + "─"*70)
    print("2. UPWELLING & NUTRIENT STATUS")
    print("─"*70)
    
    try:
        bloom_surf = surf_no3.sel(time=slice('2025-10-01', '2025-11-30')).mean().values
        bloom_deep = deep_no3.sel(time=slice('2025-10-01', '2025-11-30')).mean().values
        gradient = bloom_deep - bloom_surf
        
        print(f"   Surface NO₃ (Bloom):     {bloom_surf:.3f} mmol m⁻³")
        print(f"   Deep NO₃ (Bloom):        {bloom_deep:.3f} mmol m⁻³")
        print(f"   Upwelling Gradient:      {gradient:.3f} mmol m⁻³")
        
        if bloom_surf < 0.5 and gradient > 10:
            print("\n   ✅ DIAGNOSIS: Efficient nutrient utilization")
            print("      → Strong upwelling with active consumption")
        elif gradient > 15:
            print("\n   💪 DIAGNOSIS: Very strong upwelling")
        elif bloom_surf < 0.5:
            print("\n   ⚠️  DIAGNOSIS: Nutrient limitation")
            
    except Exception as e:
        print(f"   ⚠️  Could not analyze nutrients: {e}")
else:
    print("\n2. UPWELLING STATUS: No data available")

# 3. Oxygen Analysis
if deep_o2 is not None:
    print("\n" + "─"*70)
    print("3. HABITAT OXYGEN ASSESSMENT")
    print("─"*70)
    
    min_o2 = float(deep_o2.min().values)
    mean_o2 = float(deep_o2.mean().values)
    
    print(f"   Minimum O₂:              {min_o2:.2f} mmol m⁻³")
    print(f"   Mean O₂:                 {mean_o2:.2f} mmol m⁻³")
    
    if min_o2 < 60:
        print("\n   🚨 ALERT: Hypoxic events detected")
        print("      → Fish mortality/displacement likely")
    elif min_o2 < 120:
        print("\n   ⚠️  CAUTION: Oxygen stress periods")
        print("      → Sub-optimal for sensitive species")
    else:
        print("\n   ✅ OXYGEN: Healthy levels throughout")
else:
    print("\n3. OXYGEN STATUS: No data available")

print("\n" + "="*70 + "\n")
