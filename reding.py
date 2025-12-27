import xarray as xr
import matplotlib.pyplot as plt

# 1. Load the file
file_path = 'AQUA_MODIS.20250919.L3m.DAY.CHL.x_chlor_a.nc'
ds = xr.open_dataset(file_path)

# 2. Select the Chlorophyll variable
chlor_a = ds['chlor_a']

# 3. Quick Plot to see the Konkan coast
plt.figure(figsize=(10, 6))
chlor_a.plot(vmin=0, vmax=5, cmap='jet') # vmin/vmax sets the color scale
plt.title("Konkan Coast Chlorophyll-a")
plt.show()

# 4. To get a specific reading (e.g., near Mumbai/Ratnagiri):
# You can select by latitude/longitude slice
subset = chlor_a.sel(lat=slice(17.5, 16.5), lon=slice(72.0, 73.0))
print(subset.mean().values) # Prints average Chl-a for that area