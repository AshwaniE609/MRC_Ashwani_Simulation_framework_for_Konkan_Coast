import copernicusmarine

copernicusmarine.subset(
  dataset_id="cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m",
  variables=["chl", "phyc"],
  minimum_longitude=71,
  maximum_longitude=74,
  minimum_latitude=15,
  maximum_latitude=20.5,
  start_datetime="2025-01-01T00:00:00",
  end_datetime="2025-01-12T00:00:00",
  minimum_depth=0.4940253794193268,
  maximum_depth=109.72927856445312,
    output_directory="./data",
    output_filename="konkan_bgc_chl_phyc.nc"
)
