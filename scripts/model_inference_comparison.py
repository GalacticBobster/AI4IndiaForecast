from dotenv import load_dotenv

load_dotenv()


from datetime import datetime
from earth2studio.data import GFS
from earth2studio.data import ARCO
from earth2studio.models.px import DLWP
from earth2studio.models.px import GraphCastSmall
from earth2studio.models.px import Pangu6
from earth2studio.io import ZarrBackend
import earth2studio.run as run
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import os
import zarr

def main():
   type_data = 'ERA5'
   modeltype = 'GraphCastSmall'
   if(type_data == 'ERA5'):
     data = ARCO()
   elif(type_data == 'GFS'):
     data = GFS()

   nsteps = 10
   current_time = datetime(2022, 1, 1, 18)
  
   if(modeltype == 'DLWP'):
    package = DLWP.load_default_package()
    model = DLWP.load_model(package)
   elif(modeltype == 'Pangu'):
    package = Pangu6.load_default_package()
    model = Pangu6.load_model(package)
   elif(modeltype == 'GraphCastSmall'):
    package = GraphCastSmall.load_default_package()
    model = GraphCastSmall.load_model(package)
   io = ZarrBackend()
   io = run.deterministic([current_time], nsteps, model, data, io)

   variable = "t2m"
   fig, ax = plt.subplots(
    1,
    5,
    figsize=(12, 4),
    subplot_kw={"projection": ccrs.PlateCarree()},
    constrained_layout=True,
    )
   vmin = np.nanmin(io[variable]) - 273.15
   vmax = np.nanmax(io[variable]) - 273.15
   times = (io["lead_time"][:].astype("timedelta64[ns]").astype("timedelta64[h]").astype(int))
   z = zarr.open(io.store, mode="r")
  
   step = 2  # 24hrs
   for i, t in enumerate(range(0, 10, step)):
      ctr = ax[i].contourf(
        io["lon"][:],
        io["lat"][:],
        io[variable][0, t] - 273.15,
        vmin=-10,
        vmax=40,
        transform=ccrs.PlateCarree(),
        levels=20,
        cmap="coolwarm",
      )
      ax[i].set_title(f"{times[t]}hrs")
      ax[i].coastlines()
      ax[i].gridlines()
      ax[i].set_extent([65, 100, 5, 40], crs=ccrs.PlateCarree())

   plt.suptitle(f"{variable} - {gfs_time}")

   cbar = plt.cm.ScalarMappable(cmap="coolwarm")
   cbar.set_array(io[variable][0, 0] - 273.15)
   cbar.set_clim(-10.0, 40)
   cbar = fig.colorbar(cbar, ax=ax[-1], orientation="vertical", label="$^{o}$C", shrink=0.8)

   output_dir = "../docs/outputs"
   os.makedirs(output_dir, exist_ok=True)
   plt.savefig(os.path.join(output_dir, f"t2m_{data}_{modeltype}.png"))

if __name__ == "__main__":
    main()



