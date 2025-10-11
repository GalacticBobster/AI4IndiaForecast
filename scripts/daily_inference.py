from dotenv import load_dotenv

load_dotenv()


from datetime import datetime
from earth2studio.data import GFS
from earth2studio.data import ARCO
from earth2studio.models.px import DLWP
from earth2studio.io import ZarrBackend
import earth2studio.run as run
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import os
import zarr


def round_to_gfs(dt):
    hour = (dt.hour // 6) * 6
    day = dt.day - 1
    return dt.replace(day=day, hour=hour, minute=0, second=0, microsecond=0)

def main():
    nsteps = 10
    current_time = datetime.utcnow()
    gfs_time = round_to_gfs(current_time)
    print(f"Running inference for {gfs_time}")
    package = DLWP.load_default_package()
    model = DLWP.load_model(package)
  
    # Create the data source
    gfs = GFS()
    variable = "t2m"
    # Create the IO handler, store in memory
    io = ZarrBackend()
    # Example Earth2Studio workflow
    io = run.deterministic([gfs_time], nsteps, model, gfs, io)
    # Create a figure and axes with the specified projection
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
    print(z.tree())

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
    plt.savefig(os.path.join(output_dir, "t2m_forecast.png"))
    
    variable = "tcwv"
    fig2, ax2 = plt.subplots(
    1,
    5,
    figsize=(12, 4),
    subplot_kw={"projection": ccrs.PlateCarree()},
    constrained_layout=True,
    )
    vmin = np.nanmin(io[variable])
    vmax = np.nanmax(io[variable])
    times = (io["lead_time"][:].astype("timedelta64[ns]").astype("timedelta64[h]").astype(int))
    z = zarr.open(io.store, mode="r")
    print(z.tree())

    step = 2  # 24hrs
    for i, t in enumerate(range(0, 10, step)):
      ctr = ax2[i].contourf(
        io["lon"][:],
        io["lat"][:],
        io[variable][0, t],
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        levels=20,
        cmap="Spectral_r",
      )
      ax2[i].set_title(f"{times[t]}hrs")
      ax2[i].coastlines()
      ax2[i].gridlines()
      ax2[i].set_extent([65, 100, 5, 40], crs=ccrs.PlateCarree())

    plt.suptitle(f"{variable} - {gfs_time}")

    cbar = plt.cm.ScalarMappable(cmap="Spectral_r")
    cbar.set_array(io[variable][0, 0])
    #cbar.set_clim(-10.0, 30)
    cbar = fig2.colorbar(cbar, ax=ax2[-1], orientation="vertical",  shrink=0.8)
    output_dir = "../docs/outputs"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "tcwv_forecast.png"))


if __name__ == "__main__":
    main()
