from dotenv import load_dotenv

load_dotenv()


from datetime import datetime
from earth2studio.data import GFS
from earth2studio.models.px import DLWP
from earth2studio.io import ZarrBackend
import earth2studio.run as run
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

def round_to_gfs(dt):
    hour = (dt.hour // 6) * 6
    day = dt.day - 1
    return dt.replace(day=day, hour=hour, minute=0, second=0, microsecond=0)

def main():
    nsteps = 20
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
    subplot_kw={"projection": ccrs.Orthographic()},
    constrained_layout=True,
    )
    vmin = np.nanmin(io[variable])
    vmax = np.nanmax(io[variable])
    times = (io["lead_time"][:].astype("timedelta64[ns]").astype("timedelta64[h]").astype(int))
    step = 4  # 24hrs
    for i, t in enumerate(range(0, 20, step)):
      ctr = ax[i].contourf(
        io["lon"][:],
        io["lat"][:],
        io[variable][0, t],
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        levels=20,
        cmap="coolwarm",
      )
      ax[i].set_title(f"{times[t]}hrs")
      ax[i].coastlines()
      ax[i].gridlines()

    plt.suptitle(f"{variable} - {gfs_time}")

    cbar = plt.cm.ScalarMappable(cmap="coolwarm")
    cbar.set_array(io[variable][0, 0])
    #cbar.set_clim(-10.0, 30)
    cbar = fig.colorbar(cbar, ax=ax[-1], orientation="vertical", label="K", shrink=0.8)


    plt.savefig("../docs/outputs/latest_forecast.png")


if __name__ == "__main__":
    main()
