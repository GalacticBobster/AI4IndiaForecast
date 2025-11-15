from dotenv import load_dotenv

load_dotenv()


from datetime import datetime, timedelta
from earth2studio.data import GFS
from earth2studio.data import ARCO
from earth2studio.models.px import Aurora
from earth2studio.io import ZarrBackend
import earth2studio.run as run
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import os
import zarr



def round_to_gfs(dt):
    # Round to the nearest past 6-hour GFS cycle
    hour = (dt.hour // 6) * 6
    rounded = dt.replace(hour=hour, minute=0, second=0, microsecond=0)

    # If dt is before the rounded hour (e.g., dt=00:03, rounded=00:00), subtract a day if needed
    if rounded > dt:
        rounded -= timedelta(days=1)

    return rounded

def get_last_available_gfs_time(dt=None):
    """Return a GFS datetime that is guaranteed to exist."""
    if dt is None:
        dt = datetime.utcnow()

    gfs_time = round_to_gfs(dt)

    # Check against the last known available GFS forecast
    # For example, assume forecasts are available only up to yesterday's 18 UTC run
    now = datetime.utcnow()
    last_available = round_to_gfs(now - timedelta(days=1))

    if gfs_time > last_available:
        gfs_time = last_available

    return gfs_time

def main():
    nsteps = 10
    current_time = datetime.utcnow()
    gfs_time = get_last_available_gfs_time(current_time)
    print(f"Running inference for {gfs_time}")
    package = Aurora.load_default_package()
    model = Aurora.load_model(package)
    print('model loaded')
    # Create the data source
    gfs = GFS()
    variable = "t2m"
    # Create the IO handler, store in memory
    io = ZarrBackend()
    # Example Earth2Studio workflow
    print('running inferences')
    io = run.deterministic([gfs_time], nsteps, model, gfs, io)


if __name__ == "__main__":
    main()
