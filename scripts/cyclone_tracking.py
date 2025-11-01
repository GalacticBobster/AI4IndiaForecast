from dotenv import load_dotenv

load_dotenv()


from datetime import datetime, timedelta
from earth2studio.data import GFS
from earth2studio.data import ARCO
from earth2studio.models.px import DLWP
from earth2studio.io import ZarrBackend
from earth2studio.models.dx.tc_tracking import TCTrackerWuDuan
from earth2studio.data import fetch_data, prep_data_array
from earth2studio.utils.time import to_time_array
from earth2studio.utils.coords import map_coords
import earth2studio.run as run
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import os
import zarr
import torch
from tqdm import tqdm

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
    start_time = get_last_available_gfs_time(current_time)
    print(f"Running inference for {start_time}")
    
    package = DLWP.load_default_package()
    prognostic = DLWP.load_model(package)
    #package = DLWP.load_default_package()
    #model = DLWP.load_model(package)
    tracker = TCTrackerWuDuan()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tracker = tracker.to(device)
    prognostic = prognostic.to(device)
    # Create the data source
    data = GFS()
    variable = "t2m"
    # Create the IO handler, store in memory
    io = ZarrBackend()
    
    model_vars = prognostic.input_coords()["variable"]
    tracker_vars = tracker.input_coords()
    all_vars = list(set(model_vars) | set(tracker_vars))  # union

    x, coords = fetch_data(
     source=data,
     time=to_time_array([start_time]),
     variable=all_vars,
     lead_time=prognostic.input_coords()["lead_time"],  # or model lead_time if same
     device=device,
    )

    model = prognostic.create_iterator(x, coords)
    with tqdm(total=nsteps + 1, desc="Running inference") as pbar:
     for step, (x, coords) in enumerate(model):
        # Run tracker
        x, coords = map_coords(x, coords, tracker.input_coords())
        output, output_coords = tracker(x, coords)
        # lets remove the lead time dim
        output = output[:, 0]
        print(f"Step {step}: SFNO tracks output shape {output.shape}")

        pbar.update(1)
        if step == nsteps:
            break

    # Example Earth2Studio workflow
    #io = run.deterministic([gfs_time], nsteps, model, gfs, io)
       


if __name__ == "__main__":
    main()
