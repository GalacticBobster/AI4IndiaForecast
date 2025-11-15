import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from datetime import datetime, timedelta

import torch

from earth2studio.data import ARCO
from earth2studio.models.dx import TCTrackerWuDuan
from earth2studio.models.px import SFNO
from earth2studio.utils.time import to_time_array

# Create tropical cyclone tracker
tracker = TCTrackerWuDuan()

# Load the default model package which downloads the check point from NGC
package = SFNO.load_default_package()
prognostic = SFNO.load_model(package)

# Create the data source
data = ARCO()

nsteps = 16  # Number of steps to run the tracker for into future
start_time = datetime(2009, 8, 5)  # Start date for inference


from earth2studio.data import fetch_data, prep_data_array

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") 
tracker = tracker.to(device)

# Land fall occured August 25th 2017
times = [start_time + timedelta(hours=6 * i) for i in range(nsteps + 1)]
for step, time in enumerate(times):
    da = data(time, tracker.input_coords()["variable"])
    x, coords = prep_data_array(da, device=device)
    output, output_coords = tracker(x, coords)
    print(f"Step {step}: ARCO tracks output shape {output.shape}")
    torch.save(output.cpu(), f"outputs/era5_step_{step}.pt")

era5_tracks = output.cpu()
torch.save(era5_tracks, "outputs/13_era5_paths.pt")

from tqdm import tqdm

from earth2studio.utils.coords import map_coords

prognostic = prognostic.to(device)
# Reset the internal path buffer of tracker
tracker.reset_path_buffer()

# Load the initial state
x, coords = fetch_data(
    source=data,
    time=to_time_array([start_time]),
    variable=prognostic.input_coords()["variable"],
    lead_time=prognostic.input_coords()["lead_time"],
    device=device,
)

# Create prognostic iterator
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

sfno_tracks = output.cpu()
torch.save(sfno_tracks, "outputs/13_sfno_paths.pt")

