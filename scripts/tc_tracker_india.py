"""
Tropical Cyclone Tracker for Indian Subcontinent
Tracks multiple tropical cyclones around the Indian Ocean region using GFS/ERA5 data
and Earth2studio models.
"""
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import os
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from tqdm import tqdm

from earth2studio.data import GFS, ARCO
from earth2studio.models.dx import TCTrackerWuDuan
from earth2studio.models.px import SFNO, DLWP
from earth2studio.data import fetch_data, prep_data_array
from earth2studio.utils.time import to_time_array
from earth2studio.utils.coords import map_coords


# Indian Ocean region boundaries
INDIAN_OCEAN_REGION = {
    'lat_min': 0.0,
    'lat_max': 30.0,
    'lon_min': 40.0,
    'lon_max': 100.0
}


def round_to_6h(dt: datetime) -> datetime:
    """Round datetime to nearest past 6-hour cycle."""
    hour = (dt.hour // 6) * 6
    rounded = dt.replace(hour=hour, minute=0, second=0, microsecond=0)
    if rounded > dt:
        rounded -= timedelta(hours=6)
    return rounded


def get_last_available_time(dt: Optional[datetime] = None, source: str = "GFS") -> datetime:
    """
    Return a datetime that is guaranteed to exist for the specified data source.
    
    Args:
        dt: Reference datetime (default: current UTC time)
        source: Data source type ("GFS" or "ERA5")
    
    Returns:
        Available datetime for the data source
    """
    if dt is None:
        dt = datetime.utcnow()
    
    rounded_time = round_to_6h(dt)
    
    if source.upper() == "GFS":
        # GFS typically has 6-hour delay
        last_available = round_to_6h(datetime.utcnow() - timedelta(hours=12))
    else:  # ERA5
        # ERA5 has longer delay (typically 5 days)
        last_available = round_to_6h(datetime.utcnow() - timedelta(days=5))
    
    if rounded_time > last_available:
        rounded_time = last_available
    
    return rounded_time


class TropicalCycloneTracker:
    """Tracker for multiple tropical cyclones in the Indian Ocean region."""
    
    def __init__(
        self,
        data_source: str = "GFS",
        model_name: str = "SFNO",
        device: Optional[str] = None
    ):
        """
        Initialize the tracker.
        
        Args:
            data_source: "GFS" or "ERA5"
            model_name: "SFNO" or "DLWP"
            device: Device to use (default: auto-detect)
        """
        self.data_source = data_source.upper()
        self.model_name = model_name.upper()
        
        # Set up device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Initializing Tropical Cyclone Tracker")
        print(f"  Data Source: {self.data_source}")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        
        # Initialize tracker
        self.tracker = TCTrackerWuDuan().to(self.device)
        
        # Initialize prognostic model
        if self.model_name == "SFNO":
            package = SFNO.load_default_package()
            self.prognostic = SFNO.load_model(package)
        elif self.model_name == "DLWP":
            package = DLWP.load_default_package()
            self.prognostic = DLWP.load_model(package)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        self.prognostic = self.prognostic.to(self.device)
        
        # Initialize data source
        if self.data_source == "GFS":
            self.data = GFS()
        elif self.data_source == "ERA5":
            self.data = ARCO()
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")
        
        self.tracks = []
        self.output_dir = None
    
    def track_cyclones(
        self,
        start_time: Optional[datetime] = None,
        nsteps: int = 20,
        output_dir: str = "outputs/cyclone_tracks"
    ) -> Dict:
        """
        Track tropical cyclones from the specified start time.
        
        Args:
            start_time: Start time for tracking (default: latest available)
            nsteps: Number of forecast steps
            output_dir: Directory to save outputs
        
        Returns:
            Dictionary containing track information
        """
        if start_time is None:
            start_time = get_last_available_time(source=self.data_source)
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nTracking cyclones from {start_time}")
        print(f"Forecast steps: {nsteps}")
        print(f"Output directory: {output_dir}")
        
        # Reset tracker's internal state
        self.tracker.reset_path_buffer()
        
        # Fetch initial data
        x, coords = fetch_data(
            source=self.data,
            time=to_time_array([start_time]),
            variable=self.prognostic.input_coords()["variable"],
            lead_time=self.prognostic.input_coords()["lead_time"],
            device=self.device,
        )
        
        # Create prognostic iterator
        model_iterator = self.prognostic.create_iterator(x, coords)
        
        # Storage for tracks
        all_tracks = []
        all_coords = []
        
        # Run tracking
        with tqdm(total=nsteps + 1, desc="Running cyclone tracking") as pbar:
            for step, (x, coords) in enumerate(model_iterator):
                # Map coordinates for tracker
                x_track, coords_track = map_coords(x, coords, self.tracker.input_coords())
                
                # Run tracker
                output, output_coords = self.tracker(x_track, coords_track)
                
                # Remove lead time dimension
                output = output[:, 0]
                
                # Store results
                all_tracks.append(output.cpu())
                all_coords.append(output_coords)
                
                # Save intermediate results
                if step % 5 == 0:
                    torch.save(output.cpu(), f"{output_dir}/tracks_step_{step:03d}.pt")
                
                pbar.update(1)
                if step == nsteps:
                    break
        
        # Save final tracks
        final_tracks = torch.stack(all_tracks)
        torch.save(final_tracks, f"{output_dir}/tracks_final.pt")
        
        # Extract and save cyclone information
        cyclone_info = self._extract_cyclone_info(final_tracks, start_time)
        with open(f"{output_dir}/cyclone_info.json", "w") as f:
            json.dump(cyclone_info, f, indent=2, default=str)
        
        print(f"\nTracking complete. Found {cyclone_info['num_cyclones']} cyclone(s)")
        
        return {
            'tracks': final_tracks,
            'info': cyclone_info,
            'start_time': start_time,
            'nsteps': nsteps
        }
    
    def _extract_cyclone_info(self, tracks: torch.Tensor, start_time: datetime) -> Dict:
        """Extract cyclone information from tracking results."""
        # tracks shape: [time_steps, batch, num_detections, 4]
        # 4 channels: lat, lon, wind_speed, pressure
        
        num_steps = tracks.shape[0]
        cyclone_list = []
        
        # Analyze tracks to identify distinct cyclones
        for step in range(num_steps):
            step_data = tracks[step, 0]  # [num_detections, 4]
            
            for detection_idx in range(step_data.shape[0]):
                detection = step_data[detection_idx]
                lat, lon = detection[0].item(), detection[1].item()
                
                # Filter for Indian Ocean region
                if (INDIAN_OCEAN_REGION['lat_min'] <= lat <= INDIAN_OCEAN_REGION['lat_max'] and
                    INDIAN_OCEAN_REGION['lon_min'] <= lon <= INDIAN_OCEAN_REGION['lon_max']):
                    
                    cyclone_list.append({
                        'step': step,
                        'time': start_time + timedelta(hours=6*step),
                        'lat': lat,
                        'lon': lon,
                        'wind_speed': detection[2].item() if detection.shape[0] > 2 else None,
                        'pressure': detection[3].item() if detection.shape[0] > 3 else None
                    })
        
        return {
            'num_cyclones': len(cyclone_list),
            'cyclones': cyclone_list,
            'region': INDIAN_OCEAN_REGION,
            'data_source': self.data_source,
            'model': self.model_name
        }
    
    def visualize_tracks(
        self,
        tracks: torch.Tensor,
        start_time: datetime,
        save_path: Optional[str] = None
    ):
        """
        Visualize cyclone tracks on a map.
        
        Args:
            tracks: Tensor of cyclone tracks
            start_time: Start time of tracking
            save_path: Path to save the figure
        """
        if save_path is None and self.output_dir:
            save_path = f"{self.output_dir}/cyclone_tracks.png"
        
        fig = plt.figure(figsize=(14, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Set extent to Indian Ocean region
        ax.set_extent([
            INDIAN_OCEAN_REGION['lon_min'],
            INDIAN_OCEAN_REGION['lon_max'],
            INDIAN_OCEAN_REGION['lat_min'],
            INDIAN_OCEAN_REGION['lat_max']
        ], crs=ccrs.PlateCarree())
        
        # Add map features
        ax.coastlines(resolution='50m')
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        ax.stock_img()
        
        # Plot tracks
        num_steps = tracks.shape[0]
        colors = plt.cm.rainbow(np.linspace(0, 1, num_steps))
        
        for step in range(num_steps):
            step_data = tracks[step, 0]  # [num_detections, 4]
            
            for detection_idx in range(step_data.shape[0]):
                detection = step_data[detection_idx]
                lat, lon = detection[0].item(), detection[1].item()
                
                # Filter for Indian Ocean region
                if (INDIAN_OCEAN_REGION['lat_min'] <= lat <= INDIAN_OCEAN_REGION['lat_max'] and
                    INDIAN_OCEAN_REGION['lon_min'] <= lon <= INDIAN_OCEAN_REGION['lon_max']):
                    
                    ax.plot(lon, lat, 'o', color=colors[step], 
                           markersize=8, transform=ccrs.PlateCarree(),
                           alpha=0.7)
        
        # Add title
        end_time = start_time + timedelta(hours=6*num_steps)
        plt.title(
            f'Tropical Cyclone Tracks - Indian Ocean\n'
            f'{self.model_name} Model, {self.data_source} Data\n'
            f'{start_time.strftime("%Y-%m-%d %H:%M")} to {end_time.strftime("%Y-%m-%d %H:%M")} UTC',
            fontsize=14, fontweight='bold'
        )
        
        # Add colorbar for time
        sm = plt.cm.ScalarMappable(
            cmap='rainbow',
            norm=plt.Normalize(vmin=0, vmax=num_steps*6)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
        cbar.set_label('Forecast Hour', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Track visualization saved to {save_path}")
        
        plt.close()


def main():
    """Main function to run tropical cyclone tracking."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Track tropical cyclones in the Indian Ocean region'
    )
    parser.add_argument(
        '--data-source',
        type=str,
        default='GFS',
        choices=['GFS', 'ERA5'],
        help='Data source (default: GFS)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='SFNO',
        choices=['SFNO', 'DLWP'],
        help='Prognostic model (default: SFNO)'
    )
    parser.add_argument(
        '--nsteps',
        type=int,
        default=20,
        help='Number of forecast steps (default: 20)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/cyclone_tracks',
        help='Output directory (default: outputs/cyclone_tracks)'
    )
    parser.add_argument(
        '--start-time',
        type=str,
        default=None,
        help='Start time in YYYY-MM-DD-HH format (default: latest available)'
    )
    
    args = parser.parse_args()
    
    # Parse start time if provided
    start_time = None
    if args.start_time:
        try:
            start_time = datetime.strptime(args.start_time, '%Y-%m-%d-%H')
        except ValueError:
            print(f"Invalid start time format: {args.start_time}")
            print("Expected format: YYYY-MM-DD-HH")
            return
    
    # Initialize tracker
    tracker = TropicalCycloneTracker(
        data_source=args.data_source,
        model_name=args.model
    )
    
    # Run tracking
    results = tracker.track_cyclones(
        start_time=start_time,
        nsteps=args.nsteps,
        output_dir=args.output_dir
    )
    
    # Visualize results
    tracker.visualize_tracks(
        results['tracks'],
        results['start_time']
    )
    
    print("\nTracking complete!")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
