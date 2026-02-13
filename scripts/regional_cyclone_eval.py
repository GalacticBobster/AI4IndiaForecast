"""
Regional Cyclone Tracking and Forecasting
Combines tropical cyclone tracking with limited area modeling for the Indian subcontinent.
Evaluates multiple cyclones using GFS/ERA5 data and Earth2studio models.
"""
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm

from earth2studio.data import GFS, ARCO
from earth2studio.models.dx import TCTrackerWuDuan
from earth2studio.models.px import SFNO, DLWP, FCN
from earth2studio.data import fetch_data
from earth2studio.utils.time import to_time_array
from earth2studio.utils.coords import map_coords

from regional_model_utils import (
    RegionalModelConfig,
    get_regional_grid_info,
    REGIONAL_DOMAINS
)
from cyclone_utils import round_to_6h, get_last_available_time


class RegionalCycloneForecaster:
    """
    Regional forecasting system for tropical cyclones over Indian subcontinent.
    Combines prognostic models with cyclone tracking in a limited area setup.
    """
    
    def __init__(
        self,
        region: str = 'north_indian_ocean',
        data_source: str = 'GFS',
        model_name: str = 'SFNO',
        device: Optional[str] = None
    ):
        """
        Initialize the regional cyclone forecaster.
        
        Args:
            region: Regional domain (see REGIONAL_DOMAINS)
            data_source: "GFS" or "ERA5"
            model_name: "SFNO", "DLWP", or "FCN"
            device: Device to use (default: auto-detect)
        """
        self.region_config = RegionalModelConfig(region)
        self.data_source = data_source.upper()
        self.model_name = model_name.upper()
        
        # Set up device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Regional Cyclone Forecaster Initialization")
        print(f"  Region: {self.region_config.name}")
        print(f"  Bounds: {self.region_config.get_bounds()}")
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
        elif self.model_name == "FCN":
            package = FCN.load_default_package()
            self.prognostic = FCN.load_model(package)
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
    
    def evaluate_cyclones(
        self,
        cyclone_list: List[Dict],
        start_time: datetime,
        nsteps: int = 20,
        output_dir: str = "outputs/regional_cyclones"
    ) -> Dict:
        """
        Evaluate a list of tropical cyclones with regional forecasting.
        
        Args:
            cyclone_list: List of cyclone information dicts with 'name', 'lat', 'lon'
            start_time: Start time for evaluation
            nsteps: Number of forecast steps
            output_dir: Output directory
        
        Returns:
            Dictionary with evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nEvaluating {len(cyclone_list)} cyclone(s)")
        print(f"Start time: {start_time}")
        print(f"Forecast steps: {nsteps} (every 6 hours)")
        
        # Filter cyclones in the regional domain
        regional_cyclones = []
        for cyclone in cyclone_list:
            if self.region_config.contains_point(cyclone['lat'], cyclone['lon']):
                regional_cyclones.append(cyclone)
                print(f"  ✓ {cyclone['name']}: ({cyclone['lat']}°N, {cyclone['lon']}°E) - In domain")
            else:
                print(f"  ✗ {cyclone['name']}: ({cyclone['lat']}°N, {cyclone['lon']}°E) - Outside domain")
        
        if not regional_cyclones:
            print("\nNo cyclones in the regional domain!")
            return {'status': 'no_cyclones', 'cyclones': []}
        
        print(f"\nTracking {len(regional_cyclones)} cyclone(s) in {self.region_config.name}")
        
        # Run forecast and tracking
        results = self._run_regional_forecast(
            start_time, nsteps, output_dir
        )
        
        # Evaluate each cyclone
        cyclone_evaluations = []
        for cyclone in regional_cyclones:
            evaluation = self._evaluate_single_cyclone(
                cyclone, results, start_time, output_dir
            )
            cyclone_evaluations.append(evaluation)
        
        # Create summary
        summary = {
            'status': 'success',
            'region': self.region_config.name,
            'region_bounds': self.region_config.get_bounds(),
            'start_time': start_time.isoformat(),
            'nsteps': nsteps,
            'data_source': self.data_source,
            'model': self.model_name,
            'total_cyclones': len(cyclone_list),
            'regional_cyclones': len(regional_cyclones),
            'cyclone_evaluations': cyclone_evaluations
        }
        
        # Save summary
        with open(f"{output_dir}/evaluation_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nEvaluation complete. Results saved to {output_dir}")
        
        return summary
    
    def _run_regional_forecast(
        self,
        start_time: datetime,
        nsteps: int,
        output_dir: str
    ) -> Dict:
        """Run regional forecast with tracking."""
        print("\nRunning regional forecast...")
        
        # Reset tracker
        self.tracker.reset_path_buffer()
        
        # Fetch initial data
        x, coords = fetch_data(
            source=self.data,
            time=to_time_array([start_time]),
            variable=self.prognostic.input_coords()["variable"],
            lead_time=self.prognostic.input_coords()["lead_time"],
            device=self.device,
        )
        
        # Storage for results
        all_tracks = []
        forecast_times = []
        
        # Create prognostic iterator
        model_iterator = self.prognostic.create_iterator(x, coords)
        
        # Run forecast with tracking
        with tqdm(total=nsteps + 1, desc="Running regional forecast") as pbar:
            for step, (x, coords) in enumerate(model_iterator):
                current_time = start_time + timedelta(hours=6*step)
                forecast_times.append(current_time)
                
                # Run tracker on forecast
                x_track, coords_track = map_coords(x, coords, self.tracker.input_coords())
                output, output_coords = self.tracker(x_track, coords_track)
                
                # Remove lead time dimension
                output = output[:, 0]
                all_tracks.append(output.cpu())
                
                pbar.update(1)
                if step == nsteps:
                    break
        
        # Save tracks
        final_tracks = torch.stack(all_tracks)
        torch.save(final_tracks, f"{output_dir}/regional_tracks.pt")
        
        return {
            'tracks': final_tracks,
            'times': forecast_times
        }
    
    def _evaluate_single_cyclone(
        self,
        cyclone: Dict,
        forecast_results: Dict,
        start_time: datetime,
        output_dir: str
    ) -> Dict:
        """Evaluate a single cyclone."""
        tracks = forecast_results['tracks']
        times = forecast_results['times']
        
        # Extract positions near the cyclone
        cyclone_positions = []
        
        for step, time in enumerate(times):
            step_data = tracks[step, 0]  # [num_detections, 4]
            
            # Find closest detection to initial cyclone position
            min_distance = float('inf')
            closest_detection = None
            
            for detection_idx in range(step_data.shape[0]):
                detection = step_data[detection_idx]
                lat, lon = detection[0].item(), detection[1].item()
                
                # Check if in region
                if not self.region_config.contains_point(lat, lon):
                    continue
                
                # Calculate distance to initial position
                dlat = lat - cyclone['lat']
                dlon = lon - cyclone['lon']
                distance = np.sqrt(dlat**2 + dlon**2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_detection = {
                        'time': time.isoformat(),
                        'lat': lat,
                        'lon': lon,
                        'distance_from_initial': distance
                    }
            
            if closest_detection:
                cyclone_positions.append(closest_detection)
        
        return {
            'name': cyclone['name'],
            'initial_position': {'lat': cyclone['lat'], 'lon': cyclone['lon']},
            'track_length': len(cyclone_positions),
            'positions': cyclone_positions
        }
    
    def visualize_regional_forecast(
        self,
        summary: Dict,
        output_dir: str
    ):
        """
        Create visualization of regional cyclone forecast.
        
        Args:
            summary: Evaluation summary dictionary
            output_dir: Output directory
        """
        if summary['status'] != 'success':
            print("No successful evaluations to visualize")
            return
        
        fig = plt.figure(figsize=(16, 12))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Set regional extent
        bounds = summary['region_bounds']
        ax.set_extent([
            bounds['lon_min'], bounds['lon_max'],
            bounds['lat_min'], bounds['lat_max']
        ], crs=ccrs.PlateCarree())
        
        # Add features
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--')
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, linewidth=0.5)
        
        # Plot cyclone tracks
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for idx, cyclone_eval in enumerate(summary['cyclone_evaluations']):
            color = colors[idx % len(colors)]
            name = cyclone_eval['name']
            
            # Initial position
            init_pos = cyclone_eval['initial_position']
            ax.plot(init_pos['lon'], init_pos['lat'], 'o', 
                   color=color, markersize=12, 
                   transform=ccrs.PlateCarree(),
                   label=f"{name} (initial)", zorder=5)
            
            # Track positions
            if cyclone_eval['positions']:
                lats = [p['lat'] for p in cyclone_eval['positions']]
                lons = [p['lon'] for p in cyclone_eval['positions']]
                
                ax.plot(lons, lats, '-', color=color, linewidth=2,
                       transform=ccrs.PlateCarree(), alpha=0.7)
                ax.plot(lons, lats, 'o', color=color, markersize=6,
                       transform=ccrs.PlateCarree(), alpha=0.7)
        
        # Title
        plt.title(
            f"Regional Tropical Cyclone Forecast\n"
            f"{summary['region']} - {summary['model']} Model ({summary['data_source']} Data)\n"
            f"Start: {summary['start_time'][:16]} UTC | Forecast: {summary['nsteps']*6} hours",
            fontsize=14, fontweight='bold'
        )
        
        # Legend
        ax.legend(loc='upper right', fontsize=10)
        
        # Save
        save_path = f"{output_dir}/regional_cyclone_forecast.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to {save_path}")
        plt.close()


def main():
    """Main function for regional cyclone evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate tropical cyclones with regional forecasting'
    )
    parser.add_argument(
        '--region',
        type=str,
        default='north_indian_ocean',
        choices=list(REGIONAL_DOMAINS.keys()),
        help='Regional domain'
    )
    parser.add_argument(
        '--data-source',
        type=str,
        default='GFS',
        choices=['GFS', 'ERA5'],
        help='Data source'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='SFNO',
        choices=['SFNO', 'DLWP', 'FCN'],
        help='Prognostic model'
    )
    parser.add_argument(
        '--nsteps',
        type=int,
        default=20,
        help='Number of forecast steps'
    )
    parser.add_argument(
        '--cyclones',
        type=str,
        default=None,
        help='JSON file with cyclone list (name, lat, lon)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/regional_cyclones',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # Load cyclone list
    if args.cyclones and os.path.exists(args.cyclones):
        with open(args.cyclones, 'r') as f:
            cyclone_list = json.load(f)
    else:
        # Example cyclone list for demonstration
        cyclone_list = [
            {'name': 'Test Cyclone 1', 'lat': 15.0, 'lon': 90.0},
            {'name': 'Test Cyclone 2', 'lat': 18.0, 'lon': 70.0},
        ]
        print("\nNo cyclone file provided. Using example cyclones:")
        for c in cyclone_list:
            print(f"  - {c['name']}: ({c['lat']}°N, {c['lon']}°E)")
    
    # Get start time
    start_time = get_last_available_time(args.data_source)
    
    # Initialize forecaster
    forecaster = RegionalCycloneForecaster(
        region=args.region,
        data_source=args.data_source,
        model_name=args.model
    )
    
    # Run evaluation
    summary = forecaster.evaluate_cyclones(
        cyclone_list=cyclone_list,
        start_time=start_time,
        nsteps=args.nsteps,
        output_dir=args.output_dir
    )
    
    # Visualize results
    if summary['status'] == 'success':
        forecaster.visualize_regional_forecast(summary, args.output_dir)
    
    print("\n" + "="*70)
    print("Evaluation Summary:")
    print(f"  Total cyclones: {summary['total_cyclones']}")
    print(f"  Regional cyclones: {summary['regional_cyclones']}")
    print(f"  Results saved to: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
