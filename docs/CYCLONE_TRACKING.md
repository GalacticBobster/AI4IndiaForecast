# Tropical Cyclone Tracking for Indian Subcontinent

This directory contains scripts for tracking and evaluating tropical cyclones around the Indian subcontinent using Earth2Studio models with GFS/ERA5 data sources.

## Features

### 1. Tropical Cyclone Tracker (`tc_tracker_india.py`)
Track tropical cyclones in the Indian Ocean region using advanced AI models.

**Features:**
- Support for both GFS and ERA5 data sources
- Multiple prognostic models (SFNO, DLWP)
- Automated cyclone detection and tracking
- Visualization of cyclone paths
- Focus on Indian Ocean region (0-30°N, 40-100°E)

**Usage:**
```bash
# Track cyclones using GFS data and SFNO model
python scripts/tc_tracker_india.py --data-source GFS --model SFNO --nsteps 20

# Track cyclones using ERA5 data and DLWP model
python scripts/tc_tracker_india.py --data-source ERA5 --model DLWP --nsteps 30

# Specify custom start time (format: YYYY-MM-DD-HH)
python scripts/tc_tracker_india.py --start-time 2024-08-15-00 --nsteps 20

# Specify custom output directory
python scripts/tc_tracker_india.py --output-dir outputs/my_cyclone_tracks
```

**Command-line Arguments:**
- `--data-source`: Data source (GFS or ERA5), default: GFS
- `--model`: Prognostic model (SFNO or DLWP), default: SFNO
- `--nsteps`: Number of 6-hour forecast steps, default: 20
- `--output-dir`: Output directory, default: outputs/cyclone_tracks
- `--start-time`: Start time in YYYY-MM-DD-HH format (optional)

**Outputs:**
- `tracks_step_XXX.pt`: Intermediate tracking results (every 5 steps)
- `tracks_final.pt`: Final cyclone tracks tensor
- `cyclone_info.json`: Detected cyclone information
- `cyclone_tracks.png`: Visualization of cyclone paths

### 2. Regional Model Utilities (`regional_model_utils.py`)
Tools for limited area modeling focused on the Indian subcontinent.

**Features:**
- Pre-defined regional domains:
  - `india_full`: Full Indian subcontinent (5-40°N, 65-100°E)
  - `north_indian_ocean`: North Indian Ocean (0-30°N, 40-100°E)
  - `bay_of_bengal`: Bay of Bengal (5-25°N, 80-100°E)
  - `arabian_sea`: Arabian Sea (5-25°N, 50-80°E)
  - `india_monsoon`: Indian monsoon region (8-35°N, 68-98°E)
- Custom domain definition
- Grid information calculation
- Data subsetting for regional domains
- Regional mask creation

**Usage:**
```python
from regional_model_utils import RegionalModelConfig, get_regional_grid_info

# Create regional configuration
region = RegionalModelConfig('bay_of_bengal')

# Get grid information
grid_info = get_regional_grid_info(region, resolution=0.25)
print(f"Grid points: {grid_info['total_points']:,}")
print(f"Area: {grid_info['area_km2']:,.0f} km²")

# Check if point is in region
is_inside = region.contains_point(lat=15.0, lon=90.0)

# Custom region
custom_region = RegionalModelConfig(
    domain='custom',
    custom_bounds={
        'lat_min': 10.0, 'lat_max': 25.0,
        'lon_min': 75.0, 'lon_max': 95.0
    }
)
```

**Command-line Demo:**
```bash
# Display available domains and examples
python scripts/regional_model_utils.py
```

### 3. Regional Cyclone Evaluation (`regional_cyclone_eval.py`)
Comprehensive system for evaluating multiple cyclones with regional forecasting.

**Features:**
- Evaluate a list of tropical cyclones
- Regional domain filtering
- Combined forecasting and tracking
- Cyclone-specific track extraction
- Comprehensive visualization

**Usage:**
```bash
# Basic usage with example cyclones
python scripts/regional_cyclone_eval.py --region north_indian_ocean

# Use specific model and data source
python scripts/regional_cyclone_eval.py \
    --region bay_of_bengal \
    --data-source GFS \
    --model SFNO \
    --nsteps 25

# Provide custom cyclone list
python scripts/regional_cyclone_eval.py \
    --cyclones cyclone_list.json \
    --region north_indian_ocean
```

**Cyclone List Format (JSON):**
```json
[
  {
    "name": "Cyclone Name 1",
    "lat": 15.0,
    "lon": 90.0
  },
  {
    "name": "Cyclone Name 2",
    "lat": 18.0,
    "lon": 70.0
  }
]
```

**Command-line Arguments:**
- `--region`: Regional domain (see available domains above), default: north_indian_ocean
- `--data-source`: Data source (GFS or ERA5), default: GFS
- `--model`: Prognostic model (SFNO, DLWP, or FCN), default: SFNO
- `--nsteps`: Number of 6-hour forecast steps, default: 20
- `--cyclones`: JSON file with cyclone list (optional)
- `--output-dir`: Output directory, default: outputs/regional_cyclones

**Outputs:**
- `regional_tracks.pt`: Regional cyclone tracks
- `evaluation_summary.json`: Evaluation summary with cyclone positions
- `regional_cyclone_forecast.png`: Visualization of cyclone forecasts

## Regional Domains

### Pre-defined Domains

| Domain ID | Name | Coverage | Area |
|-----------|------|----------|------|
| `india_full` | Full Indian Subcontinent | 5-40°N, 65-100°E | ~4.3M km² |
| `north_indian_ocean` | North Indian Ocean | 0-30°N, 40-100°E | ~5.5M km² |
| `bay_of_bengal` | Bay of Bengal | 5-25°N, 80-100°E | ~1.1M km² |
| `arabian_sea` | Arabian Sea | 5-25°N, 50-80°E | ~1.5M km² |
| `india_monsoon` | Indian Monsoon Region | 8-35°N, 68-98°E | ~2.5M km² |

### Cyclone-Prone Areas

The Bay of Bengal and Arabian Sea are the primary cyclone formation regions in the Indian Ocean:

- **Bay of Bengal**: Most active during pre-monsoon (April-May) and post-monsoon (October-November)
- **Arabian Sea**: Active during similar periods but generally produces fewer cyclones
- **North Indian Ocean**: Encompasses both basins for comprehensive monitoring

## Data Sources

### GFS (Global Forecast System)
- **Update Frequency**: Every 6 hours (00, 06, 12, 18 UTC)
- **Availability Delay**: ~6-12 hours
- **Spatial Resolution**: 0.25° (~25 km)
- **Recommended for**: Real-time operational forecasting

### ERA5 (ECMWF Reanalysis v5)
- **Update Frequency**: Historical reanalysis
- **Availability Delay**: ~5 days behind real-time
- **Spatial Resolution**: 0.25° (~25 km)
- **Recommended for**: Historical analysis, validation, research

## Models

### SFNO (Spherical Fourier Neural Operator)
- Fast inference
- Good for medium-range forecasts (up to 5 days)
- Excellent for cyclone tracking

### DLWP (Deep Learning Weather Prediction)
- Robust performance
- Good for longer-range forecasts
- Balanced accuracy

### FCN (FourCastNet)
- High-resolution capable
- Fast inference
- Good for short-range high-resolution forecasts

## Examples

### Example 1: Track Current Cyclones
```bash
# Track any active cyclones in the North Indian Ocean using latest GFS data
python scripts/tc_tracker_india.py \
    --data-source GFS \
    --model SFNO \
    --nsteps 30 \
    --output-dir outputs/current_cyclones
```

### Example 2: Bay of Bengal Cyclone Season Monitoring
```bash
# Monitor Bay of Bengal during cyclone season
python scripts/regional_cyclone_eval.py \
    --region bay_of_bengal \
    --data-source GFS \
    --model SFNO \
    --nsteps 40
```

### Example 3: Historical Cyclone Analysis
```bash
# Analyze a historical cyclone using ERA5 data
python scripts/tc_tracker_india.py \
    --data-source ERA5 \
    --model DLWP \
    --start-time 2023-05-10-00 \
    --nsteps 50 \
    --output-dir outputs/historical_analysis
```

### Example 4: Multi-Cyclone Evaluation
Create a cyclone list file `active_cyclones.json`:
```json
[
  {"name": "BOB 01", "lat": 14.5, "lon": 89.5},
  {"name": "ARB 01", "lat": 16.0, "lon": 68.0}
]
```

Run evaluation:
```bash
python scripts/regional_cyclone_eval.py \
    --cyclones active_cyclones.json \
    --region north_indian_ocean \
    --nsteps 30 \
    --output-dir outputs/multi_cyclone_eval
```

## Output Interpretation

### Cyclone Track Files (`.pt`)
PyTorch tensors containing cyclone detections:
- Shape: `[time_steps, batch, num_detections, 4]`
- 4 channels: latitude, longitude, wind speed, pressure

### Cyclone Info JSON
```json
{
  "num_cyclones": 2,
  "cyclones": [
    {
      "step": 0,
      "time": "2024-01-15T00:00:00",
      "lat": 15.0,
      "lon": 90.0,
      "wind_speed": 25.5,
      "pressure": 998.0
    }
  ],
  "region": {...},
  "data_source": "GFS",
  "model": "SFNO"
}
```

### Evaluation Summary JSON
```json
{
  "status": "success",
  "region": "North Indian Ocean",
  "total_cyclones": 2,
  "regional_cyclones": 2,
  "cyclone_evaluations": [
    {
      "name": "Cyclone 1",
      "initial_position": {"lat": 15.0, "lon": 90.0},
      "track_length": 30,
      "positions": [...]
    }
  ]
}
```

## Requirements

All requirements are listed in the repository's `requirements.txt`:
- `earth2studio[dlwp]>=0.9.0`
- `torch>=2.1`
- `numpy>=1.24`
- `matplotlib>=3.7`
- `cartopy>=0.22`
- Additional dependencies as needed

## Notes

- All times are in UTC
- Forecast steps are at 6-hour intervals
- GPU acceleration is automatically used if available
- Outputs are saved in PyTorch format for flexibility
- Visualizations are saved as PNG images

## Limitations

- Model accuracy depends on initial conditions and data quality
- Cyclone intensity forecasts may have larger uncertainties
- Regional models still use global model outputs (not true limited-area modeling)
- Historical data (ERA5) has a 5-day delay

## Future Enhancements

- Integration with real-time cyclone databases (IMD, JTWC)
- Ensemble forecasting for uncertainty quantification
- True limited-area model implementation with regional physics
- Real-time alert system for cyclone formation
- Integration with satellite data for validation

## References

- Earth2Studio: https://github.com/NVIDIA/earth2studio
- GFS Data: NOAA Global Forecast System
- ERA5 Data: ECMWF Reanalysis v5
- Tropical Cyclone Tracker: Wu & Duan method implementation
