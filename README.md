# AI4IndiaForecast

[![Daily Earth2Studio Inference](https://github.com/GalacticBobster/AI4IndiaForecast/actions/workflows/run_earth2studio.yml/badge.svg)](https://github.com/GalacticBobster/AI4IndiaForecast/actions/workflows/run_earth2studio.yml)

Daily Weather Forecast for Indian subcontinent based on AI emulators through NVIDIA Earth2Studio  ☁️

## Features

### Weather Forecasting
- Daily automated weather forecasts for the Indian subcontinent
- Multiple AI models: DLWP, SFNO, FCN
- Temperature and precipitation predictions
- GFS data integration

### 🌀 Tropical Cyclone Tracking
NEW: Track and evaluate tropical cyclones around the Indian subcontinent using advanced AI models and regional forecasting capabilities.

**Key Features:**
- Multi-cyclone tracking and evaluation
- Support for GFS and ERA5 data sources
- Regional/limited area modeling for Indian Ocean
- Pre-defined domains (Bay of Bengal, Arabian Sea, etc.)
- Automated cyclone detection and path visualization

**See [Cyclone Tracking Documentation](docs/CYCLONE_TRACKING.md) for detailed usage.**

#### Quick Start - Cyclone Tracking

Track cyclones in the North Indian Ocean:
```bash
python scripts/tc_tracker_india.py --data-source GFS --model SFNO --nsteps 20
```

Evaluate specific cyclones with regional forecasting:
```bash
python scripts/regional_cyclone_eval.py --region bay_of_bengal --cyclones cyclone_list.json
```

Run the demo:
```bash
./demo_cyclone_tracking.sh
```
