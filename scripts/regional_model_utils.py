"""
Regional Model Utilities for Indian Subcontinent
Provides tools for limited area modeling focused on the Indian Ocean region.
"""
from typing import Dict, Tuple, Optional, List
import numpy as np
import xarray as xr
import torch


# Pre-defined regional domains for Indian subcontinent
REGIONAL_DOMAINS = {
    'india_full': {
        'name': 'Full Indian Subcontinent',
        'lat_min': 5.0,
        'lat_max': 40.0,
        'lon_min': 65.0,
        'lon_max': 100.0,
        'description': 'Covers entire Indian subcontinent including surrounding waters'
    },
    'north_indian_ocean': {
        'name': 'North Indian Ocean',
        'lat_min': 0.0,
        'lat_max': 30.0,
        'lon_min': 40.0,
        'lon_max': 100.0,
        'description': 'Arabian Sea and Bay of Bengal'
    },
    'bay_of_bengal': {
        'name': 'Bay of Bengal',
        'lat_min': 5.0,
        'lat_max': 25.0,
        'lon_min': 80.0,
        'lon_max': 100.0,
        'description': 'Bay of Bengal cyclone-prone region'
    },
    'arabian_sea': {
        'name': 'Arabian Sea',
        'lat_min': 5.0,
        'lat_max': 25.0,
        'lon_min': 50.0,
        'lon_max': 80.0,
        'description': 'Arabian Sea cyclone-prone region'
    },
    'india_monsoon': {
        'name': 'Indian Monsoon Region',
        'lat_min': 8.0,
        'lat_max': 35.0,
        'lon_min': 68.0,
        'lon_max': 98.0,
        'description': 'Core monsoon region of India'
    }
}


class RegionalModelConfig:
    """Configuration for limited area model over Indian subcontinent."""
    
    def __init__(
        self,
        domain: str = 'india_full',
        custom_bounds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize regional model configuration.
        
        Args:
            domain: Pre-defined domain name or 'custom'
            custom_bounds: Custom domain boundaries if domain='custom'
                          Dict with keys: lat_min, lat_max, lon_min, lon_max
        """
        if domain == 'custom':
            if custom_bounds is None:
                raise ValueError("custom_bounds required when domain='custom'")
            self.bounds = custom_bounds
            self.name = 'Custom Region'
            self.description = 'User-defined region'
        elif domain in REGIONAL_DOMAINS:
            config = REGIONAL_DOMAINS[domain]
            self.bounds = {
                'lat_min': config['lat_min'],
                'lat_max': config['lat_max'],
                'lon_min': config['lon_min'],
                'lon_max': config['lon_max']
            }
            self.name = config['name']
            self.description = config['description']
        else:
            raise ValueError(
                f"Unknown domain: {domain}. "
                f"Available domains: {list(REGIONAL_DOMAINS.keys())} or 'custom'"
            )
        
        self.domain = domain
    
    def __repr__(self):
        return (
            f"RegionalModelConfig(domain='{self.domain}', "
            f"lat=[{self.bounds['lat_min']}, {self.bounds['lat_max']}], "
            f"lon=[{self.bounds['lon_min']}, {self.bounds['lon_max']}])"
        )
    
    def get_bounds(self) -> Dict[str, float]:
        """Get regional boundaries."""
        return self.bounds.copy()
    
    def contains_point(self, lat: float, lon: float) -> bool:
        """Check if a point is within the regional domain."""
        return (
            self.bounds['lat_min'] <= lat <= self.bounds['lat_max'] and
            self.bounds['lon_min'] <= lon <= self.bounds['lon_max']
        )


def subset_data_to_region(
    data: xr.DataArray,
    region_config: RegionalModelConfig,
    lat_name: str = 'lat',
    lon_name: str = 'lon'
) -> xr.DataArray:
    """
    Subset global data to regional domain.
    
    Args:
        data: Global data array
        region_config: Regional configuration
        lat_name: Name of latitude dimension
        lon_name: Name of longitude dimension
    
    Returns:
        Subsetted data array
    """
    bounds = region_config.get_bounds()
    
    # Handle longitude wrapping if needed
    lon_values = data[lon_name].values
    if lon_values.max() > 180:
        # Data is in 0-360 format
        lon_min = bounds['lon_min']
        lon_max = bounds['lon_max']
    else:
        # Data is in -180 to 180 format
        lon_min = bounds['lon_min'] if bounds['lon_min'] <= 180 else bounds['lon_min'] - 360
        lon_max = bounds['lon_max'] if bounds['lon_max'] <= 180 else bounds['lon_max'] - 360
    
    # Subset the data
    subset = data.sel(
        {
            lat_name: slice(bounds['lat_min'], bounds['lat_max']),
            lon_name: slice(lon_min, lon_max)
        }
    )
    
    return subset


def subset_tensor_to_region(
    tensor: torch.Tensor,
    coords: Dict,
    region_config: RegionalModelConfig,
    lat_name: str = 'lat',
    lon_name: str = 'lon'
) -> Tuple[torch.Tensor, Dict]:
    """
    Subset global tensor data to regional domain.
    
    Note: This is a simplified implementation that assumes specific tensor dimensions.
    For production use, this should be extended to handle arbitrary tensor shapes
    and dimension orders. The current implementation works for tensors where lat/lon
    are the last two dimensions or can be easily identified.
    
    Args:
        tensor: Global data tensor
        coords: Coordinate dictionary
        region_config: Regional configuration
        lat_name: Name of latitude dimension
        lon_name: Name of longitude dimension
    
    Returns:
        Tuple of (subsetted tensor, updated coords)
    """
    bounds = region_config.get_bounds()
    
    # Get coordinate arrays
    lat_array = coords[lat_name]
    lon_array = coords[lon_name]
    
    # Find indices for subsetting
    lat_mask = (lat_array >= bounds['lat_min']) & (lat_array <= bounds['lat_max'])
    lon_mask = (lon_array >= bounds['lon_min']) & (lon_array <= bounds['lon_max'])
    
    lat_indices = torch.where(lat_mask)[0]
    lon_indices = torch.where(lon_mask)[0]
    
    # Get dimension positions
    dims = list(coords.keys())
    lat_dim = dims.index(lat_name)
    lon_dim = dims.index(lon_name)
    
    # Subset tensor along lat and lon dimensions
    subset_tensor = tensor.index_select(lat_dim, lat_indices)
    subset_tensor = subset_tensor.index_select(lon_dim, lon_indices)
    
    # Update coordinates
    new_coords = coords.copy()
    new_coords[lat_name] = lat_array[lat_mask]
    new_coords[lon_name] = lon_array[lon_mask]
    
    return subset_tensor, new_coords


def create_regional_mask(
    lat_array: np.ndarray,
    lon_array: np.ndarray,
    region_config: RegionalModelConfig
) -> np.ndarray:
    """
    Create a boolean mask for the regional domain.
    
    Args:
        lat_array: 1D or 2D array of latitudes
        lon_array: 1D or 2D array of longitudes
        region_config: Regional configuration
    
    Returns:
        Boolean mask (True inside region)
    """
    bounds = region_config.get_bounds()
    
    # Handle 1D or 2D coordinate arrays
    if lat_array.ndim == 1 and lon_array.ndim == 1:
        # Create 2D meshgrid
        lon_2d, lat_2d = np.meshgrid(lon_array, lat_array)
    else:
        lat_2d = lat_array
        lon_2d = lon_array
    
    # Create mask
    mask = (
        (lat_2d >= bounds['lat_min']) &
        (lat_2d <= bounds['lat_max']) &
        (lon_2d >= bounds['lon_min']) &
        (lon_2d <= bounds['lon_max'])
    )
    
    return mask


def get_regional_grid_info(
    region_config: RegionalModelConfig,
    resolution: float = 0.25
) -> Dict:
    """
    Get grid information for the regional domain.
    
    Args:
        region_config: Regional configuration
        resolution: Grid resolution in degrees (default: 0.25)
    
    Returns:
        Dictionary with grid information
    """
    bounds = region_config.get_bounds()
    
    # Calculate grid points
    lat_points = int((bounds['lat_max'] - bounds['lat_min']) / resolution) + 1
    lon_points = int((bounds['lon_max'] - bounds['lon_min']) / resolution) + 1
    
    # Create coordinate arrays
    lat_array = np.linspace(bounds['lat_min'], bounds['lat_max'], lat_points)
    lon_array = np.linspace(bounds['lon_min'], bounds['lon_max'], lon_points)
    
    # Calculate area (approximate)
    lat_center = (bounds['lat_min'] + bounds['lat_max']) / 2
    lon_span = bounds['lon_max'] - bounds['lon_min']
    lat_span = bounds['lat_max'] - bounds['lat_min']
    
    # Approximate area in km^2
    km_per_degree_lat = 111.0
    km_per_degree_lon = 111.0 * np.cos(np.radians(lat_center))
    area_km2 = (lon_span * km_per_degree_lon) * (lat_span * km_per_degree_lat)
    
    return {
        'lat_points': lat_points,
        'lon_points': lon_points,
        'total_points': lat_points * lon_points,
        'lat_array': lat_array,
        'lon_array': lon_array,
        'resolution': resolution,
        'area_km2': area_km2,
        'bounds': bounds,
        'name': region_config.name
    }


def print_regional_domains():
    """Print available pre-defined regional domains."""
    print("\nAvailable Regional Domains for Indian Subcontinent:")
    print("=" * 70)
    
    for domain_id, config in REGIONAL_DOMAINS.items():
        print(f"\nDomain: {domain_id}")
        print(f"  Name: {config['name']}")
        print(f"  Description: {config['description']}")
        print(f"  Latitude: [{config['lat_min']}°N, {config['lat_max']}°N]")
        print(f"  Longitude: [{config['lon_min']}°E, {config['lon_max']}°E]")
        
        # Calculate approximate coverage
        lat_span = config['lat_max'] - config['lat_min']
        lon_span = config['lon_max'] - config['lon_min']
        lat_center = (config['lat_min'] + config['lat_max']) / 2
        
        km_per_degree_lat = 111.0
        km_per_degree_lon = 111.0 * np.cos(np.radians(lat_center))
        area_km2 = (lon_span * km_per_degree_lon) * (lat_span * km_per_degree_lat)
        
        print(f"  Approximate area: {area_km2:,.0f} km²")
    
    print("\n" + "=" * 70)


def main():
    """Demonstrate regional model utilities."""
    print("Regional Model Utilities for Indian Subcontinent")
    print("=" * 70)
    
    # Print available domains
    print_regional_domains()
    
    # Example: Create regional configuration
    print("\n\nExample: Creating regional configuration for Bay of Bengal")
    print("-" * 70)
    
    region = RegionalModelConfig('bay_of_bengal')
    print(f"\n{region}")
    print(f"Name: {region.name}")
    print(f"Description: {region.description}")
    
    # Get grid information
    grid_info = get_regional_grid_info(region, resolution=0.25)
    print(f"\nGrid Information (0.25° resolution):")
    print(f"  Latitude points: {grid_info['lat_points']}")
    print(f"  Longitude points: {grid_info['lon_points']}")
    print(f"  Total grid points: {grid_info['total_points']:,}")
    print(f"  Area: {grid_info['area_km2']:,.0f} km²")
    
    # Test point containment
    print(f"\nTesting point containment:")
    test_points = [
        (15.0, 90.0, "Cyclone-prone area"),
        (20.0, 88.0, "Near Bangladesh coast"),
        (30.0, 70.0, "Outside domain")
    ]
    
    for lat, lon, description in test_points:
        is_inside = region.contains_point(lat, lon)
        status = "Inside" if is_inside else "Outside"
        print(f"  Point ({lat}°N, {lon}°E) - {description}: {status}")


if __name__ == "__main__":
    main()
