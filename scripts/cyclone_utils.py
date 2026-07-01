"""
Common utility functions for cyclone tracking and regional modeling.
Shared functions to avoid code duplication across modules.
"""
from datetime import datetime, timedelta
from typing import Optional


def round_to_6h(reference_time: datetime) -> datetime:
    """
    Round datetime to nearest past 6-hour cycle.
    
    Args:
        reference_time: Datetime to round
    
    Returns:
        Rounded datetime
    """
    hour = (reference_time.hour // 6) * 6
    rounded = reference_time.replace(hour=hour, minute=0, second=0, microsecond=0)
    if rounded > reference_time:
        rounded -= timedelta(hours=6)
    return rounded


def get_last_available_time(reference_time: Optional[datetime] = None, source: str = "GFS") -> datetime:
    """
    Return a datetime that is guaranteed to exist for the specified data source.
    
    Args:
        reference_time: Reference datetime (default: current UTC time)
        source: Data source type ("GFS" or "ERA5")
    
    Returns:
        Available datetime for the data source
    """
    if reference_time is None:
        reference_time = datetime.utcnow()
    
    rounded_time = round_to_6h(reference_time)
    
    if source.upper() == "GFS":
        # GFS typically has 6-12 hour delay
        last_available = round_to_6h(datetime.utcnow() - timedelta(hours=12))
    else:  # ERA5
        # ERA5 has longer delay (typically 5 days)
        last_available = round_to_6h(datetime.utcnow() - timedelta(days=5))
    
    if rounded_time > last_available:
        rounded_time = last_available
    
    return rounded_time
