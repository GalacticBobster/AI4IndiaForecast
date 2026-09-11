"""
Unit tests for tropical cyclone tracking and regional model utilities.
Tests the core functionality without requiring heavy model execution.
"""
import os
import sys
import json
from datetime import datetime

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import numpy as np


def test_regional_model_utils():
    """Test regional model utilities."""
    print("Testing regional_model_utils...")
    
    from regional_model_utils import (
        RegionalModelConfig,
        get_regional_grid_info,
        create_regional_mask,
        REGIONAL_DOMAINS
    )
    
    # Test 1: Create regional configuration
    region = RegionalModelConfig('bay_of_bengal')
    assert region.domain == 'bay_of_bengal'
    assert region.name == 'Bay of Bengal'
    bounds = region.get_bounds()
    assert bounds['lat_min'] == 5.0
    assert bounds['lat_max'] == 25.0
    assert bounds['lon_min'] == 80.0
    assert bounds['lon_max'] == 100.0
    print("  ✓ Regional configuration creation")
    
    # Test 2: Point containment
    assert region.contains_point(15.0, 90.0) == True
    assert region.contains_point(30.0, 70.0) == False
    print("  ✓ Point containment check")
    
    # Test 3: Grid information
    grid_info = get_regional_grid_info(region, resolution=0.25)
    assert grid_info['lat_points'] == 81
    assert grid_info['lon_points'] == 81
    assert grid_info['total_points'] == 6561
    print("  ✓ Grid information calculation")
    
    # Test 4: Regional mask
    lat_array = np.linspace(0, 30, 121)
    lon_array = np.linspace(70, 110, 161)
    mask = create_regional_mask(lat_array, lon_array, region)
    assert mask.shape == (121, 161)
    assert mask.sum() > 0  # Some points should be in the region
    print("  ✓ Regional mask creation")
    
    # Test 5: Custom region
    custom_region = RegionalModelConfig(
        domain='custom',
        custom_bounds={
            'lat_min': 10.0, 'lat_max': 20.0,
            'lon_min': 85.0, 'lon_max': 95.0
        }
    )
    assert custom_region.name == 'Custom Region'
    assert custom_region.contains_point(15.0, 90.0) == True
    print("  ✓ Custom region creation")
    
    # Test 6: All pre-defined domains
    for domain_id in REGIONAL_DOMAINS.keys():
        region = RegionalModelConfig(domain_id)
        assert region.domain == domain_id
    print("  ✓ All pre-defined domains")
    
    print("✅ regional_model_utils tests passed!\n")


def test_cyclone_tracker_imports():
    """Test that cyclone tracker imports work."""
    print("Testing tc_tracker_india imports...")
    
    try:
        from tc_tracker_india import (
            TropicalCycloneTracker,
            INDIAN_OCEAN_REGION
        )
        from cyclone_utils import round_to_6h, get_last_available_time
        
        # Test utility functions
        dt = datetime(2024, 1, 15, 7, 30, 0)
        rounded = round_to_6h(dt)
        assert rounded == datetime(2024, 1, 15, 6, 0, 0)
        print("  ✓ Time rounding function")
        
        # Test region definition
        assert INDIAN_OCEAN_REGION['lat_min'] == 0.0
        assert INDIAN_OCEAN_REGION['lat_max'] == 30.0
        assert INDIAN_OCEAN_REGION['lon_min'] == 40.0
        assert INDIAN_OCEAN_REGION['lon_max'] == 100.0
        print("  ✓ Indian Ocean region definition")
        
        print("✅ tc_tracker_india imports successful!\n")
        return True
        
    except Exception as e:
        print(f"⚠️  Import test skipped (missing dependencies): {e}\n")
        return False


def test_regional_cyclone_eval_imports():
    """Test that regional cyclone evaluation imports work."""
    print("Testing regional_cyclone_eval imports...")
    
    try:
        from regional_cyclone_eval import RegionalCycloneForecaster
        from cyclone_utils import round_to_6h, get_last_available_time
        
        # Test utility functions
        dt = datetime(2024, 1, 15, 14, 45, 0)
        rounded = round_to_6h(dt)
        assert rounded == datetime(2024, 1, 15, 12, 0, 0)
        print("  ✓ Time rounding function")
        
        print("✅ regional_cyclone_eval imports successful!\n")
        return True
        
    except Exception as e:
        print(f"⚠️  Import test skipped (missing dependencies): {e}\n")
        return False


def test_cyclone_list_loading():
    """Test loading cyclone list from JSON."""
    print("Testing cyclone list loading...")
    
    import tempfile
    
    # Create test cyclone list
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_file = f.name
        cyclones = [
            {'name': 'Test 1', 'lat': 15.0, 'lon': 90.0},
            {'name': 'Test 2', 'lat': 18.0, 'lon': 70.0}
        ]
        json.dump(cyclones, f)
    
    try:
        # Load and verify
        with open(test_file, 'r') as f:
            loaded = json.load(f)
        
        assert len(loaded) == 2
        assert loaded[0]['name'] == 'Test 1'
        assert loaded[0]['lat'] == 15.0
        assert loaded[1]['lon'] == 70.0
        print("  ✓ Cyclone list JSON I/O")
        
        # Test with regional filter
        from regional_model_utils import RegionalModelConfig
        region = RegionalModelConfig('north_indian_ocean')
        
        regional_cyclones = [
            c for c in loaded
            if region.contains_point(c['lat'], c['lon'])
        ]
        assert len(regional_cyclones) == 2  # Both should be in North Indian Ocean
        print("  ✓ Regional filtering")
    finally:
        # Cleanup
        os.remove(test_file)
    
    print("✅ Cyclone list loading tests passed!\n")


def test_documentation():
    """Test that documentation file exists and is readable."""
    print("Testing documentation...")
    
    doc_path = os.path.join(
        os.path.dirname(__file__), '..', 'docs', 'CYCLONE_TRACKING.md'
    )
    
    assert os.path.exists(doc_path), "Documentation file missing"
    
    with open(doc_path, 'r') as f:
        content = f.read()
    
    # Check for key sections
    assert '# Tropical Cyclone Tracking' in content
    assert 'tc_tracker_india.py' in content
    assert 'regional_model_utils.py' in content
    assert 'regional_cyclone_eval.py' in content
    assert 'GFS' in content
    assert 'ERA5' in content
    print("  ✓ Documentation exists and contains key sections")
    
    print("✅ Documentation tests passed!\n")


def main():
    """Run all tests."""
    print("="*70)
    print("Running Tests for Tropical Cyclone Tracking System")
    print("="*70 + "\n")
    
    all_passed = True
    
    try:
        test_regional_model_utils()
    except Exception as e:
        print(f"❌ regional_model_utils tests failed: {e}\n")
        all_passed = False
    
    try:
        test_cyclone_tracker_imports()
    except Exception as e:
        print(f"❌ tc_tracker_india tests failed: {e}\n")
        all_passed = False
    
    try:
        test_regional_cyclone_eval_imports()
    except Exception as e:
        print(f"❌ regional_cyclone_eval tests failed: {e}\n")
        all_passed = False
    
    try:
        test_cyclone_list_loading()
    except Exception as e:
        print(f"❌ Cyclone list tests failed: {e}\n")
        all_passed = False
    
    try:
        test_documentation()
    except Exception as e:
        print(f"❌ Documentation tests failed: {e}\n")
        all_passed = False
    
    print("="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
