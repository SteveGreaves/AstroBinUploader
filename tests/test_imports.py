
import sys
import os

# Add the parent directory to sys.path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    try:
        import AstroBinUpload
        import config_functions
        import headers_functions
        import processing_functions
        import sites_functions
        import utils
    except ImportError as e:
        assert False, f"Import failed: {e}"

def test_version_consistency():
    import AstroBinUpload
    import utils
    # Check if the versions match as expected in the main script
    assert AstroBinUpload.version == utils.utils_version
