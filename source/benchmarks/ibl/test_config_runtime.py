import config
from config_runtime import apply_config_snapshot


def test_legacy_precision_kmeans_maps_to_precision_assign():
    old_assign = config.PRECISION_ASSIGN
    try:
        apply_config_snapshot({"PRECISION_KMEANS": 5})
        assert config.PRECISION_ASSIGN == 5
    finally:
        config.PRECISION_ASSIGN = old_assign
