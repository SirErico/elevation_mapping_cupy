#
# Slope filter plugin for elevation_mapping_cupy.
# Computes slope in degrees from elevation gradients, NaN-tolerant.
#
import cupy as cp
from typing import List

from .plugin_manager import PluginBase


class SlopeFilter(PluginBase):
    """Slope angle (degrees) from elevation gradients.

    Uses forward/backward differences combined with cp.fmax so a single
    NaN neighbor does not poison the cell. A cell is NaN only if both
    sides of an axis are NaN.
    """

    def __init__(self, cell_n: int = 100, input_layer_name: str = "elevation",
                 resolution: float = 0.1, **kwargs):
        super().__init__()
        self.input_layer_name = input_layer_name
        self.resolution = cp.float32(resolution)

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        if self.input_layer_name in layer_names:
            h = elevation_map[layer_names.index(self.input_layer_name)]
        elif self.input_layer_name in plugin_layer_names:
            h = plugin_layers[plugin_layer_names.index(self.input_layer_name)]
        else:
            h = elevation_map[0]

        dx_f = (cp.roll(h, -1, axis=1) - h) / self.resolution
        dx_b = (h - cp.roll(h,  1, axis=1)) / self.resolution
        dy_f = (cp.roll(h, -1, axis=0) - h) / self.resolution
        dy_b = (h - cp.roll(h,  1, axis=0)) / self.resolution

        gx = cp.fmax(cp.abs(dx_f), cp.abs(dx_b))
        gy = cp.fmax(cp.abs(dy_f), cp.abs(dy_b))

        slope_rad = cp.arctan(cp.sqrt(gx * gx + gy * gy))
        return cp.degrees(slope_rad)
