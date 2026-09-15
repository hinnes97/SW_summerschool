"""Exercise installed imports, both boundaries, time stepping and saved output."""
from pathlib import Path
import tempfile
import unittest

import netCDF4
import numpy as np
from sw_summerschool import SW_model


class ModelTests(unittest.TestCase):
    def test_integration_and_output(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        for config in ("cfg.yaml", "cfg_rossby.yaml"):
            with self.subTest(config=config), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "simulation.nc"
                model = SW_model(examples / config, N=8, Ro=0.25,
                                 outfile=str(output), io_freq=2, print_freq=2)
                try:
                    model.integrate(8)
                    self.assertAlmostEqual(model.time, 0.008)
                    expected = {name: getattr(model.grid, name)[model.grid.realslice[name]].copy()
                                for name in ("h", "u", "v")}
                    for values in expected.values():
                        self.assertTrue(np.isfinite(values).all())
                finally:
                    model.io.close()
                with netCDF4.Dataset(output) as data:
                    np.testing.assert_allclose(data["time"][:], [0, .002, .004, .006, .008])
                    for name, values in expected.items():
                        np.testing.assert_array_equal(data[name][-1], values)
                    self.assertEqual(data["h"].dimensions, ("time", "xmid", "ymid"))


if __name__ == "__main__":
    unittest.main()
