"""
Unit Test Template Matching with SSIM
"""
import unittest
import numpy as np
from pyvhs.utils.edits import templates_similarity

# Set seed
np.random.seed(42)

# Dummy data
template = np.random.randint(low=0,
                             high=256,
                             size=(32, 32, 3),
                             dtype='uint8')
frame = np.random.randint(low=0,
                          high=256,
                          size=(32, 32, 3),
                          dtype='uint8')
class TestSSIM(unittest.TestCase):
    """
    Check SSIM calculation between two arrays
    """

    def test_ssim_calc(self):
        sim, _ = templates_similarity(template_imgs=[template],
                                      img=frame,
                                      threshold=0.9)
        # sim = round(sim, 3)
        # print(f'\n\n\nlook here: {sim}\n\n\n')
        self.assertAlmostEqual(first=sim,
                               second=-0.02271268822123081,
                               places=5,
                               msg='Incorrect SSIM Calculation')


if __name__ == "__main__":
    # Run the unittest
    unittest.main()
