import unittest
from your_module import (
    remaining_angles,
)  # Replace 'your_module' with the actual module name


class TestRemainingAngles(unittest.TestCase):
    def test_no_angles(self):
        self.assertEqual(remaining_angles([]), [])

    def test_single_angle(self):
        self.assertEqual(remaining_angles([90]), [270])

    def test_multiple_angles(self):
        self.assertEqual(remaining_angles([0, 90, 180]), [360, 270, 180])

    def test_invalid_angle(self):
        with self.assertRaises(ValueError):
            remaining_angles([-10])

    def test_angles_with_wrap_around(self):
        self.assertEqual(remaining_angles([350, 10]), [10, 350])


if __name__ == "__main__":
    unittest.main()
