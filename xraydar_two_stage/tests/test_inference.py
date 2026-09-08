"""Synthetic regression tests; no checkpoints or patient data required."""
import unittest
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk
import torch

from predict import (
    centered_logit_ensemble, classify, mask_for_reference, positive_mask,
    prepare_input, restore_probability,
)
from window_normalization import oriented_image_and_valid, polarity_inverted


class InferenceTests(unittest.TestCase):
    def test_polarity_precedence(self):
        for photo, lut, expected in [
            ('MONOCHROME1', '', True), ('MONOCHROME2', '', False),
            ('MONOCHROME1', 'INVERSE', True),
            ('MONOCHROME1', 'IDENTITY', False),
            ('MONOCHROME2', 'INVERSE', True),
        ]:
            with self.subTest(photo=photo, lut=lut):
                ds = SimpleNamespace(PhotometricInterpretation=photo,
                                     PresentationLUTShape=lut)
                self.assertEqual(polarity_inverted(ds), expected)

    def test_single_inversion_and_padding(self):
        ds = SimpleNamespace(pixel_array=np.array([[0, 255]], dtype=np.uint8),
                             BitsStored=8, PhotometricInterpretation='MONOCHROME1',
                             PresentationLUTShape='INVERSE', PixelPaddingValue=0)
        image, valid, padding = oriented_image_and_valid(ds)
        np.testing.assert_array_equal(image, [[1, 0]])
        np.testing.assert_array_equal(valid, [[False, True]])
        np.testing.assert_array_equal(padding, [[True, False]])

    def test_classifier_mean_probabilities(self):
        class Fixed(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, images):
                return torch.full((len(images),), self.value)

        result = classify([Fixed(0.), Fixed(2.)], torch.zeros(2, 1, 4, 4),
                          torch.device('cpu'), 0.)
        np.testing.assert_allclose(result, (0.5 + torch.sigmoid(torch.tensor(2.)).item()) / 2)

    def test_centered_boundary(self):
        result = centered_logit_ensemble(np.array([.06]), np.array([.54]),
                                        .06, .54, .5)
        np.testing.assert_allclose(result, [.5])
        self.assertTrue(np.isfinite(centered_logit_ensemble(
            np.array([0., 1.]), np.array([0., 1.]), .06, .54, .5)).all())

    def test_crop_and_restore_geometry(self):
        normalizer = SimpleNamespace(normalize=lambda _: np.ones((4, 8), np.float32))
        runtime = dict(input_size=4, xraydar_mean=0., xraydar_std=1.,
                       mask2former_input_mean=.5, mask2former_input_std=.5)
        xray, m2f, geometry = prepare_input(None, normalizer, runtime)
        self.assertEqual(tuple(xray.shape), (1, 4, 4))
        self.assertEqual(geometry['x0'], 2)
        np.testing.assert_allclose(m2f.numpy(), 1.)
        restored = restore_probability(np.ones((4, 4)), geometry, 4)
        expected = np.zeros((4, 8), np.float32)
        expected[:, 2:6] = 1
        np.testing.assert_array_equal(restored, expected)
        geometry = dict(original_size=(8, 4), resized_size=(8, 4), x0=0)
        restored = restore_probability(np.ones((4, 4)), geometry, 4)
        np.testing.assert_array_equal(restored[4:], 0)

    def test_positive_rescue(self):
        for probability, expected_kind, count in [
            (np.array([[.9, .1]]), 'none', 1),
            (np.array([[.2, .01]]), 'relative', 1),
            (np.zeros((2, 2)), 'peak', 1),
            (np.full((2, 2), np.nan), 'peak', 1),
        ]:
            with self.subTest(kind=expected_kind):
                mask, kind, _ = positive_mask(probability, .5, .5)
                self.assertEqual(kind, expected_kind)
                self.assertEqual(int(mask.sum()), count)

    def test_reference_dimensions(self):
        mask = np.ones((3, 5))
        self.assertEqual(mask_for_reference(mask, sitk.Image([5, 3], sitk.sitkUInt8)).shape,
                         (3, 5))
        self.assertEqual(mask_for_reference(mask, sitk.Image([5, 3, 1], sitk.sitkUInt8)).shape,
                         (1, 3, 5))
        with self.assertRaises(RuntimeError):
            mask_for_reference(mask, sitk.Image([5, 3, 2], sitk.sitkUInt8))


if __name__ == '__main__':
    unittest.main()
