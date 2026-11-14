"""
Layer testing module

Tests the parameter access interface (get__, set__ and reset__ methods) of
the Gaussian mixture modeling layers, for every parametrization.
"""

from typing import Self
from unittest import TestCase

import torch

import gmml.layer as layer
import gmml.validation as val

NUM_COMPONENTS = 5
DIM = 3

LAYERS_FACTORS_SHAPES = [
    (layer.FullGMMLayer, (NUM_COMPONENTS, DIM, DIM)),
    (layer.DiagGMMLayer, (NUM_COMPONENTS, DIM)),
    (layer.IsoGMMLayer, (NUM_COMPONENTS,)),
    (layer.SharedIsoGMMLayer, (1,)),
]


class TestGetFactors(TestCase):
    """
    'get__factors' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Returns a parameter of the shape specific to each layer.
        """

        for layer_cls, shape in LAYERS_FACTORS_SHAPES:
            with self.subTest(layer=layer_cls.__name__):
                gmml = layer_cls(NUM_COMPONENTS, DIM)
                _factors = gmml.get__factors()

                self.assertIsInstance(_factors, torch.nn.Parameter)
                self.assertEqual(tuple(_factors.shape), shape)

    def test_1(self: Self) -> None:
        """
        Returns a registered parameter, also exposed by 'factors'.
        """

        for layer_cls, _ in LAYERS_FACTORS_SHAPES:
            with self.subTest(layer=layer_cls.__name__):
                gmml = layer_cls(NUM_COMPONENTS, DIM)
                _factors = gmml.get__factors()

                self.assertTrue(any(p is _factors for p in gmml.parameters()))
                self.assertIs(gmml.factors, _factors)


class TestSetFactors(TestCase):
    """
    'set__factors' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Updates the parameter in place.
        """

        for layer_cls, shape in LAYERS_FACTORS_SHAPES:
            with self.subTest(layer=layer_cls.__name__):
                gmml = layer_cls(NUM_COMPONENTS, DIM)
                _factors = gmml.get__factors()
                new__factors = torch.randn(shape)
                gmml.set__factors(new__factors)

                self.assertIs(gmml.get__factors(), _factors)
                self.assertTrue(torch.equal(_factors, new__factors))

    def test_1(self: Self) -> None:
        """
        Rejects tensors of the wrong shape.
        """

        for layer_cls, shape in LAYERS_FACTORS_SHAPES:
            with self.subTest(layer=layer_cls.__name__):
                gmml = layer_cls(NUM_COMPONENTS, DIM)

                with self.assertRaises(val.HasUnexpectedShape):
                    gmml.set__factors(torch.randn(*shape, 2))


class TestResetFactors(TestCase):
    """
    'reset__factors' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Resets the parameter to zero, with identity Cholesky factors.
        """

        for layer_cls, shape in LAYERS_FACTORS_SHAPES:
            with self.subTest(layer=layer_cls.__name__):
                gmml = layer_cls(NUM_COMPONENTS, DIM)
                gmml.set__factors(torch.randn(shape))
                gmml.reset__factors()

                self.assertTrue((gmml.get__factors() == 0.0).all())
                self.assertTrue(
                    torch.allclose(
                        gmml.full_factors,
                        torch.eye(DIM).expand(NUM_COMPONENTS, DIM, DIM),
                    )
                )


class TestSetMeans(TestCase):
    """
    'set__means' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Updates the parameter in place.
        """

        gmml = layer.FullGMMLayer(NUM_COMPONENTS, DIM)
        _means = gmml.get__means()
        new__means = torch.randn(NUM_COMPONENTS, DIM)
        gmml.set__means(new__means)

        self.assertIs(gmml.get__means(), _means)
        self.assertTrue(torch.equal(gmml.means, new__means))

    def test_1(self: Self) -> None:
        """
        Rejects tensors of the wrong shape.
        """

        gmml = layer.FullGMMLayer(NUM_COMPONENTS, DIM)

        with self.assertRaises(val.HasUnexpectedShape):
            gmml.set__means(torch.randn(DIM, NUM_COMPONENTS))


class TestSetLogits(TestCase):
    """
    'set__logits' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Updates the parameter in place.
        """

        gmml = layer.FullGMMLayer(NUM_COMPONENTS, DIM)
        _logits = gmml.get__logits()
        new__logits = torch.randn(NUM_COMPONENTS)
        gmml.set__logits(new__logits)

        self.assertIs(gmml.get__logits(), _logits)
        self.assertTrue(torch.equal(gmml.logits, new__logits))

    def test_1(self: Self) -> None:
        """
        Rejects tensors of the wrong shape.
        """

        gmml = layer.FullGMMLayer(NUM_COMPONENTS, DIM)

        with self.assertRaises(val.HasUnexpectedShape):
            gmml.set__logits(torch.randn(NUM_COMPONENTS + 1))

    def test_2(self: Self) -> None:
        """
        Rejects any update of an equi-weighted codebook.
        """

        gmml = layer.FullGMMLayer(NUM_COMPONENTS, DIM, equi_weighted=True)

        with self.assertRaises(val.IsEquiWeighted):
            gmml.set__logits(torch.randn(NUM_COMPONENTS))

        self.assertTrue((gmml.get__logits() == 0.0).all())


class TestResetParameters(TestCase):
    """
    'reset_parameters' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Resets the means, logits and Cholesky factors parameters.
        """

        for layer_cls, shape in LAYERS_FACTORS_SHAPES:
            with self.subTest(layer=layer_cls.__name__):
                gmml = layer_cls(NUM_COMPONENTS, DIM)
                old__means = gmml.get__means().detach().clone()
                gmml.set__logits(torch.randn(NUM_COMPONENTS))
                gmml.set__factors(torch.randn(shape))
                gmml.reset_parameters()

                self.assertFalse(torch.equal(gmml.get__means(), old__means))
                self.assertTrue((gmml.get__logits() == 0.0).all())
                self.assertTrue((gmml.get__factors() == 0.0).all())


class TestForward(TestCase):
    """
    'forward' unit testing
    """

    def test_0(self: Self) -> None:
        """
        Propagates gradients to every parameter.
        """

        for layer_cls, _ in LAYERS_FACTORS_SHAPES:
            with self.subTest(layer=layer_cls.__name__):
                gmml = layer_cls(NUM_COMPONENTS, DIM)
                inpt = torch.randn(4, DIM)
                gmml.get_log_likelihoods(inpt).sum().backward()

                for _param in [
                    gmml.get__means(),
                    gmml.get__logits(),
                    gmml.get__factors(),
                ]:
                    self.assertIsNotNone(_param.grad)
                    self.assertTrue(_param.grad.abs().sum() > 0.0)
