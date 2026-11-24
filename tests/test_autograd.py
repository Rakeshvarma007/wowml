"""Tests for the autograd engine."""
import pytest


class TestAutogradBasic:
    def test_add_backward(self):        pass
    def test_mul_backward(self):        pass
    def test_chain_rule(self):          pass

class TestAutogradMatmul:
    def test_matmul_backward(self):     pass

class TestAutogradActivations:
    def test_relu_backward(self):       pass
    def test_sigmoid_backward(self):    pass

class TestAutogradNumericalGradient:
    def test_numerical_vs_analytical(self): pass
