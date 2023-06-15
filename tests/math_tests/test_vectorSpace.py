from lib.math.base_classes import Vector, VectorSpace
import pytest
from lib.exceptions.math_exceptions import *


# Инициализация
def test_vectorspace_vectors_different_length():
    vec1 = Vector([1, 0, 0])
    vec2 = Vector([0, 1])
    vec3 = Vector([1, 0, 1])
    with pytest.raises(VectorSpaceIncorrectInitialization):
        VectorSpace([vec1, vec2, vec3])


def test_vectorspace_vectors_not_basis_by_len():
    vec1 = Vector([1, 0, 0, 0])
    vec2 = Vector([0, 1, 0, 0])
    vec3 = Vector([1, 0, 1, 0])
    with pytest.raises(VectorSpaceIncorrectInitialization):
        VectorSpace([vec1, vec2, vec3])


def test_vectorspace_vectors_not_basis_by_elem():
    vec1 = Vector([1, 1, 1])
    vec2 = Vector([0, 1, 0])
    vec3 = Vector([1, 1, 1])
    with pytest.raises(VectorSpaceIncorrectInitialization):
        VectorSpace([vec1, vec2, vec3])


def test_vectorspace_vectors_basis():
    vec1 = Vector([1, 0, 0])
    vec2 = Vector([0, 1, 0])
    vec3 = Vector([0, 0, 1])
    assert VectorSpace([vec1, vec2, vec3])