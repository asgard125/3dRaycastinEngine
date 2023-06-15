import math

from lib.math.base_classes import Vector
import pytest
from lib.exceptions.math_exceptions import *


# Cравнение векторов
def test_vector_eqq_different_height():
    vec = Vector([[1], [3], [5]])
    vec1 = Vector([[1], [3]])
    assert vec != vec1


def test_vector_eqq_different_elem():
    vec = Vector([[1], [3], [5]])
    vec1 = Vector([[1], [3], [6]])
    assert vec != vec1


def test_vector_eqq_equal_elem():
    vec = Vector([[1], [3], [5]])
    vec1 = Vector([[1], [3], [5]])
    assert vec == vec1


# Векторное произведение
# ------------------------------------------------------------
def test_vector_vector_product_incorrect_size():
    vec = Vector([[1], [3], [5], [6]])
    vec1 = Vector([[1], [3], [5]])
    with pytest.raises(VectorWrongSizeException):
        vec**vec1


def test_vector_vector_product_incorrect_dim():
    vec = Vector([[1], [3], [5], [6]])
    vec1 = Vector([[1], [3], [5], [7]])
    with pytest.raises(VectorIncorrectOperationException):
        vec**vec1


def test_vector_vector_product_correct():
    vec = Vector([[1], [3], [5]])
    vec1 = Vector([[1], [3], [5]])
    result = vec**vec1
    expected_result = Vector([[0], [0], [0]])
    assert result == expected_result


# Скалярное произведение
# ------------------------------------------------------------
def test_vector_scalar_product_correct_size():
    vec = Vector([[1], [3], [5]])
    vec1 = Vector([[1], [3], [5]])
    result_vec = vec & vec1
    expected_result = 35
    assert result_vec == expected_result


def test_vector_scalar_product_incorrect_size():
    vec = Vector([[1], [3], [5]])
    vec1 = Vector([[1], [3], [5], [6]])
    with pytest.raises(VectorWrongSizeException):
        vec & vec1


# Длина вектора
# ------------------------------------------------------------
def test_vector_length():
    vec = Vector([[1], [3], [5]])
    result_vec = abs(vec)
    expected_result = math.sqrt(1*1+3*3+5*5)
    assert result_vec == expected_result


# Транспонирование вектора
# ------------------------------------------------------------
def test_matrix_transpose_correct():
    vec = Vector([1, 2, 4])
    expected_vec = Vector([[1, 2, 4]])
    res_vec = vec.transposed()
    assert res_vec == expected_vec


def test_matrix_transpose_incorrect():
    vec = Vector([1, 2, 4])
    expected_vec = Vector([[1], [2], [4]])
    res_vec = vec.transposed()
    assert not res_vec == expected_vec