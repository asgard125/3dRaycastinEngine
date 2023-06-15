import math

from lib.math.base_classes import Matrix, Vector
import pytest
from lib.exceptions.math_exceptions import *


# Сравнение матриц
# ------------------------------------------------------------
def test_matrix_eqq_different_width():
    mat = Matrix([[1, 2, 2], [3, 4, 4], [5, 6, 6]])
    mat1 = Matrix([[1, 2], [3, 4], [5, 6]])
    assert mat != mat1


def test_matrix_eqq_different_height():
    mat = Matrix([[1, 2], [3, 4], [5, 6]])
    mat1 = Matrix([[1, 2], [3, 4]])
    assert mat != mat1


def test_matrix_eqq_different_elem():
    mat = Matrix([[1, 2], [3, 4], [5, 6]])
    mat1 = Matrix([[1, 2], [3, 5], [5, 6]])
    assert mat != mat1


def test_matrix_eqq_equal_elem():
    mat = Matrix([[1, 2], [3, 4], [5, 6]])
    mat1 = Matrix([[1, 2], [3, 4], [5, 6]])
    assert mat == mat1


# Сложение матриц
# ------------------------------------------------------------
def test_matrix_add_same_size():
    mat = Matrix([[1, 2], [3, 4], [5, 6]])
    mat1 = Matrix([[1, 1], [2, 2], [3, 3]])
    res_mat = Matrix([[2, 3], [5, 6], [8, 9]])
    assert mat + mat1 == res_mat


def test_matrix_add_different_size_height():
    mat = Matrix([[1, 2], [3, 4]])
    mat1 = Matrix([[1, 1], [2, 2], [3, 3]])
    with pytest.raises(MatrixWrongSizeException):
        mat + mat1


def test_matrix_add_different_size_width():
    mat = Matrix([[1, 2, 1], [3, 4, 2], [5, 6, 3]])
    mat1 = Matrix([[1, 1], [2, 2], [3, 3]])
    with pytest.raises(MatrixWrongSizeException):
        mat + mat1


def test_matrix_add_different_size_height_width():
    mat = Matrix([[1, 2, 1], [3, 4, 2]])
    mat1 = Matrix([[1, 1], [2, 2], [3, 3]])
    with pytest.raises(MatrixWrongSizeException):
        mat + mat1


def test_matrix_add_number():
    mat = Matrix([[1, 2, 1], [3, 4, 2]])
    with pytest.raises(AttributeError):
        mat + 1


# Вычитание матриц
# ------------------------------------------------------------
def test_matrix_sub_same_size():
    mat = Matrix([[1, 2], [3, 4], [5, 6]])
    mat1 = Matrix([[1, 1], [2, 2], [3, 3]])
    res_mat = Matrix([[0, 1], [1, 2], [2, 3]])
    assert mat - mat1 == res_mat


def test_matrix_sub_different_size_height():
    mat = Matrix([[1, 2], [3, 4]])
    mat1 = Matrix([[1, 1], [2, 2], [3, 3]])
    with pytest.raises(MatrixWrongSizeException):
        mat - mat1


def test_matrix_sub_different_size_width():
    mat = Matrix([[1, 2, 1], [3, 4, 2], [5, 6, 3]])
    mat1 = Matrix([[1, 1], [2, 2], [3, 3]])
    with pytest.raises(MatrixWrongSizeException):
        mat - mat1


def test_matrix_sub_different_size_height_width():
    mat = Matrix([[1, 2, 1], [3, 4, 2]])
    mat1 = Matrix([[1, 1], [2, 2], [3, 3]])
    with pytest.raises(MatrixWrongSizeException):
        mat - mat1


def test_matrix_sub_number():
    mat = Matrix([[1, 2, 1], [3, 4, 2]])
    with pytest.raises(AttributeError):
        mat - 1


# Умножение матрицы на матрицу
# -----------------------------------------------------------
def test_matrix_mul_different_size_height_width_correct():
    mat = Matrix([[1, 2, 1], [3, 4, 2]])
    mat1 = Matrix([[1, 1], [2, 2], [3, 3]])
    res_mat = Matrix([[8, 8], [17, 17]])
    assert mat * mat1 == res_mat


def test_matrix_mul_different_size_height_width_incorrect():
    mat = Matrix([[1, 2, 1, 5], [3, 4, 2, 5]])
    mat1 = Matrix([[1, 1], [2, 2], [3, 3]])
    with pytest.raises(MatrixWrongSizeException):
        mat * mat1


# Умножение матрицы и числа
# ------------------------------------------------------------
def test_matrix_mul_int():
    mat = Matrix([[1, 2, 1], [3, 4, 2]])
    res_mat = Matrix([[2, 4, 2], [6, 8, 4]])
    assert mat * 2 == res_mat


def test_matrix_mul_float():
    mat = Matrix([[1, 2, 1], [3, 4, 2]])
    res_mat = Matrix([[2, 4, 2], [6, 8, 4]])
    assert mat * 2.0 == res_mat


# Вычисление определителя
# ------------------------------------------------------------
def test_determinant_non_square_width():
    mat = Matrix([[1, 2, 1], [3, 4, 2]])
    with pytest.raises(MatrixNotSquareException):
        mat.determinant()


def test_determinant_non_square_height():
    mat = Matrix([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(MatrixNotSquareException):
        mat.determinant()


def test_determinant_zero():
    mat = Matrix([[5, 5, 5, 5], [35, 35, 35, 35], [70, 70, 70, 70], [210, 210, 210, 210]])
    assert mat.determinant() == 0


def test_determinant_positive():
    mat = Matrix([[1, 2, 1, 3], [3, 2, 4, 4], [6, 7, 6, 5], [4, 8, 0, 2]])
    assert mat.determinant() == 58


def test_determinant_negative():
    mat = Matrix([[6, 2, 1, 3], [3, 2, 4, 4], [6, 7, 6, 5], [4, 8, 0, 2]])
    assert mat.determinant() == -262


# Вычисление обратной матрицы
# ------------------------------------------------------------
def test_invert_non_square_width():
    mat = Matrix([[1, 2, 1], [3, 4, 2]])
    with pytest.raises(MatrixNotSquareException):
        ~mat


def test_invert_non_square_height():
    mat = Matrix([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(MatrixNotSquareException):
        ~mat


def test_invert_singular():
    mat = Matrix([[5, 5, 5, 5], [35, 35, 35, 35], [70, 70, 70, 70], [210, 210, 210, 210]])
    with pytest.raises(MatrixIncorrectOperationException):
        ~mat


def test_invert_not_singular():
    mat = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [6, 7, 6, 5], [4, 8, 0, 2]])
    inverted_mat = Matrix([[-1.625, 1.125, -0.5, 0], [0.95, -0.75, 0.4, 0.1], [0.975, -0.875, 0.7, -0.2], [-0.55, 0.75, -0.6, 0.1]])
    assert ~mat == inverted_mat


# Деление матрицы на матрицу
# ------------------------------------------------------------
def test_matrix_truediv_not_square_matrix():
    mat = Matrix([[1, 2], [3, 4]])
    mat1 = Matrix([[1, 1], [2, 2], [3, 3]])
    with pytest.raises(MatrixNotSquareException):
        mat / mat1


def test_matrix_truediv_singular():
    mat = Matrix([[1, 3], [3, 4]])
    mat1 = Matrix([[1, 1], [2, 2]])
    with pytest.raises(MatrixIncorrectOperationException):
        mat / mat1


def test_matrix_truediv_different_sizes():
    mat = Matrix([[1, 2], [3, 4]])
    mat1 = Matrix([[3, 1, 2], [3, 2, 3], [2, 3, 4]])
    with pytest.raises(MatrixWrongSizeException):
        mat / mat1


def test_matrix_truediv_same_sizes():
    mat = Matrix([[1, 2, 7], [3, 4, 3]])
    mat1 = Matrix([[3, 1, 2], [3, 2, 3], [2, 3, 4]])
    result = mat / mat1
    expected_result = Matrix([[22, -31, 14], [-12, 17, -6]])
    assert result == expected_result


# Деление матрицы на число
# ------------------------------------------------------------
def test_matrix_truediv_int():
    mat = Matrix([[1, 2, 1], [3, 4, 2]])
    expected_mat = Matrix([[0.5, 1, 0.5], [1.5, 2, 1]])
    result = mat / 2
    assert result == expected_mat


def test_matrix_truediv_float():
    mat = Matrix([[1, 2, 1], [3, 4, 2]])
    expected_mat = Matrix([[0.5, 1, 0.5], [1.5, 2, 1]])
    result = mat / 2.0
    assert result == expected_mat


# Транспонирование матрицы
# ------------------------------------------------------------
def test_matrix_transpose_non_square():
    mat = Matrix([[1, 2, 1], [3, 4, 2]])
    res_mat = Matrix([[1, 3], [2, 4], [1, 2]])
    assert mat.transposed() == res_mat


def test_matrix_transpose_square():
    mat = Matrix([[1, 2], [3, 4]])
    res_mat = Matrix([[1, 3], [2, 4]])
    assert mat.transposed() == res_mat


def test_matrix_transpose_one_one():
    mat = Matrix([[1]])
    res_mat = Matrix([[1]])
    assert mat.transposed() == res_mat


# Сравнение двух матриц по заданной точности
# ------------------------------------------------------------
def test_matrix_equality_different_width():
    mat = Matrix([[1, 2, 1], [3, 4, 2]])
    res_mat = Matrix([[1, 2], [3, 4]])
    precision = 10e-5
    assert not Matrix.matrix_equality(mat, res_mat, precision)


def test_matrix_equality_different_height():
    mat = Matrix([[1, 2], [3, 4]])
    res_mat = Matrix([[1, 2], [3, 4], [5, 6]])
    precision = 10e-5
    assert not Matrix.matrix_equality(mat, res_mat, precision)


def test_matrix_equality_incorrect():
    mat = Matrix([[1, 2], [3, 4]])
    res_mat = Matrix([[1.003, 2.002], [3.001, 4]])
    precision = 10e-5
    assert not Matrix.matrix_equality(mat, res_mat, precision)


def test_matrix_equality_correct():
    mat = Matrix([[1, 2], [3, 4]])
    res_mat = Matrix([[1.000003, 2.000002], [3.000001, 4]])
    precision = 10e-5
    assert Matrix.matrix_equality(mat, res_mat, precision)


# Создание единичной матрицы
# ------------------------------------------------------------
def test_matrix_identity_negative_size():
    with pytest.raises(MatrixIncorrectOperationException):
        Matrix.identity(-10)


def test_matrix_identity_zero_size():
    with pytest.raises(MatrixIncorrectOperationException):
        Matrix.identity(0)


def test_matrix_identity_one_size():
    mat = Matrix.identity(1)
    res_mat = Matrix([[1]])
    assert mat == res_mat


def test_matrix_identity_postitve_size():
    mat = Matrix.identity(3)
    res_mat = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert mat == res_mat


# Создание матрицы Грама
# ------------------------------------------------------------
def test_matrix_gram_different_not_vector_basis():
    basis_vec1 = Vector([[1], [2], [3]])
    basis_vec2 = Matrix([[1], [2], [3]])
    basis_vec3 = Vector([[1], [2], [3]])
    basis = [basis_vec1, basis_vec2, basis_vec3]
    with pytest.raises(MatrixIncorrectOperationException):
        Matrix.gram(basis)


def test_matrix_gram_different_basis_vectors_sizes():
    basis_vec1 = Vector([[1], [2], [3]])
    basis_vec2 = Vector([[1], [2], [3], [4]])
    basis_vec3 = Vector([[1], [2], [3]])
    basis = [basis_vec1, basis_vec2, basis_vec3]
    with pytest.raises(MatrixIncorrectOperationException):
        Matrix.gram(basis)


def test_matrix_gram_zero_vector():
    basis = []
    with pytest.raises(MatrixIncorrectOperationException):
        Matrix.gram(basis)


def test_matrix_gram_one_vector():
    basis_vec1 = Vector([[1]])
    basis = [basis_vec1]
    with pytest.raises(MatrixIncorrectOperationException):
        Matrix.gram(basis)


def test_matrix_gram_basis():
    basis_vec1 = Vector([[1], [0], [0]])
    basis_vec2 = Vector([[0], [1], [0]])
    basis_vec3 = Vector([[0], [0], [1]])
    basis = [basis_vec1, basis_vec2, basis_vec3]
    result = Matrix.gram(basis)
    expected_mat = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert result == expected_mat


def test_matrix_gram__not_basis():
    basis_vec1 = Vector([[1], [0], [0]])
    basis_vec2 = Vector([[1], [1], [1]])
    basis_vec3 = Vector([[1], [1], [1]])
    basis = [basis_vec1, basis_vec2, basis_vec3]
    with pytest.raises(MatrixIncorrectOperationException):
        Matrix.gram(basis)


# Билинейная форма
# ------------------------------------------------------------
def test_matrix_bilinear_form_different_size_vector_one():
    vec1 = Vector([[1], [2], [3], [4]])
    vec2 = Vector([[4], [5], [6]])
    mat = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(MatrixWrongSizeException):
        Matrix.bilinear_form(mat, vec1, vec2)


def test_matrix_bilinear_form_different_size_vector_two():
    vec1 = Vector([[1], [2], [3]])
    vec2 = Vector([[4], [5], [6], [7]])
    mat = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(MatrixWrongSizeException):
        Matrix.bilinear_form(mat, vec1, vec2)


def test_matrix_bilinear_form_different_size_matrix():
    vec1 = Vector([[1], [2], [3]])
    vec2 = Vector([[4], [5], [6], [7]])
    mat = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    with pytest.raises(MatrixWrongSizeException):
        Matrix.bilinear_form(mat, vec1, vec2)


def test_matrix_bilinear_form_matrix_not_square():
    vec1 = Vector([[1], [2], [3]])
    vec2 = Vector([[4], [5], [6]])
    mat = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    with pytest.raises(MatrixNotSquareException):
        Matrix.bilinear_form(mat, vec1, vec2)


def test_matrix_bilinear_form():
    vec1 = Vector([[1], [2], [3]])
    vec2 = Vector([[4], [5], [6]])
    mat = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    result = Matrix.bilinear_form(mat, vec1, vec2)
    expected_result = 32
    assert result == expected_result


# матрица поворота
# ------------------------------------------------------------
def test_rotation_matrix_equal_axis():
    with pytest.raises(MatrixIncorrectOperationException):
        Matrix.get_rotation_matrix(50, 5, 1, 1)


def test_rotation_matrix_less_zero_axis_1():
    with pytest.raises(MatrixIncorrectOperationException):
        Matrix.get_rotation_matrix(50, 5, -1, 1)


def test_rotation_matrix_less_zero_axis_2():
    with pytest.raises(MatrixIncorrectOperationException):
        Matrix.get_rotation_matrix(50, 5, 1, -1)


def test_rotation_matrix_higher_than_n_axis_1():
    with pytest.raises(MatrixIncorrectOperationException):
        Matrix.get_rotation_matrix(50, 5, 6, 1)


def test_rotation_matrix_higher_than_n_axis_2():
    with pytest.raises(MatrixIncorrectOperationException):
        Matrix.get_rotation_matrix(50, 5, 1, 6)


def test_rotation_matrix_equal():
    result = Matrix.get_rotation_matrix(math.pi/2, 3, 1, 2)
    expected_result = Matrix([[1, 0, 0],
                                  [0, math.cos(math.pi/2), -math.sin(math.pi/2)],
                                  [0, math.sin(math.pi/2), math.cos(math.pi/2)]])
    assert result == expected_result