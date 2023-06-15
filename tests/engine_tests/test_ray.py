from lib.math.base_classes import Matrix


# Сравнение матриц
# ------------------------------------------------------------
def test_matrix_eqq_different_width():
    mat = Matrix([[1, 2, 2], [3, 4, 4], [5, 6, 6]])
    mat1 = Matrix([[1, 2], [3, 4], [5, 6]])
    assert mat != mat1