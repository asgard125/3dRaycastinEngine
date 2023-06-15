import copy
import math
from lib.exceptions import math_exceptions as me


class Matrix:

    PRECISION = 4

    """
    A class that implements a mathematical object matrix and operations on them.
    """
    def __init__(self, data: list):
        """Possible initialization options.

        Matrix(data: list[list[float | int]]).
        Creates a matrix with the specified elements of size NxM.
        """
        self.data = data
        self.n = len(data)
        self.m = len(data[0])

    def __str__(self) -> str:
        """
        Returns:
            The matrix in an easy-to-read form.

        """

        return '\n'.join([' | '.join([str(i) for i in self.data[row]]) for row in range(self.n)])

    def __getitem__(self, index: int) -> 'Vector':
        """
        Method to return a row of the matrix.
        Args:
            index (int): index of the required row of the matrix. Not less than 0 and higher than height - 1.

        Returns:
            Vector which is a row of a matrix

        """
        return Vector(self.data[index])

    def __eq__(self, other: 'Matrix') -> bool:
        """
        Method for comparing matrices.
        Args:
            other (Matrix): An arbitrary matrix

        Returns:
            true if matrices are equal and false if matrices are not equal.

        """
        precision = 10**(-Matrix.PRECISION - 1)
        if self.n != other.n or self.m != other.m:
            return False
        for i in range(self.n):
            for j in range(self.m):
                if math.fabs(self[i][j] - other[i][j]) > precision:
                    return False
        return True

    def __add__(self, other: 'Matrix') -> 'Matrix':
        """
        Method for calculating the sum of matrices.
        Args:
            other (Matrix): Matrix with the same height and width as the height and width of the matrix
            to which is added.

        Raises:
            MatrixWrongSizeException

        Returns:
            Matrix which is a sum of two matrices

        """
        if self.n != other.n or self.m != other.m:
            raise me.MatrixWrongSizeException(n1=self.n, m1=self.m, n2=other.n, m2=other.m)
        new_matrix_data = [[0 for i in range(self.m)] for i in range(self.n)]
        for i in range(self.n):
            for j in range(self.m):
                new_matrix_data[i][j] = round(self.data[i][j] + other.data[i][j], Matrix.PRECISION)
        return Matrix(new_matrix_data)

    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """
        Method for calculating the difference of matrices.
        Args:
            other (Matrix): Matrix with the same height and width as the height and width of the matrix from
            which is subtracted.

        Raises:
            MatrixWrongSizeException

        Returns:
            Matrix which is a difference of two matrices

        """
        return self + other * (-1)

    def __mul__(self, other) -> 'Matrix':
        """A method for calculating the product of matrices or a matrix and a number.
        Args:
            other (Matrix | int | float): Matrix the same height as the width of the matrix being multiplied
            or a number.

        Raises:
            MatrixWrongSizeException

        Returns:
            Matrix which is a product of two matrices if argument other is Matrix and Matrix
        which is a product of matrix and number if argument other is number.
        """
        if isinstance(other, Matrix | Vector):
            if self.m != other.n:
                raise me.MatrixWrongSizeException(n1=self.n, m1=self.m, n2=other.n, m2=other.m)
            new_matrix_data = [[0 for j in range(other.m)] for i in range(self.n)]
            for row in range(self.n):
                for col in range(other.m):
                    for k in range(other.n):
                        new_matrix_data[row][col] += round(self.data[row][k] * other.data[k][col], Matrix.PRECISION)
            return Matrix(new_matrix_data)
        if isinstance(other, int | float):
            new_matrix_data = self.data.copy()
            for row in range(self.n):
                for col in range(self.m):
                    new_matrix_data[row][col] = round(new_matrix_data[row][col] * other, Matrix.PRECISION)
            return Matrix(new_matrix_data)

    def __truediv__(self, other) -> 'Matrix':
        """Method for calculating the product of a matrix by the inverse or dividing a matrix by a number.

        Args:
            other (Matrix | int | float): Square non-singular matrix with the same height as the width of the matrix
            being multiplied or a non-zero number.

        Raises:
            MatrixWrongSizeException
            MatrixIncorrectOperationException
            MatrixNotSquareException

        Returns:
            Matrix which is a product of matrix and inverted matrix f argument other is Matrix and Matrix
        which is a product of matrix and inverse number if argument other is number.
        """
        if isinstance(other, Matrix):
            return self * ~other
        if isinstance(other, int | float):
            return self * (1 / other)

    def __invert__(self) -> 'Matrix':
        """Returns the inverse of the matrix given.
        Args:

        Raises:
            MatrixIncorrectOperationException
            MatrixNotSquareException

        Returns:
            Inverted matrix.
        """
        precision = 10e-4
        if self.m != self.n:
            raise me.MatrixNotSquareException
        if self.determinant() == 0:
            raise me.MatrixIncorrectOperationException('Matrix is singular')
        curr_matrix_copy = Matrix([[self.data[i][j] for j in range(self.m)] for i in range(self.n)])
        result_matrix = self.identity(n=self.n)

        indices = list(range(self.n))
        for fd in range(self.n):
            fdScaler = round(1.0 / curr_matrix_copy.data[fd][fd], 7)

            for j in range(self.n):
                curr_matrix_copy.data[fd][j] = round(curr_matrix_copy.data[fd][j] * fdScaler, 7)
                result_matrix.data[fd][j] = round(result_matrix.data[fd][j] * fdScaler, 7)

            for i in indices[0:fd] + indices[fd + 1:]:

                crScaler = curr_matrix_copy.data[i][fd]
                for j in range(self.n):
                    curr_matrix_copy.data[i][j] = round(
                        curr_matrix_copy.data[i][j] - crScaler * curr_matrix_copy.data[fd][j], 7)
                    result_matrix.data[i][j] = round(result_matrix.data[i][j] - crScaler * result_matrix.data[fd][j], 7)

        if self.matrix_equality(self.identity(self.n), self * result_matrix, precision):
            return result_matrix
        else:
            raise me.MatrixIncorrectOperationException("Matrix inverse out of tolerance.")

    def determinant(self) -> float:
        """Method for calculating the determinant of the matrix.
        Args:

        Raises:
            MatrixNotSquareException

        Returns:
            float which is the determinant of a matrix.
        """
        if self.m != self.n:
            raise me.MatrixNotSquareException
        new_matrix_data = [[self.data[i][j] for j in range(self.m)] for i in range(self.n)]

        for fd in range(self.n):
            if new_matrix_data[fd][fd] == 0:
                new_matrix_data[fd][fd] = 1.0e-18
            for i in range(fd + 1, self.n):
                crScaler = new_matrix_data[i][fd] / new_matrix_data[fd][fd]
                for j in range(self.n):
                    new_matrix_data[i][j] = new_matrix_data[i][j] - crScaler * new_matrix_data[fd][j]
        product = 1.0
        for i in range(self.n):
            product *= new_matrix_data[i][i]
        return round(product, 4)

    def transposed(self) -> 'Matrix':
        """Method for transposing a matrix.
        Args:

        Raises:

        Returns:
            Matrix with rearranged columns and rows and dimensions (NxM -> MxN).
        """
        new_matrix_data = [[self.data[j][i] for j in range(self.n)] for i in range(self.m)]
        return Matrix(new_matrix_data)

    @staticmethod
    def matrix_equality(A: 'Matrix', B: 'Matrix', precision: float) -> bool:
        """
        Method for comparing matrices to a given accuracy.

        Args:
            A (Matrix): an arbitrary matrix
            B (Matrix): an arbitrary matrix
            precision (float): required precision

        Returns:
            true if sizes of matrices are equal and the difference in modulus between two elements at the same
        positions is not more than the specified precision and false if not.

        """
        if A.m != B.m or A.n != B.n:
            return False
        for i in range(A.n):
            for j in range(B.m):
                if math.fabs(A.data[i][j] - B.data[i][j]) > precision:
                    return False
        return True

    @classmethod
    def identity(cls, n=1) -> 'Matrix':
        """Method for creating an identity matrix.
        Args:
            n (int): size of matrix. Can not be less than 1.

        Raises:
            MatrixIncorrectOperationException

        Returns:
            Square n-size Matrix with 1 on the main diagonal and with 0 on other positions
        """
        if n <= 0:
            raise me.MatrixIncorrectOperationException('identity matrix cannot be created with sizes <=0')
        ID_Matrix = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        return Matrix(ID_Matrix)

    @classmethod
    def gram(cls, basis: tuple | list) -> 'Matrix':
        """A method for creating a Gram matrix on a given basis.
        Args:
            basis (tuple | list): vectors with the same sizes that form basis

        Raises:
            MatrixIncorrectOperationException

        Returns:
            Gram matrix on given basis
        """
        if any([not isinstance(i, Vector) for i in basis]):
            raise me.MatrixIncorrectOperationException('the basis should consist of vectors')
        if len(basis) < 2:
            raise me.MatrixIncorrectOperationException('basis should contain at least one vector')
        if len(set([vector.n for vector in basis])) > 1:
            raise me.MatrixIncorrectOperationException('the basis vectors have different sizes')
        if Matrix([[v[i] for i in range(v.n)] for v in basis]).determinant() == 0:
            raise me.MatrixIncorrectOperationException('vectors do not form a basis')
        gram_data = []
        for i in range(len(basis)):
            scalar_mul_row = []
            for j in range(len(basis)):
                scalar_mul_row.append(basis[i] & basis[j])
            gram_data.append(scalar_mul_row)
        return Matrix(gram_data)

    @classmethod
    def bilinear_form(cls, M: 'Matrix', v1: 'Vector', v2: 'Vector') -> float:
        """Method for calculating a bilinear form with respect to two vectors.
        Args:
            M (Matrix): square Matrix NxN which cols form basis
            v1 (Vector): N-size vector
            v2 (Vector): N-size vector

        Raises:
            MatrixWrongSizeException
            MatrixNotSquareException

        Returns:
            a number that is a bilinear form with respect to two vectors.

        """
        if v1.n != M.n != v2.n:
            raise me.MatrixWrongSizeException
        if M.n != M.m:
            raise me.MatrixNotSquareException
        return (v1.transposed() * M * v2).data[0][0]

    @classmethod
    def get_rotation_matrix(cls, angle: int | float, n: int, axis1: int, axis2: int) -> 'Matrix':
        """Method for calculating the rotation matrix in a given surface (by indices) by a given angle in a space
            of a given dimension.
            Args:
                angle (int | float): angle of rotation
                n (int): dimension
                axis1 (int): first coordinate of a surface
                axis2 (int): second coordinate of a surface

            Raises:
                MatrixIncorrectOperationException

            Returns:
                a rotation Matrix in a given plane by a given angle in a space of a given dimension.
            """

        if (
                (n < 2 or axis1 < 0 or axis2 < 0)
                or (axis1 == axis2)
                or (axis1 >= n or axis2 >= n)
        ):
            raise me.MatrixIncorrectOperationException('surface axis should be different, and in [0, n]')

        mat = Matrix.identity(n)

        mat.data[axis1][axis1] = math.cos(angle)
        mat.data[axis2][axis2] = math.cos(angle)
        mat.data[axis1][axis2] = (-1) ** (axis1 + axis2) * math.sin(angle)
        mat.data[axis2][axis1] = (-1) ** (axis1 + axis2 + 1) * math.sin(angle)

        return mat

    @classmethod
    def get_teit_bryan_matrix(cls, angle1: int | float, angle2: int | float, angle3: int | float) -> 'Matrix':
        """Method for calculating the rotation matrix in 3-dimensional
            space using the Tate-Bryan transform for the given 3 angles around the axes.
            Args:
                angle1 (int | float): first angle of rotation
                angle2 (int | float): second angle of rotation
                angle3 (int | float): third angle of rotation

            Raises:


            Returns:
                a rotation Matrix in 3-dimensional
                space using the Tate-Bryan transform for the given 3 angles around the axes.
            """
        teit_bryan = Matrix.get_rotation_matrix(angle1, 3, 1, 2) * \
                     Matrix.get_rotation_matrix(angle2, 3, 2, 0) * \
                     Matrix.get_rotation_matrix(angle3, 3, 0, 1)

        return teit_bryan


class Vector(Matrix):
    """
    A class that implements a mathematical object vector and operations on them.
    """

    def __init__(self, data: list):
        """Possible initialization options.

            Vector(data: list[list[float | int]]).
            Vector(data: list[float | int]).
            Creates a vector with the specified size Nx1.
            """
        if not isinstance(data[0], list):
            data = [[data[i]] for i in range(len(data))]
        self.is_transposed = False
        super().__init__(data)

    def __eq__(self, other: 'Vector') -> bool:
        """
        Method for comparing matrices.
        Args:
            other (Vector): An arbitrary vector

        Returns:
            true if vectors are equal and false if vectors are not equal.

        """
        if self.n != other.n:
            return False
        for i in range(self.n):
            if self[i] != other[i]:
                return False
        return True

    def __getitem__(self, index: int) -> float | int:
        """
        Method to return an element of the vector.
        Args:
            index (int): index of the required element of the vector.

        Returns:
            number which is required elem of vector.

        """
        if self.is_transposed:
            return self.data[0][index]
        return self.data[index][0]

    def __pow__(self, other: 'Vector') -> 'Vector':
        """Method for to calculate the vector product in 3 dimensions.
        Args:
            other (Vector): Vector size 3.

        Raises:
            VectorWrongSizeException

        Returns:
            Vector size 3 that is vector product of two vectors.
        """
        if self.n != other.n:
            raise me.VectorWrongSizeException(self.n, other.n)
        if self.n != 3:
            raise me.VectorIncorrectOperationException('this operation is only for vectors dim=3')
        return Vector([self.data[1][0] * other.data[2][0] - self.data[2][0] * other.data[1][0],
                       self.data[2][0] * other.data[0][0] - self.data[0][0] * other.data[2][0],
                       self.data[0][0] * other.data[1][0] - self.data[1][0] * other.data[0][0]]
                      )

    def __and__(self, other: 'Vector') -> float | int:
        """Method for to calculate the scalar product.
        Args:
            other (Vector): Vector size as a multipliable vector.

        Raises:
            VectorWrongSizeException

        Returns:
            Number that is vector product of two vectors in CCS.
        """
        if self.n != other.n:
            raise me.VectorWrongSizeException(self.n, other.n)
        return round((self.transposed() * other)[0][0], 5)

    def __abs__(self) -> float | int:
        """Method for to calculate length of the vector.
        Args:

        Raises:

        Returns:
            Number that is a length of the vector
        """
        return round(math.sqrt(sum([self[i] * self[i] for i in range(self.n)])), 5)

    def __truediv__(self, other):
        if isinstance(other, int | float):
            return self.to_vector(super().__truediv__(other))

    def __mul__(self, other):
        if isinstance(other, int | float):
            return self.to_vector(super().__mul__(other))
        else:
            return super().__mul__(other)

    def normalize(self):
        if abs(self) == 0:
            return self
        return Vector([round(self.data[i][0] / abs(self), 5) for i in range(self.n)])

    @classmethod
    def to_vector(cls, data):
        return Vector([[i[0]] for i in data.data])

    def transposed(self) -> 'Vector':
        """Method for transposing a vector.
        Args:

        Raises:

        Returns:
            Vector with rearranged row to columns and dimensions (Nx1 -> 1xN).
        """
        new_vector_data = [[self.data[i][0] for i in range(self.n)]]
        new_vector = Vector(new_vector_data)
        new_vector.is_transposed = True
        return new_vector


class VectorSpace:
    """Vector space class."""

    def __init__(self, basis: tuple | list):
        """Possible initialization options.

        VectorSpace(v_1, ..., v_n)
        v_1, ..., v1 - objects of the Vector class, vectors forming the basis.

        Raises:
            VectorSpaceIncorrectInitialization
        """
        if any([not isinstance(i, Vector) for i in basis]):
            raise me.VectorSpaceIncorrectInitialization('the basis should consist of vectors')
        if len(basis) < 1:
            raise me.VectorSpaceIncorrectInitialization('basis should contain at least one vector')
        if any([vec.n != len(basis) for vec in basis]):
            raise me.VectorSpaceIncorrectInitialization('the basis vectors should have equal sizes')
        if Matrix([[v[i] for i in range(v.n)] for v in basis]).determinant() == 0:
            raise me.VectorSpaceIncorrectInitialization('vectors do not form a basis')
        self.basis = basis
        self.basis_dim = len(basis)

    def scalar_product(self, v1: 'Vector', v2: 'Vector') -> float | int:
        """Method to calculate the scalar product in a given basis.
        Args:
            v1 (Vector): Vector size as a multipliable vector.
            v2 (Vector): Vector size as a multipliable vector.

        Raises:
            VectorWrongSizeException

        Returns:
            Number that is scalar product of two vectors in a given basis.
        """
        return Matrix.bilinear_form(Matrix.gram(self.basis), v1, v2)

    def basis_decompose(self, point):
        """Method to decompose the coordinates of a point into basis vectors.
        Args:
            point (Point | Vector): point same size as basis vectors.

        Raises:
            VectorWrongSizeException

        Returns:
            Vector obtained by decomposing the coordinates of a point into basis vectors.
        """
        if isinstance(point, Point):
            return Point([point & basis_vec for basis_vec in self.basis])
        return Vector([point & basis_vec for basis_vec in self.basis])


class Point(Vector):
    """A class that implements a mathematical object point."""

    def __init__(self, data):
        """
        Possible initialization options.

        Point(data: list[list[float | int]]).
        Point(data: list[float | int]).
        Creates a point with the specified size elements Nx1.
        """
        super().__init__(data)

    def __add__(self, other: 'Vector') -> 'Point':
        """Method that implements transfering the point in space.
        Args:
            other (Vector): Vector same size as the point.

        Raises:
            VectorWrongSizeException

        Returns:
            Number that is scalar product of two vectors in a given basis.
        """
        if self.n != other.n:
            assert me.VectorWrongSizeException(self.n, other.n)
        return Point([self.data[i][0] + other[i] for i in range(self.n)])

    def __sub__(self, other: 'Vector') -> 'Point':
        """Method that implements transfering the point in space.
        Args:
            other (Vector): Vector same size as the point.

        Raises:
            VectorWrongSizeException

        Returns:
            Number that is scalar product of two vectors in a given basis.
        """
        if self.n != other.n:
            assert me.VectorWrongSizeException(self.n, other.n)
        return Point([self.data[i][0] - other[i] for i in range(self.n)])

    def __truediv__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __pow__(self, other):
        pass

    @classmethod
    def to_point(cls, data):
        return Point([i for i in data.data])


class CoordinateSystem:
    """A class that implements a coordinate system."""

    def __init__(self, basis: tuple | list, initial_pt: 'Point'):
        """Possible initialization options.

        VectorSpace(basis: list[Vector, ...] | tuple[Vector, ...], initial_pt: Point).
        Creates a vectorspace with specified basis and initial point.

        Raises:
            CoordinateSystemIncorrectInitialization
        """
        self.basis = VectorSpace([vec for vec in basis])
        if initial_pt.n != self.basis.basis_dim:
            raise me.CoordinateSystemIncorrectInitialization('initial point not in basis')
        self.initial_pt = initial_pt

    def get_dim_size(self):
        return self.basis.basis_dim
