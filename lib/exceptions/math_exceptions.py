class MathOperationException(Exception):
	def __init__(self):
		pass

	def __str__(self):
		pass


class MatrixException(MathOperationException):
	def __init__(self):
		pass

	def __str__(self):
		pass


class VectorException(MathOperationException):
	def __init__(self):
		pass

	def __str__(self):
		pass


class PointException(MathOperationException):
	def __init__(self):
		pass

	def __str__(self):
		pass


class VectorSpaceException(MathOperationException):
	def __init__(self):
		pass

	def __str__(self):
		pass


class CoordinateSystemException(MathOperationException):
	def __init__(self):
		pass

	def __str__(self):
		pass

class MatrixWrongSizeException(MatrixException):
	def __init__(self, n1=None, m1=None, n2=None, m2=None):
		if n1 and m1 and n2 and m2:
			self.message = f"Matrix 1 sizes is: {n1}x{m1} and Matrix 2 sizes is: {n2}x{m2}"
		else:
			self.message = "Incorrect sizes of Matrix"

	def __str__(self):
		return self.message


class MatrixNotSquareException(MatrixException):
	def __init__(self, *args):
		if args:
			self.message = args[0]
		else:
			self.message = "Matrix is not Square-form"

	def __str__(self):
		return self.message


class MatrixIncorrectOperationException(MatrixException):
	def __init__(self, *args):
		if args:
			self.message = args[0]
		else:
			self.message = "Incorrect operation"

	def __str__(self):
		return self.message


class VectorWrongSizeException(VectorException):
	def __init__(self, n1=None, n2=None):
		if n1 and n2:
			self.message = f"Vector 1 sizes is: {n1} and Matrix 2 sizes is: {n2}"
		else:
			self.message = "Incorrect sizes of Vectors"

	def __str__(self):
		return self.message


class VectorIncorrectOperationException(VectorException):
	def __init__(self, *args):
		if args:
			self.message = args[0]
		else:
			self.message = "Incorrect operation"

	def __str__(self):
		return self.message


class VectorSpaceIncorrectInitialization(VectorSpaceException):
	def __init__(self, *args):
		if args:
			self.message = args[0]
		else:
			self.message = "Incorrect initialization"

	def __str__(self):
		return self.message


class CoordinateSystemIncorrectInitialization(CoordinateSystemException):
	def __init__(self, *args):
		if args:
			self.message = args[0]
		else:
			self.message = "Incorrect initialization"

	def __str__(self):
		return self.message
