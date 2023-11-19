
import csv, os, glob
import copy
from fractions import Fraction
from sympy import *

DEBUG = False
import inspect

def debug_print(message):
	line_number = inspect.currentframe().f_back.f_lineno
	if DEBUG:
		print(f"[Line {line_number}] - {message}")

def get_name(default):
	name = input(f"Use: {default} ('1' to except)?\nOr Enter New Name: ")
	if name == '1':
		return default
	else:
		return name

# CLEAR SCREEN
def clear_screen():
	if os.name == 'posix':  # For UNIX or Linux or MacOS
		os.system('clear')
	elif os.name == 'nt':  # For Windows
		os.system('cls')

def format_item(item, as_decimal=False):
	return pretty(item)

def format_expression(expr, as_decimal=False):
	if expr.free_symbols:
		return " + ".join([f"{format_item(coeff, as_decimal)}*{var}" for var, coeff in expr.as_coefficients_dict().items()])
	else:
		return format_item(expr, as_decimal)


class Matrix_Calc:
	def __init__(self):
		self.matrices = [] 
		self.precision = 2
		self.mode = 'fraction'

	# File handling methods
	# # # #	# # # #	# # # #	

	def load_from_file(self, overwrite=True):
		folder_path = './matrix_files/'
		if not os.path.exists(folder_path):
			os.mkdir(folder_path)

		files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.txt')]

		if files == []:
			print(f"No saved files found")
			return
		files = sorted(files)

		while True:

			clear_screen()

			print("\nAvailable files:")
			for i, file in enumerate(files):
				print(f"{i + 1}: {file}")

		
			file_choice = input("Enter the number of the file you want to load or 'x' to cancel: ")
			if file_choice == 'x':
				return False
			elif file_choice.isdigit() and int(file_choice) - 1 in range(len(files)):
				break
			else:
				print("Invalid selection. Please try again.")

		file_name = os.path.join(folder_path, files[int(file_choice) - 1])
		
		if not self.matrices:
			overwrite = False
		else:
			invalid = False
			while overwrite:

				clear_screen()

				print("\nCurrent matrices:")
				for i, matrix in enumerate(self.matrices):
					print(f"{i + 1}: {matrix[0]}")

				if invalid:
					print("Invalid selection. Please try again.")
					invalid = False

				matrix_choice = input("Enter the number of the matrix to overwrite, 'n' to add a new matrix, or 'x' to cancel: ")
				if matrix_choice == 'x':
					return False
				elif matrix_choice.isdigit() and int(matrix_choice) - 1 in range(len(self.matrices)):
					overwrite = True
					break
				elif matrix_choice.lower() == 'n':
					overwrite = False
					break
				else:
					invalid = True
					

		with open(file_name, 'r') as f:
			reader = csv.reader(f)
			data = [[sympify(i) for i in row] for row in reader]
			data = Matrix(data)

		if overwrite:
			self.matrices[int(matrix_choice) - 1] = [file_name.replace(folder_path,''), data]
		else:
			self.matrices.append([file_name.replace(folder_path,''), data])
		return True

	def save_to_file(self):

		invalid = False
		while True:
			clear_screen()

			for i, (file_name, matrix) in enumerate(self.matrices):
				print(f"{i + 1}) loaded from '{file_name}'")

			if invalid:
				print(f"Invalid matrix choice. Please choose a number between 1 and {len(self.matrices)}.")
				invalid = False

			matrix_choice = input("Please choose a matrix to save or 'x' to cancel: ")
			if matrix_choice == 'x':
				return
			elif matrix_choice.isdigit() and int(matrix_choice)-1 in range(len(self.matrices)):
				break
			else:
				invalid = True

		matrix_choice = int(matrix_choice) - 1

		file_name = input("Please enter the name for the new file (without .txt) or 'x' to cancel: ")
		if file_name == 'x':
			return

		file_name = './matrix_files/' + file_name

		with open(file_name + '.txt', 'w', newline='') as f:
			writer = csv.writer(f)
			matrix = self.matrices[matrix_choice][1]
			for row in matrix.tolist():
				writer.writerow(row)

	def delete_files(self):
		directory = "matrix_files"
		
		invalid = False

		while True:
			clear_screen()

			files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.txt')]
			
			if not files:
				print("No files available for deletion.")
				return

			print("\nAvailable files:")
			for i, file in enumerate(files):
				print(f"{i + 1}: {file}")

			if invalid:
				print("Invalid selection. Please try again.")
				invalid = False

			file_choice = input("Enter the number of the file you want to delete or 'x' to cancel: ")
			
			if file_choice == 'x':
				return
			elif file_choice.isdigit() and int(file_choice) - 1 in range(len(files)):
				file_name = files[int(file_choice) - 1]
				os.remove(os.path.join(directory, file_name))
				print(f"Deleted file: {file_name}")
			else:
				invalid = True


	# Create and Delete Matrices
	# # # #	# # # #	# # # #	# # 

	def delete_matrix(self):
		invalid = False
		while True:

			clear_screen()

			print("\nPlease choose a matrix to delete:")
			self.print_heading()

			if invalid:
				print(f"Invalid matrix choice. Please choose a number between 1 and {len(self.matrices)}.")
				invalid = False
				
			matrix_choice = input("\nEnter your choice ('c' to clear all, 'x' to go back): ")
			if matrix_choice.lower() == 'x':
				return False
			if matrix_choice.lower() == 'c':
				self.matrices = []
				return True
			if matrix_choice.isdigit() and int(matrix_choice) - 1 in range(len(self.matrices)):
				del self.matrices[int(matrix_choice) - 1]
				print("Matrix deleted successfully.")
				return False
			else:
				invalid = True

	def create_matrix(self):
		invalid = False
		while True:

			clear_screen()
			if invalid:
				print("Invalid input. The name cannot be an empty string. Please try again.")
				invalid = False
			name = input("Enter a name for the new matrix: ").strip()  # strip() removes leading/trailing white space
			if name:  # if the name is not an empty string
				break
			else:
				invalid = True

		# Request and validate the width of the matrix
		invalid = False
		while True:
			clear_screen()
			if invalid:
				print("Invalid input. Please enter a positive integer.")
				invalid = False
			width = input("Enter the width of the matrix ('x' to go back): ")
			if width.lower() == 'x':
				return False
			if width.isdigit() and int(width) > 0:
				width = int(width)
				break
			else:
				invalid = True
				

		# Request and validate the height of the matrix
		invalid = False
		while True:
			clear_screen()
			if invalid:
				print("Invalid input. Please enter a positive integer.")
				invalid = False
			height = input("Enter the height of the matrix ('x' to go back): ")
			if height.lower() == 'x':
				return False
			if height.isdigit() and int(height) > 0:
				height = int(height)
				break
			else:
				invalid = True

		invalid = False
		while True:
			clear_screen()

			if invalid:
				print("Invalid selection.")
				invalid = False

			print("Matrix Options")
			if width == height:
				print("I) identity matrix")
			matrix_type = input("U) unit matrix\nV) variable matrix\n Choose options ('x' to go back)? ")
			if matrix_type.lower() == 'x':
				return False
			if matrix_type.lower() == 'i' and width == height:
				matrix = Matrix.eye(width)
			elif matrix_type.lower() == 'u':
				matrix = Matrix.ones(height, width)
			elif matrix_type.lower() == 'v':
				matrix = self.create_variable_matrix(width, height)
			else:
				invalid = True
				
			if matrix is None:
				return False
			else:
				self.matrices.append([name, matrix])
				pprint(self.matrices[-1][1])
				return True

	def create_variable_matrix(self, width, height):
		matrix = []

		invalid = False
		while True:

			clear_screen()
			if invalid:
				print("Invalid input. Please enter 1, 2 or 3.")
				invalid = False

			print("Matrix variable options:")
			print("1) Variables unique to each column.")
			print("2) All variables unique (a,b,c...).")
			print("3) All variables unique (double index).")
			option = input("Choose an option ('x' to go back): ")

			if option.lower() == 'x':
				return None
			else:
				for i in range(height):
					row = []
					for j in range(width):
						if option == '1':
							# Variables are unique to each column
							row.append(symbols(f'x_{j+1}'))
						elif option == '2':
							# All variables are unique a,b,c 
							row.append(symbols(f'{chr(97+i)}_{j+1}'))
						elif option == '3':
							# All variables are unique double index
							row.append(symbols(f'a_{i+1}{j+1}'))
						else:
							invalid = True
							continue
					matrix.append(row)
				break
			

		return Matrix(matrix)

	# Single Matrix Operations
	# # # #	# # # #	# # # #	# 

	def eval_variables(self, matrix_num,new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		# Get all symbols in the matrix and sort them in alphabetical order
		symbols_in_matrix = sorted(list(matrix.free_symbols), key=lambda s: str(s))
		if not symbols_in_matrix:
			print("No variables to evaluate in this matrix.")
			input("Enter to continue")
			return

		substitutions = {}
		i = 0

		invalid = False
		while i < len(symbols_in_matrix):

			clear_screen()
			if invalid:
				print("Invalid input. Please enter a number or a fraction.")
				invalid = False

			print(f"\nMatrix {matrix_title}:\n")
			pprint(matrix)  # pretty print the matrix

			print(f"Enter a value: 'c' to keep constant,'-' to go back, 'x' to quit")
			symbol = symbols_in_matrix[i]
			while True:
				user_input = input(f"  >   {pretty(symbol)}: ")
				if user_input.lower() == 'c':
					i += 1
					break
				elif user_input.lower() == '-' and i != 0:
					i -= 1
					break
				elif user_input.lower() == 'x':
					return False
				else:
					try:
						# Convert input to number or fraction
						value = sympify(user_input)
						substitutions[symbol] = value
						i += 1
						break
					except (SympifyError):
						invalid = True

		# Create new matrix with evaluated variables
		if substitutions != {}:
			new_matrix = matrix.subs(substitutions)
			if new:
				# Append new matrix to matrices
				self.matrices.append([get_name(f"{matrix_title}_evaluated"), new_matrix])
			else:
				self.matrices[matrix_num][1] = new_matrix
			return True
		else:
			print("All variables where constant")
			input("Enter to continue")
			return False

	def get_scale_factor(self,matrix_num):
		matrix_title, matrix = self.matrices[matrix_num]
		invalid = False
		while True:

			clear_screen()
			if invalid:
				print("Invalid input. Please enter a number or a fraction.")
				invalid = False

			print(f"\nMatrix {matrix_title}:\n")
			pprint(matrix)  # pretty print the matrix

			user_input = input("Enter the scaling factor (e.g., 3 or 1/2, 'x' to go back): ")
			if user_input.lower() == 'x':
				return None
			try:
				# Convert input to number or fraction
				scale_factor = sympify(user_input)
				return scale_factor
			except (SympifyError):
				invalid = True


	def get_row_number(self, message, matrix_num,col=False):
		matrix_title, matrix = self.matrices[matrix_num]
		invalid = ""
		while True:

			clear_screen()
			if invalid:
				print(invalid)
				invalid = ""

			print(f"\nMatrix {matrix_title}:\n")
			pprint(matrix)  # pretty print the matrix

			n = matrix.rows if not col else matrix.cols
			row_num = input(f"{message} (1-{n}) or 'x' to go back: ")
			if row_num.lower() == 'x':
				return None
			try:
				row_num = int(row_num) - 1
				
				if row_num in range(n):
					return row_num
				else:
					invalid = "Invalid row number."
			except ValueError:
				invalid = "Invalid input. Please enter an integer."

	
	def scale_matrix(self,matrix_num,new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		scaling_factor = self.get_scale_factor(matrix_num)
		if scaling_factor is not None:
			if new:
				new_matrix = self.matrices[matrix_num][1] * scaling_factor
				self.matrices.append([get_name(f"{matrix_title}_*_{scaling_factor}"), new_matrix])
			else:
				self.matrices[matrix_num][1] *= scaling_factor
			return True


	def scale_row(self,matrix_num,new=False):
		matrix_title ,matrix = self.matrices[matrix_num]
		invalid = False
		while True:
			clear_screen()

			print(f"\nMatrix {matrix_title}:\n")
			pprint(matrix)  # pretty print the matrix
			
			# Get row number
			row_num = self.get_row_number("Enter the row number to scale", matrix_num)
			if row_num is None:
				return False

			# Get scaling factor
			scaling_factor = self.get_scale_factor(matrix_num)
			if scaling_factor is None:
				return False

			# Scale row elements
			new_matrix = Matrix(matrix.tolist())
			new_matrix[row_num,:] = new_matrix[row_num,:] * scaling_factor
			new_matrix[row_num,:].simplify
			if new:
				self.matrices.append([get_name(f"{matrix_title}_row{row_num + 1}_*_{scaling_factor}"), new_matrix])
			else:
				self.matrices[matrix_num][1] = new_matrix

		
			# Return from the method
			return True
		

	def rearrange(self, matrix_num,new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		# get row1
		clear_screen()
		print(f"\nMatrix {matrix_title}:\n")
		pprint(matrix)  # pretty print the matrix

		row1 = self.get_row_number("Enter the first row number", matrix_num)
		if row1 is None:
			return False
		# get row2
		clear_screen()
		print(f"\nMatrix {matrix_title}:\n")
		pprint(matrix)  # pretty print the matrix

		row2 = self.get_row_number("Enter the second row number", matrix_num)
		if row2 is None:
			return False
		new_matrix = Matrix(matrix.tolist())
		new_matrix.row_swap(row1, row2)
		if new:
			self.matrices.append([get_name(f"{matrix_title}_row{row1 + 1}_row{row2 + 1}_swap"), new_matrix])
		else:
			self.matrices[matrix_num][1] = new_matrix		
		return True


	def scale_and_combine(self, matrix_num,new=False):
		# Get matrix
		matrix_title, matrix = self.matrices[matrix_num]

		# Get row to scale
		clear_screen()
		print(f"\nMatrix {matrix_title}:\n")
		pprint(matrix)  # pretty print the matrix
		row_to_scale = self.get_row_number("Enter the row to scale", matrix_num)
		if row_to_scale is None:
			return False

		# Get row to add to
		clear_screen()
		print(f"\nMatrix {matrix_title}:\n")
		pprint(matrix)  # pretty print the matrix
		row_to_add = self.get_row_number("Enter the row to add to", matrix_num)
		if row_to_add is None:
			return False

		# Get scaling factor
		scaling_factor = self.get_scale_factor(matrix_num)

		# Scale and combine
		new_matrix = Matrix(matrix.tolist())
		scaled_row = new_matrix.row(row_to_scale) * scaling_factor
		combined_row = new_matrix.row(row_to_add) + scaled_row
		combined_row.simplify()

		# Update matrix with new row
		new_matrix[row_to_add,:] = combined_row

		if new:
			self.matrices.append([get_name(f"{matrix_title}_scale_and_combine"), new_matrix])
		else:
			self.matrices[matrix_num][1] = new_matrix	
		return True


	def echelon_form(self,matrix_num,new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		if new:
			new_matrix = matrix.echelon_form()
			self.matrices.append([get_name(f"{matrix_title}_echelon_form"), new_matrix])
		else:
			self.matrices[matrix_num][1] = matrix.echelon_form()  # Use SymPy's echelon_form method
		return True


	def rref(self,matrix_num,new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		if new:
			new_matrix = matrix.rref(pivots=False)
			self.matrices.append([get_name(f"{matrix_title}_rref"), new_matrix])
		else:
			self.matrices[matrix_num][1] = matrix.rref(pivots=False)  # Use SymPy's rref method
		return True

	def invert_matrix(self, matrix_num,new=False):
		matrix_title, matrix = self.matrices[matrix_num]

		# Check if the matrix is square
		if matrix.shape[0] != matrix.shape[1]:
			print("Error: Only square matrices can be inverted.")
			input("Enter to continue")
			return False

		# Check if the matrix is invertible
		if matrix.det() == 0:
			print("Error: Matrix is singular (not invertible).")
			input("Enter to continue")
			return False

		try:
			# Invert the matrix
			inverted_matrix = matrix.inv()
			# Append the inverted matrix to the matrices list
			if new:
				self.matrices.append([get_name(f"{matrix_title}_inverted"), inverted_matrix])
			else:
				self.matrices[matrix_num][1] = inverted_matrix
			return True
		except ValueError as e:
			print("Error: ", e)
			input("Enter to continue")
			return False

	def adjugate_matrix(self, matrix_num, new=False):
		matrix_title, matrix = self.matrices[matrix_num]

		# Check if the matrix is square
		if matrix.shape[0] != matrix.shape[1]:
			print("Error: Only square matrices can be adjugated.")
			input("Enter to continue")
			return False

		try:
			# Compute the adjugate of the matrix
			adjugated_matrix = matrix.adjugate()
			# Append the adjugated matrix to the matrices list
			if new:
				self.matrices.append([get_name(f"{matrix_title}_adjugated"), adjugated_matrix])
			else:
				self.matrices[matrix_num][1] = adjugated_matrix
			return True

		except ValueError as e:
			print("Error: ", e)
			input("Enter to continue")
			return False


	def transpose_matrix(self,matrix_num,new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		if new:
			self.matrices.append([get_name(f"{matrix_title}_transpose"), matrix.transpose()])
		else:
			self.matrices[matrix_num][1] = matrix.transpose()
		return True

	def raise_matrix_to_power(self,matrix_num,new=False):
		matrix_title, matrix = self.matrices[matrix_num]

		# Check if the matrix is square
		if matrix.shape[0] != matrix.shape[1]:
			print("Error: Only square matrices can be raised to a power.")
			input("Enter to continue")
			return

		invalid = ""
		while True:

			clear_screen()
			if invalid:
				print(invalid)
				invalid = ""

			print(f"\nMatrix {matrix_title}:\n")
			pprint(matrix)  # pretty print the matrix

			power = input("Enter the power to raise the matrix to ('x' to go back): ")
			if power.lower() == 'x':
				return
			else:
				try:
					power = int(power)
					if power >= 0:
						break
					else:
						invalid = "Error: Power must be a non-negative integer."
				except ValueError:
					invalid = "Error: Invalid input. Please enter a non-negative integer."

		# Raise the matrix to the power
		if new:
			self.matrices.append([get_name(f"{matrix_title}^{power}"), matrix**power])
		else:
			self.matrices[matrix_num][1] = matrix**power
		return True

	def transform_vector(self, matrix_num, new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		num_cols = matrix.shape[1]

		# Create 'x' vector
		x_vector = []
		i = 0

		while i < num_cols:

			clear_screen()

			print(f"\nMatrix {matrix_title}:\n")
			pprint(matrix)  # pretty print the matrix

			print("Description: Use your matrix 'A' to transform a vector 'x' (right-multiply 'Ax')")
			print("Enter values for the 'x' vector, '-' to go back, 'x' to quit:")

			value, status = self.input_value(i, row=False, n=0)
			if status == 'back':
				i -= 1
				x_vector = x_vector[:-1]  # remove the last value added
			elif status == 'quit':
				return False

			elif status == 'continue':
				x_vector.append(value)
				i += 1

		# Convert 'x' vector to a column Matrix
		x_vector = Matrix(x_vector)

		while True:

			# Left multiply x by matrix
			try:
				result_matrix = matrix * x_vector

				# Print the result
				print(f"\nResult of multiplying vector 'x' by matrix '{matrix_title}':\n")
				print(pretty(result_matrix))

				# Save the result
				user_input = input(f"\nMultiply {matrix_title} by {result_matrix.tolist()} again? ('y' for yes, else quit)\n > > > ")
				if user_input.lower() == 'y':
					x_vector = result_matrix
					continue
				elif user_input.lower() == 'n':
					new=True
				if new:
					self.matrices.append([get_name(f"{matrix_title}_x_{x_vector.tolist()}_transform_vector"), result_matrix])	
				else:
					user_input = input(f"Overwrite {matrix_title} with {result_matrix.tolist()}? ('y' to confirm)')\n > > > ")	
					if user_input.lower() == 'y':
						self.matrices[matrix_num][1] = result_matrix
						self.matrices[matrix_num][0] = get_name(f"{matrix_title}_x_{x_vector.tolist()}_transform_vector")
				return True
			except Exception as e:
				print("Error: ", e)
				input("Enter to continue")
				return False


	def projection_onto_subspace(self, matrix_num, new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		num_rows,num_cols = matrix.shape
		
		# Check if all the columns of 'A' are orthogonal to each other
		is_orthogonal = all(Matrix.dot(matrix[:, i], matrix[:, j]) == 0 for i in range(num_cols) for j in range(i + 1, num_cols))
		

		# Ask the user to input a point in R^n
		y_vector = []
		
		i = 0
		while i < num_rows:

			clear_screen()

			print(f"\nMatrix {matrix_title}:\n")
			pprint(matrix)  # pretty print the matrix
			print(f"All columns of '{matrix_title}' are orthogonal to each other: {is_orthogonal}")

			# print("Could use a description of what this function is doing :)")
			print("Enter values for the 'y' vector, '-' to go back, 'x' to quit:")


			value, status = self.input_value(i, row=False, n=0)
			if status == 'back':
				i -= 1
				y_vector = y_vector[:-1]  # remove the last value added
			elif status == 'quit':
				return False
			elif status == 'continue':
				y_vector.append(value)
				i += 1

		clear_screen()
		# Convert 'y' vector to a column Matrix
		y_vector = Matrix(y_vector)

		# Calculate the projection and orthogonal vectors
		projection = Matrix.zeros(y_vector.shape[0], 1)
		print("\nCalculated projection weights for each vector:")
		for i in range(num_cols):
			w = matrix[:, i]
			debug_print(f"Matrix.dot(y_vector, w) {Matrix.dot(y_vector, w)}")
			debug_print(f"Matrix.dot(w, w) {Matrix.dot(w, w)}")

			weight = (Matrix.dot(y_vector, w) / Matrix.dot(w, w))
			projection += weight * w
			print(f"  Weight for vector {i+1}: {weight}")

		y_hat = projection
		orthogonal_vector = y_vector - y_hat

		print("\nCalculated vector y_hat:")
		print(pretty(y_hat))

		print("\nOrthogonal vector z = y - y_hat:")
		print(pretty(orthogonal_vector))

		input("Enter to continue")

		# Save y_hat as a column vector if 'new' is True
		if new:
			self.matrices.append([get_name(f"proj_{matrix_title}"), y_hat])

		return True

	def gram_schmidt(self, matrix_num, new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		num_cols = matrix.shape[1]
		
		# Check if the columns of the matrix are linearly independent
		is_linearly_independent = matrix.rank() == num_cols
		if not is_linearly_independent:
			print(f"The columns of the matrix '{matrix_title}' are not linearly independent.")
			input("Enter to continue")
			return False

		print(f"All columns of the matrix '{matrix_title}' are linearly independent.")

		# Initialize lists to hold the orthogonal and orthonormal bases
		orthogonal_basis = []
		orthonormal_basis = []

		# Apply the Gram-Schmidt process
		for i in range(num_cols):
			# Compute orthogonal basis
			orthogonal_vector = matrix[:, i]
			w = matrix[:, i]
			for j in range(i):
				orthogonal_vector -= orthogonal_basis[j] * (Matrix.dot(orthogonal_basis[j], w) / Matrix.dot(orthogonal_basis[j], orthogonal_basis[j]))
			orthogonal_basis.append(orthogonal_vector)

			# Compute orthonormal basis
			orthonormal_vector = orthogonal_vector.normalized()
			orthonormal_basis.append(orthonormal_vector)

		# Print the orthogonal and orthonormal bases
		print("\nOrthogonal basis:")
		orthogonal_matrix = Matrix([vector.T for vector in orthogonal_basis]).T
		pprint(orthogonal_matrix)
		#for vector in orthogonal_basis:
		#	print(pretty(vector))
		orthonormal_matrix = Matrix([vector.T for vector in orthonormal_basis]).T
		print("\nOrthonormal basis:")
		pprint(orthonormal_matrix)
		#for vector in orthonormal_basis:
		#	print(pretty(vector))
		input("Enter to continue")

		# Save the orthogonal basis as a new matrix if desired
		if new:
			user_input = input(f"Orthogonal ('g'), Orthonormal ('n'), or both ('b')? (: ")
			if user_input.lower() == 'g' or user_input.lower() == 'b':
				self.matrices.append([get_name(f"{matrix_title}_orthogonal_basis"), orthogonal_matrix])
			if user_input.lower() == 'n' or user_input.lower() == 'b':
				self.matrices.append([get_name(f"{matrix_title}_orthonormal_matrix"), orthonormal_matrix])

			return True
		else:
			self.matrices[matrix_num][1] = orthogonal_matrix
			return True



	def determinant(self, matrix_num,new=False):
		matrix_title, matrix = self.matrices[matrix_num]

		# Check if the matrix is square
		if matrix.shape[0] != matrix.shape[1]:
			print("Error: Only square matrices can have a determinant.")
			input("Enter to continue")
			return False

		try:
			# Calculate the determinant
			determinant = matrix.det()

			# Print the determinant
			if '\n' in pretty(determinant):
				print(f"\nDeterminant of matrix '{matrix_title}': {determinant}\n")
			else:
				print(f"\nDeterminant of matrix '{matrix_title}': {pretty(determinant)}\n")
			input("Enter to continue")
			return True
		except ValueError as e:
			print("Error: ", e)
			input("Enter to continue")
			return False



	def input_value(self, i, row=True, n=None):

		while True:
			if row:
				user_input = input(f"  >   row {n+1}, column {i+1}: ")
			else:
				user_input = input(f"  >   row {i+1}, column {n+1}: ")
			
			if user_input.lower() == '-' and i != 0:
				return None, 'back'
			elif user_input.lower() == 'x':
				return None, 'quit'
			else:
				try:
					# Convert input to number or fraction
					value = sympify(user_input)
					return value, 'continue'
				except (SympifyError):
					print("Invalid input. Please enter a number or a fraction.")


	def modify_row(self, matrix_num,new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		num_cols = matrix.shape[1]
		
		row_num = self.get_row_number("Enter the row number to modify", matrix_num)
		if row_num is None:
			return False

		new_matrix = Matrix(matrix.tolist())

		i = 0
		while i < num_cols:
			value, status = self.input_value(i, row=True, n=row_num)
			if status == 'back':
				i -= 1
			elif status == 'quit':
				return False
			elif status == 'continue':
				new_matrix[row_num, i] = value
				i += 1

		# Update the matrix
		if new:
			self.matrices.append([get_anem(f"{matrix_title}_∆row{row_num + 1}"), new_matrix])
		else:
			self.matrices[matrix_num][1] = new_matrix
		return True


	def modify_column(self, matrix_num,new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		num_rows = matrix.shape[0]
		
		col_num = self.get_row_number("Enter the column number to modify", matrix_num,col=True)
		if col_num is None:
			return False

		new_matrix = Matrix(matrix.tolist())

		i = 0
		while i < num_rows:
			value, status = self.input_value(i, row=False, n=col_num)
			if status == 'back':
				i -= 1
			elif status == 'quit':
				return False
			elif status == 'continue':
				new_matrix[i, col_num] = value
				i += 1

		# Update the matrix
		if new:
			self.matrices.append([get_name(f"{matrix_title}_∆col{col_num + 1}"), new_matrix])
		else:
			self.matrices[matrix_num][1] = new_matrix
		return True


	def add_row(self, matrix_num,new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		new_row = []
		num_cols = matrix.shape[1]

		i = 0
		while i < num_cols:
			value, status = self.input_value(i, row=True, n=matrix.rows)
			if status == 'back':
				i -= 1
				new_row = new_row[:-1]  # remove the last value added
			elif status == 'quit':
				return False
			elif status == 'continue':
				new_row.append(value)
				i += 1

		# Add new row to the matrix
		new_matrix = matrix.row_insert(matrix.rows, Matrix([new_row]))
		if new:
			self.matrices.append([get_name(f"{matrix_title}_add_row{new_matrix.rows}"), new_matrix])
		else:
			self.matrices[matrix_num][1] = new_matrix
		return True


	def add_column(self, matrix_num,new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		new_column = []
		num_rows = matrix.shape[0]

		i = 0
		while i < num_rows:
			value, status = self.input_value(i, row=False, n=matrix.cols)
			if status == 'back':
				i -= 1
				new_column = new_column[:-1]  # remove the last value added
			elif status == 'quit':
				return False
			elif status == 'continue':
				new_column.append(value)
				i += 1

		# Add new column to the matrix
		new_matrix = matrix.col_insert(matrix.cols, Matrix(new_column))
		if new:
			self.matrices.append([get_name(f"{matrix_title}_add_col{new_matrix.cols}"), new_matrix])
		else:
			self.matrices[matrix_num][1] = new_matrix
		return True

	def delete_row(self, matrix_num, new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		
		# Ensure matrix has more than one row
		if matrix.rows <= 1:
			print("Error: The matrix should have more than one row to delete a row.")
			return False
		
		# Ask the user for the row number to delete
		while True:
			row_num = input(f"Enter the row number (1-{matrix.rows}) to delete or 'x' to cancel: ")
			if row_num.lower() == 'x':
				return False
			try:
				row_num = int(row_num) - 1
				if 0 <= row_num < matrix.rows:
					break
				else:
					print(f"Invalid input. Please enter a number between 1 and {matrix.rows}.")
			except ValueError:
				print("Invalid input. Please enter an integer.")

		# Delete the specified row
		new_matrix = matrix.copy()
		new_matrix.row_del(row_num)
		
		# Update or append the new matrix
		if new:
			self.matrices.append([get_name(f"{matrix_title}_del_row{new_matrix.rows}"), new_matrix])
		else:
			self.matrices[matrix_num][1] = new_matrix

		return True

	def delete_column(self, matrix_num, new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		
		# Ensure matrix has more than one column
		if matrix.cols <= 1:
			print("Error: The matrix should have more than one column to delete a column.")
			return False
		
		# Ask the user for the column number to delete
		while True:
			col_num = input(f"Enter the column number (1-{matrix.cols}) to delete or 'x' to cancel: ")
			if col_num.lower() == 'x':
				return False
			try:
				col_num = int(col_num) - 1
				if 0 <= col_num < matrix.cols:
					break
				else:
					print(f"Invalid input. Please enter a number between 1 and {matrix.cols}.")
			except ValueError:
				print("Invalid input. Please enter an integer.")

		# Delete the specified column
		new_matrix = matrix.copy()
		new_matrix.col_del(col_num)
		
		# Update or append the new matrix
		if new:
			self.matrices.append([get_name(f"{matrix_title}_del_col{new_matrix.cols}"), new_matrix])
		else:
			self.matrices[matrix_num][1] = new_matrix

		return True



	# Eigenvalue
	# # # # # # # # # # # # #


	def eigenvects(self, matrix_num, new=False):
		matrix_title, matrix = self.matrices[matrix_num]

		# Check if the matrix is square
		if matrix.shape[0] != matrix.shape[1]:
			print("Error: Only square matrices can have eigenvectors.")
			input("Enter to continue")
			return False

		# TO DO
		# The eigenvects method hangs up if the matrix contains variables instead of constant scalars
		# Add data validation to ensure this won't happen

		display_option = input("Do you want to display the results as decimal or fraction? Enter 'd' for decimal, 'f' for fraction: ").strip().lower()
		while display_option not in ['d', 'f']:
			clear_screen()
			display_option = input("Invalid input. Enter 'd' for decimal, 'f' for fraction: ").strip().lower()
		as_decimal = (display_option == 'd')

		try:
			# Calculate the eigenvectors and eigenvalues
			eigenvectors = matrix.eigenvects()

			# Print the eigenvectors and eigenvalues
			clear_screen()
			print(f"\nMatrix {matrix_title}:\n")
			pprint(matrix)  # pretty print the matrix

			print(f"\nEigenvectors and Eigenvalues of matrix '{matrix_title}':")

			for eigenvalue, multiplicity, vectors in eigenvectors:

				# Not sure why this is here: 
				# some eigenvalues have bad formatting, this was probably an attempt to fix the formatting
				if '\n' in pretty(eigenvalue):
					c = '\n'
				else:
					c = ''
				# 'c' is never used :(

				print(f"Eigenvalue: {format_expression(eigenvalue, as_decimal)}, Multiplicity: {multiplicity}")
				print("Eigenvectors:")
				for vector in vectors:
					print(pretty([format_expression(item, as_decimal) for item in vector]))
					print("\n")  # Add an extra newline for formatting

			input("Enter to continue")
			return True
		except ValueError as e:
			print("Error: ", e)
			input("Enter to continue")
			return False



	def char_poly(self, matrix_num, new=False):
		matrix_title, matrix = self.matrices[matrix_num]

		# Verify the matrix is square
		if matrix.shape[0] != matrix.shape[1]:
			print("Error: The characteristic polynomial can only be calculated for square matrices.")
			input("Enter to continue")
			return False

		clear_screen()
		print(f"\nMatrix {matrix_title}:\n")
		pprint(matrix)  # pretty print the matrix

		# Compute the characteristic polynomial
		char_polynomial = matrix.charpoly()

		# Extract the symbol used in the characteristic polynomial (usually 'lambda')
		lamda = char_polynomial.gens[0]

		# Pretty print the characteristic polynomial
		print(f"The characteristic polynomial for matrix '{matrix_title}' is:")
		print(pretty(char_polynomial.as_expr()))
		input("Enter to continue")

		# Attempt to factor the characteristic polynomial
		factored_polynomial = factor(char_polynomial.as_expr())
		if factored_polynomial != char_polynomial.as_expr():
			print(f"The factored form of the characteristic polynomial is:")
			print(pretty(factored_polynomial))
		else:
			print("The characteristic polynomial could not be factored further.")
		input("Enter to continue")

		# Find all real and complex roots of the polynomial
		roots = solveset(char_polynomial.as_expr(), lamda, domain=S.Complexes)
		if roots:
			print("The roots of the characteristic polynomial are:")
			for root in roots:
				print(pretty(root))
		else:
			print("No roots found for the characteristic polynomial.")

		input("Enter to continue")
		return True




	def diagonalization(self, matrix_num, new=False):
		matrix_title, matrix = self.matrices[matrix_num]

		# Check if the matrix is square, as only square matrices can be diagonalized
		if matrix.shape[0] != matrix.shape[1]:
			print("Error: Only square matrices can be diagonalized.")
			input("Enter to continue")
			return False

		clear_screen()
		print(f"\nMatrix {matrix_title}:\n")
		pprint(matrix)  # pretty print the matrix

		try:
			# Attempt to diagonalize the matrix
			P, D = matrix.diagonalize()
		except ValueError as e:
			print(f"Error: Matrix '{matrix_title}' cannot be diagonalized. {e}")
			input("Enter to continue")
			return False

		# Print the diagonalized matrix and the transformation matrix
		print(f"Matrix '{matrix_title}' has been diagonalized:")
		print(f"Transformation matrix P:\n{pretty(P)}")
		print(f"Diagonalized matrix D:\n{pretty(D)}")

		# Overwrite the existing matrix or append a new one based on the 'new' argument
		if new:
			self.matrices.append([get_name(f"{matrix_title}_D_diagonalized"), D])
			self.matrices.append([get_name(f"{matrix_title}_P_transformation"), P])
		else:
			self.matrices[matrix_num][1] = D
			input("Enter to continue")
			
		return True

	# Multi Matrix Operations
	# # # # # # # # # # # # #

	def choose_matrix(self, message, choice1=None):

		invalid = False
		while True:

			if invalid:
				print(f"Invalid matrix choice. Please choose a letter.")
				invalid = False
			clear_screen()

			print()
			self.print_heading(mode='choice')
			print()
			self.print()
			print()
			matrix_choice = input(message)
			if matrix_choice.lower() == 'x':
				return None
			matrix_choice = ord(matrix_choice.upper()) - 65
			if matrix_choice in range(len(self.matrices)):
				return matrix_choice
			else:
				invalid = True


	def matrix_mult(self):
		# Get first matrix
		matrix_1_choice = self.choose_matrix("Choose the left matrix for multiplication ('x' to go back): ")
		if matrix_1_choice is None:
			return False

		title_A, matrix_A = self.matrices[matrix_1_choice]
		temp = self.matrices.pop(matrix_1_choice)

		# Get second matrix
		matrix_2_choice = self.choose_matrix("Choose the right matrix for multiplication ('x' to go back): ")
		if matrix_2_choice is None:
			# If operation is canceled, insert the first matrix back
			self.matrices.insert(matrix_1_choice, temp)
			return False

		title_B, matrix_B = self.matrices[matrix_2_choice]

		# Insert the first matrix back
		self.matrices.insert(matrix_1_choice, temp)

		# Multiply matrices
		try:
			result = matrix_A * matrix_B  # Use SymPy's built-in multiplication
		except ValueError:
			print("Matrix multiplication failed. Please make sure the matrices are compatible for multiplication.")
			input("Enter to continue")
			return False

		self.matrices.append([get_name(f"{title_A}_x_{title_B}"),result])
		return True



	# Printing Methods
	# # # # # # # # # 

	def print_heading(self, mode=None):
		for i in range(len(self.matrices)):

			file_name, matrix_data = self.matrices[i]

			# Print matrix number and file name
			if mode != 'choice':
				first_row = f"{i + 1}: '{file_name}' "
			else:
				first_row = f"{chr(i+65)}): '{file_name}' "

			# Calculate and print the dimensions
			num_rows, num_cols = matrix_data.shape
			first_row += f" [{num_rows} X {num_cols}] "

			# Print the first row of the matrix
			#first_row = matrix_data[0]

			# Limit the number of items printed to 6
			first_row += " ["
			for i in range(min(matrix_data.shape[1],6)):
				if "\n" in pretty(matrix_data[0,i]):
					first_row += sstr(matrix_data[0,i])
				else:
					first_row += pretty(matrix_data[0,i])
				first_row += ", "
			if matrix_data.shape[1] > 6:
				first_row += "..."	
			first_row = first_row.strip(", ") + "]"
			print(first_row)


	def print(self,debug_print=False):

		if self.matrices == []:
			print("No Matrices Loaded")
			return

		# Initialize matrices
		if debug_print:
			c = '@'
		else:
			c = ' '

		max_len = self.get_length()

		max_row = max([m[1].rows for m in self.matrices])
		matrices = [m[1] for m in self.matrices]

		if debug_print:
			print(f"max_row = {max_row}")

		self.double_headrow(max_len)
		
		for i in range(max_row):
			for n, matrix in enumerate(matrices):
				if debug_print:
					print(f"\n{matrix}")
				if i < matrix.rows:
					if debug_print:
						print(f"i = {i}, matrix.rows = {matrix.rows}, i < matrix.rows = {i < matrix.rows}")
						print(f"matrix = {matrix}")
						print(f"matrix.row(1) = {matrix.row(1)}")
						print(f"matrix.row(1).tolist() = {matrix.row(1).tolist()}")
						print(f"matrix.row(1).tolist()[0] = {matrix.row(1).tolist()[0]}")
					row = matrix.row(i).tolist()[0]
					m = max_len[n]
					if n == 0:
						print(f"{i:>{m[0]}}", end="")
						print("—" * m[0], end="")
					else:
						print(f"{i:>{m[0]*2}}", end="")
						print("—" * m[0], end="")
					self.print_row(i, row, m)

				if i >= matrix.rows:
					if debug_print:
						print(f"i = {i}, len(matrix) = {len(matrix)}, i >= len(matrix) = {i >= len(matrix)}")
					m = max_len[n]
					if n == 0:
						print(c*m[0], end="")
					else:
						print(c*m[0]*2, end="")
					print(c*m[0], end="")
					for j in range(matrix.cols):
						m = max_len[n][j]
						print(c*m, end="")
			print()


	def print_row(self, index, row, max_len, debug_print=False):
		for i, item in enumerate(row):
			m = max_len[i]
			item_evaluated = N(item)
			if debug_print:
				print(f"item = {item}, item_evaluated = {item_evaluated}")
			if item_evaluated.is_number:

				# this crashes if item_evaluated is complex
				item_float = float(item_evaluated)

				closest_fraction = Fraction.from_float(item_float).limit_denominator()
				if abs(closest_fraction - item_float) < 0.01:
					print(f"{str(closest_fraction).rjust(m)}", end="")
				elif item.evalf() % 1 < 1E-10 or item.evalf() % 1 > 0.9999999999:
					print(f"{round(item):>{m}}", end="")
				elif item.evalf() % 10 < .0000000001 or item.evalf() % 10 > 9.9999999999:
					print(f"{round(item, -1):>{m}}", end="")
				else:
					print(f"{round(item, precision):>{m}}", end="")
			else:
				if "\n" in pretty(item):
					print(f"{sstr(item).strip():>{m}}", end="")
				else:
					print(f"{pretty(item).strip():>{m}}", end="")


	def get_col_length(self,col):
		max_len = 3
		for item in col:
			item = item[0]
			item_evaluated = N(item)
			if item_evaluated.is_number:

				# This crashed is item_evaluated is complex
				item_float = float(item_evaluated)

				closest_fraction = Fraction.from_float(item_float).limit_denominator()
				if abs(closest_fraction - item_float) < 0.01:
					max_len = max(max_len,len(f"{str(closest_fraction).rjust(max_len)}"))
				elif item.evalf() % 1 < 1E-10 or item.evalf() % 1 > 0.9999999999:
					max_len = max(max_len,len(f"{round(item):>{max_len}}"))
				elif item.evalf() % 10 < .0000000001 or item.evalf() % 10 > 9.9999999999:
					max_len = max(max_len,len(f"{round(item, -1):>{max_len}}"))
				else:
					max_len = max(max_len,len(f"{round(item, precision):>{max_len}}"))
			else:
				if "\n" in pretty(item):
					max_len = max(max_len,len(f"{sstr(item).strip():>{max_len}}"))
				else:
					max_len = max(max_len,len(f"{pretty(item).strip():>{max_len}}"))
		return max_len + 1


	def get_length(self,debug_print=False):
		max_len = [[] for _ in range(len(self.matrices))]
		
		for i, matrix in enumerate(self.matrices):
			m = 0
			for j in range(matrix[1].cols):
				if debug_print:
					print(f"i = {i}, j = {j}, matrix = {matrix[0]}")
					print(f"matrix[1].col(j).tolist()[0] = {matrix[1].col(j).tolist()}")
				m = max(m,self.get_col_length(matrix[1].col(j).tolist()))
				if debug_print:
					print(f"m = {m}")
				max_len[i].append(m)
			if debug_print:
				print(f"max_len = {max_len}")

		if debug_print:
			print(f"max_len = {max_len}")

		return max_len


	def double_headrow(self,max_len,debug_print=False):
		if debug_print:
			c = '@'
		else:
			c = ' '

		for i, matrix in enumerate(self.matrices):
			m = max_len[i][0]
			if i == 0:
				print(c*(m-1), end="")
			else:
				print(c*(m*2-1), end="")
			for j in range(matrix[1].cols):
				m += max_len[i][j]
			print(f"{matrix[0]:<{m}}", end="")
			print(c,end="")
			
		print()  # Go to the next line			

		for i, matrix in enumerate(self.matrices):
			m = max_len[i][0]
			if i == 0:
				print(c*(m)*2, end="")
			else:
				print(c*(m)*3, end="")
			for j in range(matrix[1].cols):
				m = max_len[i][j]
				print(f"{j:>{m}}", end="")
			
		print()  # Go to the next line

		for i, matrix in enumerate(self.matrices):
			m = max_len[i][0]
			if i == 0:
				print(c*(m)*2, end="")
			else:
				print(c*(m)*3, end="")
			for j in range(matrix[1].cols):
				m = max_len[i][j]
				print(f"{'|':>{m}}", end="")
		print()