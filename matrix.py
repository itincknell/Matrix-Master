
import csv, os, glob
import copy
from fractions import Fraction
import sympy
import numpy as np

DEBUG = False
import inspect

def debug_print(message):
	line_number = inspect.currentframe().f_back.f_lineno
	if DEBUG:
		print(f"[Line {line_number}] - {message}")

def get_name(default):
	name = input(f"Use: {default} ('1' to except)?\nor enter new name: ")
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

def expression_output_options():
	print("Select Output Format:")
	print("1) for LaTex")
	print("2) for pretty SymPy")
	print("3) for plain SymPy")
	user_input = input(": ")

	if user_input == '1':
		return sympy.latex
	if user_input == '2':
		return sympy.pretty
	else:
		return lambda x: x

def format_sci_notation(expr):
	try:
		# Check if it's a SymPy expression
		if isinstance(expr, sympy.Expr):
			# Simplify and evaluate the expression
			return '{:.3e}'.format(float(expr))  
		else:
			return expr
	except (ValueError, TypeError):
		return expr		

class Matrix_Calc:
	def __init__(self):
		self.matrices = [] 
		self.precision = 2
		self.mode = 'pretty'
		self.print_expr_function = sympy.pretty
		self.print_matrix_function = sympy.pretty


	# allow user to flip between display modes inside most methods
	def toggle_print_mode(self):
		if self.mode == 'pretty':
			self.mode = 'latex'
			self.print_expr_function = sympy.latex
			self.print_matrix_function = sympy.latex
		elif self.mode == 'latex':
			self.mode = 'pretty'
			self.print_expr_function = sympy.pretty
			self.print_matrix_function = sympy.pretty
		'''
			self.mode = 'sci-pretty'
			self.print_expr_function = lambda expr : sympy.pretty(format_sci_notation(expr))
			self.print_matrix_function = lambda matr : sympy.pretty(matr.applyfunc(format_sci_notation))	
		elif self.mode == 'sci-pretty':
			self.mode = 'sci-latex'
			self.print_expr_function = lambda expr : sympy.latex('{:.1e}'.format(expr.evalf()))
			self.print_matrix_function = lambda matr : sympy.latex(matr.applyfunc(format_sci_notation))
		elif self.mode == 'sci-latex':
			self.mode = 'pretty'
			self.print_expr_function = sympy.pretty
			self.print_matrix_function = sympy.pretty
		'''

	def get_matrix_choices(self, mode=None):
		remaining = [i for i in range(len(self.matrices))]
		mask = []
		choices = []
		invalid = ""

		while True:

			if remaining == []:
				return choices

			clear_screen()

			if invalid:
				print(invalid)
				invalid = ""

			print("Matrix choices")
			self.print_matrix_indices(remaining)

			if mode == 'Matrix Algebra' and len(choices) == 0:
				print("Choose LHS Matrix:")
			elif mode == 'Matrix Algebra' and len(choices) == 1:
				print("Choose RHS Matrix:")
			elif mode == 'Matrix Algebra' and len(choices) == 2:
				return choices

			
			self.print_heading(mask=mask)

			user_input = input("Enter your choice. 'x' to finish\n")

			if user_input.lower() == 'x':
				return choices
			else:
				try:
					user_input = int(user_input) - 1
					if user_input not in remaining:
						invalid = "Invalid selection, choose from the available options"
						continue
					else:
						mask.append(user_input)
						choices.append(user_input)
						remaining.remove(user_input)

				except ValueError:
					invalid = "Invalid selection, enter an integer"


	# File handling methods
	# # # #	# # # #	# # # #	

	def load_from_file(self, overwrite=True):
		folder_path = './matrix_files/'
		if not os.path.exists(folder_path):
			os.mkdir(folder_path)

		files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.txt')]

		if files == []:
			return "No saved files found"
		files = sorted(files)

		while True:

			clear_screen()

			print("\nAvailable files:")
			for i, file in enumerate(files):
				print(f"{i + 1}: {file}")

		
			file_choice = input("Enter the number of the file you want to load or 'x' to cancel: ")
			if file_choice == 'x':
				return
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
					return
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
			data = [[sympy.sympify(i) for i in row] for row in reader]
			data = sympy.Matrix(data)

		if overwrite:
			self.matrices[int(matrix_choice) - 1] = [file_name.replace(folder_path,''), data]
		else:
			self.matrices.append([file_name.replace(folder_path,''), data])
		return

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
			matrix_type = input("M) Manual entry\nU) unit matrix\nV) variable matrix\nR) random matrix\n Choose options ('x' to go back)? ")
			if matrix_type.lower() == 'x':
				return False
			elif matrix_type.lower() == 'm':
				matrix = self.manual_entry(width, height)
			elif matrix_type.lower() == 'i' and width == height:
				matrix = sympy.Matrix.eye(width)
			elif matrix_type.lower() == 'u':
				matrix = sympy.Matrix.ones(height, width)
			elif matrix_type.lower() == 'v':
				matrix = self.create_variable_matrix(width, height)
			elif matrix_type.lower() == 'r':
				matrix = self.create_random_matrix(width, height)
			else:
				invalid = True
				continue
				
			if matrix is None:
				return False
			else:
				self.matrices.append([name, matrix])
				self.single_matrix_print(self.matrices[-1][1])
				return True

	def manual_entry(self, width, height):
		# create a matrix by manual cell-by-cell data entry
		i = 0
		j = 0

		new_matrix = []
		while i < height:

			new_column = []
			while j < width:

				value, status = self.input_value(i, n=j, col_mode=True)

				if status == 'back':
					if j > 0:
						j -= 1
						new_column.pop(-1)
					elif j == 0:
						if i > 0:
							i -= 1
							j = width - 1
							new_column = new_matrix.pop(-1)
							new_column.pop(-1)
						elif i == 0:
							return False

				elif status == 'quit':
					return False

				elif status == 'continue':
					new_column.append(value)
					j += 1
			i += 1
			j = 0
			new_matrix.append(new_column)

		return sympy.Matrix(new_matrix)

	def get_character(self):
		invalid = ""
		while True:

			clear_screen()
			if invalid:
				print(invalid)
				invalid = ""

			user_input = input("Input a character for variable matrix\n")

			if len(user_input) == 1 and user_input.isalpha():
				return user_input
			else:
				invalid = "Invalid choice"

	def get_float_input(self, prompt):
		invalid = False
		while True:

			clear_screen()
			if invalid:
				print("Invalid input. Please enter a number or a fraction.")
				invalid = False

			user_input = input(prompt)
			if user_input.lower() == 'x':
				return None
			try:
				# Convert input to number or fraction
				scaling_factor = Fraction(user_input)
				scaling_factor = float(scaling_factor)
				return scaling_factor
			except ValueError:
				invalid = True
				continue

	def create_random_matrix(self, width, height):
		invalid = ""
		while True:
			clear_screen()
			if invalid:
				print(invalid)
				invalid = ""

			print("Random Matrix Options:")
			print("1) Bernoulli")
			print("2) Discrete uniform")
			print("3) Binomial")
			print("4) Poisson")
			print("5) Geometric")
			print("6) Hypergeometric")
			print("7) Negative binomial")

			distribution = input("Choose an option ('x' to go back): ")
			if distribution.lower() == 'x':
				return None

			if distribution == '1':  # Bernoulli
				p = self.get_float_input("Enter the probability of success: ")
				matrix = np.random.binomial(1, p, (height, width))
				return sympy.Matrix(matrix)

			elif distribution == '2':  # Discrete uniform
				lower_bound = int(self.get_float_input("Enter the lower integer bound: "))
				upper_bound = int(self.get_float_input("Enter the upper integer bound: "))
				matrix = np.random.randint(lower_bound, upper_bound + 1, (height, width))
				return sympy.Matrix(matrix)

			elif distribution == '3':  # Binomial
				n = int(self.get_float_input("Enter the number of trials: "))
				p = self.get_float_input("Enter the probability of success: ")
				matrix = np.random.binomial(n, p, (height, width))
				return sympy.Matrix(matrix)

			elif distribution == '4':  # Poisson
				lam = self.get_float_input("Enter the lambda parameter: ")
				matrix = np.random.poisson(lam, (height, width))
				return sympy.Matrix(matrix)

			elif distribution == '5':  # Geometric
				p = self.get_float_input("Enter the probability of success: ")
				matrix = np.random.geometric(p, (height, width))
				return sympy.Matrix(matrix)

			elif distribution == '6':  # Hypergeometric
				M = int(self.get_float_input("Enter the total number of objects: "))
				n = int(self.get_float_input("Enter the number of type 1 objects: "))
				N = int(self.get_float_input("Enter the number of draws: "))
				matrix = np.random.hypergeometric(n, M - n, N, (height, width))
				return sympy.Matrix(matrix)

			elif distribution == '7':  # Negative binomial
				r = int(self.get_float_input("Enter the number of successes: "))
				p = self.get_float_input("Enter the probability of success: ")
				matrix = np.random.negative_binomial(r, p, (height, width))
				return sympy.Matrix(matrix)

			else:
				invalid = "Invalid selection."

	def create_variable_matrix(self, width, height):
		matrix = []

		invalid = False
		while True:

			clear_screen()
			if invalid:
				print("Invalid input. Please enter 1, 2 or 3.")
				invalid = False

			print("Matrix variable options:")
			print("1) All variables unique (a,b,c...).")
			print("2) All variables unique (double index).")
			if width == height:
				print("3) Diagonal matrix")
			option = input("Choose an option ('x' to go back): ")

			if option.lower() == 'x':
				return None
			else:
				character = self.get_character()

				if option == '1':
					counter = 0
				if option == '3':
					user_input = input("All variables the same? ('y' for yes)\n")


				for i in range(height):
					row = []
					for j in range(width):
						if option == '1':
							# Variables are unique to each column
							row.append(sympy.symbols(f'{chr(ord(character) + counter)}'))
							counter += 1
						elif option == '2':
							# All variables are unique double index
							row.append(sympy.symbols(f'{character}_{i+1}{j+1}'))
						elif option == '3' and width == height:
							# All variables are unique a,b,c 
							if j == i:
								if user_input.lower() == 'y':
									row.append(sympy.symbols(f'{chr(ord(character))}'))
								else:
									row.append(sympy.symbols(f'{chr(ord(character) + i)}'))
							else:
								row.append(0)
						else:
							invalid = True
							continue
					matrix.append(row)
				break

		return sympy.Matrix(matrix)

	# Single Matrix Operations
	# # # #	# # # #	# # # #	# 


	def eval_variables_operation(self, index, substitutions, new):
		matrix_title, matrix = self.matrices[index]
		new_matrix = matrix.subs(substitutions)
		if new:
			# Append new matrix to matrices
			self.matrices.append([get_name(f"{matrix_title}_evaluated"), new_matrix])
		else:
			self.matrices[index][1] = new_matrix

	def eval_variables(self, matrix_num, new=False, reverse_mode=False):

		# Add functionality to perform simultenous operations on 
		# a array of matrices. Each single matrix operation will
		# treat matrix_num as a tuple of list indices, print matrices
		# side-by-side, perform operations a selected matrices or print
		# error messages if matrices are not compatible

		if isinstance(matrix_num, list):
			symbols_in_matrix = set()

			# Iterate over the indices in the list
			for index in matrix_num:
				if index < len(self.matrices):
					matrix_title, matrix = self.matrices[index]
					# Add symbols in the matrix to the set
					symbols_in_matrix.update(matrix.free_symbols)
				else:
					print(f"Index {index} is out of range.")
					# Handle error or break loop
			symbols_in_matrix = sorted(list(symbols_in_matrix), key=lambda s: str(s))
			if not symbols_in_matrix:
				print("No variables to evaluate in the selected matrices.")

		else:
			matrix_title, matrix = self.matrices[matrix_num]
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

			self.print_matrix_indices(matrix_num)

			print(f"Enter a value: 'c' to keep constant,'-' to go back, 'x' to quit")
			symbol = symbols_in_matrix[i]

			user_input = input(f"  >   {sympy.pretty(symbol)}: ")
			if user_input.lower() == 'c':
				i += 1
			elif user_input.lower() == '-' and i != 0:
				i -= 1
			elif user_input.lower() == 'x':
				return False
			else:
				try:
					# Convert input to number or fraction
					value = sympy.sympify(user_input)
					substitutions[symbol] = value
					i += 1
				except (sympy.SympifyError):
					invalid = True

		# Create new matrix with evaluated variables
		if substitutions != {}:
			if isinstance(matrix_num,list):
				for i in matrix_num:
					self.eval_variables_operation(i,substitutions,new)
			else:
				self.eval_variables_operation(matrix_num,substitutions,new)
			return True
		else:
			print("All variables where constant")
			input("Enter to continue")
			return False

	def get_scale_factor(self, matrix_num, message=""):

		invalid = False
		while True:

			clear_screen()
			if invalid:
				print("Invalid input. Please enter a number or a fraction.")
				invalid = False

			if message:
				print(message)

			self.print_matrix_indices(matrix_num)

			user_input = input("Enter the scaling factor (e.g., 3 or 1/2, 'x' to go back): ")
			if user_input.lower() == 'x':
				return None
			try:
				# Convert input to number or fraction
				scale_factor = sympy.sympify(user_input)
				return scale_factor
			except (sympy.SympifyError):
				invalid = True


	def get_row_number(self, message, matrix_num, col=False):
		invalid = ""
		while True:

			clear_screen()
			if invalid:
				print(invalid)
				invalid = ""

			if isinstance(matrix_num,list):
				self.print_matrix_indices(matrix_num)
				_, matrix = self.matrices[matrix_num[0]]
			else:
				matrix_title, matrix = self.matrices[matrix_num]
				print(f"\nMatrix {matrix_title}:\n")
				self.single_matrix_print(matrix)

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

	def scale_matrix_operation(self, index, scaling_factor, new):
		if new:
			matrix_title, matrix = self.matrices[index]
			new_matrix = self.matrices[index][1] * scaling_factor
			self.matrices.append([get_name(f"({scaling_factor}){matrix_title}"), new_matrix])
		else:
			self.matrices[index][1] *= scaling_factor

	def scale_matrix(self, matrix_num, new=False, reverse_mode=False):
		
		scaling_factor = self.get_scale_factor(matrix_num)
		if scaling_factor is not None:
			
			if isinstance(matrix_num, list):
				for i in matrix_num:
					self.scale_matrix_operation(i, scaling_factor, new)
			else:
				self.scale_matrix_operation(matrix_num, scaling_factor, new)

			return True

	def same_height(self, indices):
		h = self.matrices[indices[0]][1].rows
		for i in indices:
			if self.matrices[i][1].rows != h:
				return False
		return True

	def scale_row_operation(self, index, row_num, scaling_factor, new):
		matrix_title ,matrix = self.matrices[index]
		new_matrix = sympy.Matrix(matrix.tolist())
		new_matrix[row_num,:] = new_matrix[row_num,:] * scaling_factor
		new_matrix[row_num,:].simplify
		if new:
			self.matrices.append([get_name(f"({scaling_factor})R{row_num + 1}\\rightarrow R{row_num + 1}"), new_matrix])
		else:
			self.matrices[index][1] = new_matrix

	def scale_row(self, matrix_num, new=False, reverse_mode=False):

		if isinstance(matrix_num,list):
			if not self.same_height(matrix_num):
				print("Row Operations restricted to matrices of the same height")
				input("Select different matrices. Enter to continue\n")
				return False
		
		invalid = False
		while True:
			clear_screen()

			self.print_matrix_indices(matrix_num)
			
			# Get row number
			row_num = self.get_row_number("Enter the row number to scale", matrix_num)
			if row_num is None:
				return False

			# Get scaling factor
			scaling_factor = self.get_scale_factor(matrix_num)
			if scaling_factor is None:
				return False

			# Scale row elements
			if isinstance(matrix_num,list):
				for i in matrix_num:
					if reverse_mode == False or i == matrix_num[0]:
						self.scale_row_operation(i, row_num, scaling_factor, new)
					else:
						self.scale_row_operation(i, row_num, 1/scaling_factor, new)
			else:
				self.scale_row_operation(matrix_num, row_num, scaling_factor, new)
		
			# Return from the method
			return True
		
	def rearrange_operation(self, index, row1, row2, new):
		matrix_title ,matrix = self.matrices[index]
		new_matrix = sympy.Matrix(matrix.tolist())
		new_matrix.row_swap(row1, row2)
		if new:
			self.matrices.append([get_name(f"R{row1 + 1}\\leftrightarrow R{row2 + 1}"), new_matrix])
		else:
			self.matrices[index][1] = new_matrix


	def rearrange(self, matrix_num, new=False, reverse_mode=False):

		if isinstance(matrix_num,list):
			if not self.same_height(matrix_num):
				print("Row Operations restricted to matrices of the same height")
				input("Select different matrices. Enter to continue\n")
				return False
		
		# get row1
		clear_screen()
		self.print_matrix_indices(matrix_num)

		row1 = self.get_row_number("Enter the first row number", matrix_num)
		if row1 is None:
			return False

		# get row2
		clear_screen()
		self.print_matrix_indices(matrix_num)

		row2 = self.get_row_number("Enter the second row number", matrix_num)
		if row2 is None:
			return False

		if isinstance(matrix_num,list):
			for i in matrix_num:
				self.rearrange_operation(i, row1, row2, new)
		else:
			self.rearrange_operation(matrix_num, row1, row2, new)	

		return True

	def scale_and_combine_operation(self, index, row_to_scale, row_to_add, scaling_factor, new):
		# Scale and combine
		matrix_title, matrix = self.matrices[index]
		new_matrix = sympy.Matrix(matrix.tolist())
		scaled_row = new_matrix.row(row_to_scale) * scaling_factor
		combined_row = new_matrix.row(row_to_add) + scaled_row
		combined_row.simplify()

		# Update matrix with new row
		new_matrix[row_to_add,:] = combined_row

		if new:
			self.matrices.append([get_name(f"(R{row_to_add + 1}) + ({scaling_factor})(R{row_to_scale + 1})\\rightarrow R{row_to_add + 1}"), new_matrix])
		else:
			self.matrices[index][1] = new_matrix

	def scale_and_combine(self, matrix_num, new=False, reverse_mode=False):

		if isinstance(matrix_num,list):
			if not self.same_height(matrix_num):
				print("Row Operations restricted to matrices of the same height")
				input("Select different matrices. Enter to continue\n")
				return False

		# Get row to scale
		clear_screen()
		self.print_matrix_indices(matrix_num)

		row_to_scale = self.get_row_number("Enter the row to scale", matrix_num)
		if row_to_scale is None:
			return False

		# Get row to add to
		clear_screen()
		self.print_matrix_indices(matrix_num)

		row_to_add = self.get_row_number("Enter the row to add to", matrix_num)
		if row_to_add is None:
			return False

		# Get scaling factor
		scaling_factor = self.get_scale_factor(matrix_num)

		if isinstance(matrix_num, list):
			for i in matrix_num:
				if reverse_mode == False or i == matrix_num[0]:
					self.scale_and_combine_operation(i, row_to_scale, row_to_add, scaling_factor, new)
				else:
					self.scale_and_combine_operation(i, row_to_scale, row_to_add, -scaling_factor, new)
		else:
			self.scale_and_combine_operation(matrix_num, row_to_scale, row_to_add, scaling_factor, new)

		return True


	def echelon_form(self,matrix_num,new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		if new:
			new_matrix = matrix.echelon_form()
			self.matrices.append([get_name(f"{matrix_title}_\"echelon_form\""), new_matrix])
		else:
			self.matrices[matrix_num][1] = matrix.echelon_form()
		return True


	def rref(self,matrix_num,new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		if new:
			new_matrix = matrix.rref(pivots=False)
			self.matrices.append([get_name(f"{matrix_title}_\"rref\""), new_matrix])
		else:
			self.matrices[matrix_num][1] = matrix.rref(pivots=False)
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
				self.matrices.append([get_name(f"{matrix_title}^-1"), inverted_matrix])
			else:
				self.matrices[matrix_num][1] = inverted_matrix
			return True
		except ValueError as e:
			print("Error: ", e)
			input("Enter to continue")
			return False

	def cofactor_matrix(self, matrix_num, new=False):
		matrix_title, matrix = self.matrices[matrix_num]

		# Check if the matrix is square
		if matrix.shape[0] != matrix.shape[1]:
			print("Error: Only square matrices have cofactors.")
			input("Enter to continue")
			return False

		try:
			# Compute the cofactor matrix of the matrix
			cofactor_matrix = sympy.Matrix(matrix.shape[0], matrix.shape[1], lambda i, j: matrix.cofactor(i, j))

			# Append the cofactor matrix to the matrices list
			if new:
				self.matrices.append([get_name(f"{matrix_title}_\"cofactor\""), cofactor_matrix])
			else:
				self.matrices[matrix_num][1] = cofactor_matrix
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
				self.matrices.append([get_name(f"{matrix_title}_\"adjugated\""), adjugated_matrix])
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
			self.matrices.append([get_name(f"{matrix_title}^T"), matrix.transpose()])
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
			self.single_matrix_print(matrix)  

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

	def exponential_of_matrix(self, matrix_num, new=False):
		matrix_title, matrix = self.matrices[matrix_num]

		# Check if the matrix is square
		if matrix.shape[0] != matrix.shape[1]:
			print("Error: Only square matrices can have their exponential calculated.")
			input("Enter to continue")
			return False

		# Compute the exponential of the matrix
		exp_matrix = sympy.exp(matrix)

		# Append or replace the matrix in the matrices list
		if new:
			self.matrices.append([get_name(f"exp({matrix_title})"), exp_matrix])
		else:
			self.matrices[matrix_num][1] = exp_matrix
		return True

	def complex_conjugate(self, matrix_num, new=False):
		matrix_title, matrix = self.matrices[matrix_num]

		# Compute the complex conjugate of each element in the matrix
		conjugate_matrix = matrix.applyfunc(lambda x: x.conjugate())

		# Append or replace the matrix in the matrices list
		if new:
			self.matrices.append([get_name(f"{matrix_title}_conjugate"), conjugate_matrix])
		else:
			self.matrices[matrix_num][1] = conjugate_matrix
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
			self.single_matrix_print(matrix)  

			print("Description: Use your matrix 'A' to transform a vector 'x' (right-multiply 'Ax')")
			print("Enter values for the 'x' vector, '-' to go back, 'x' to quit:")

			value, status = self.input_value(i, n=0)
			if status == 'back':
				i -= 1
				x_vector = x_vector[:-1]  # remove the last value added
			elif status == 'quit':
				return False

			elif status == 'continue':
				x_vector.append(value)
				i += 1



		# Convert 'x' vector to a column Matrix
		x_vector = sympy.Matrix(x_vector)

		while True:

			# Left multiply x by matrix
			try:
				result_matrix = matrix * x_vector

				# Print the result
				print(f"\nResult of multiplying vector 'x' by matrix '{matrix_title}':\n")
				print(sympy.pretty(result_matrix))

				# Save the result
				user_input = input(f"\nMultiply {matrix_title} by {result_matrix.tolist()} again? ('y' for yes, else quit)\n > > > ")
				if user_input.lower() == 'y':
					x_vector = result_matrix
					continue
				elif user_input.lower() == 'n':
					new=True
				if new:
					self.matrices.append([get_name(f"({matrix_title}){x_vector.tolist()}"), result_matrix])	
				else:
					user_input = input(f"Overwrite {matrix_title} with {result_matrix.tolist()}? ('y' to confirm)')\n > > > ")	
					if user_input.lower() == 'y':
						self.matrices[matrix_num][1] = result_matrix
						self.matrices[matrix_num][0] = get_name(f"({matrix_title}){x_vector.tolist()}")
				return True
			except Exception as e:
				print("Error: ", e)
				input("Enter to continue")
				return False


	def projection_onto_subspace(self, matrix_num, new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		num_rows,num_cols = matrix.shape
		
		# Check if all the columns of 'A' are orthogonal to each other
		is_orthogonal = all(sympy.Matrix.dot(matrix[:, i], matrix[:, j]) == 0 for i in range(num_cols) for j in range(i + 1, num_cols))
		

		# Ask the user to input a point in R^n
		y_vector = []
		
		i = 0
		while i < num_rows:

			clear_screen()

			print(f"\nMatrix {matrix_title}:\n")
			self.single_matrix_print(matrix) 

			print(f"All columns of '{matrix_title}' are orthogonal to each other: {is_orthogonal}")

			# print("Could use a description of what this function is doing :)")
			print("Enter values for the 'y' vector, '-' to go back, 'x' to quit:")


			value, status = self.input_value(i, n=0)
			if status == 'back':
				i -= 1
				y_vector = y_vector[:-1]  # remove the last value added
			elif status == 'quit':
				return False
			elif status == 'continue':
				y_vector.append(value)
				i += 1

		format_f = expression_output_options()

		clear_screen()
		# Convert 'y' vector to a column Matrix
		y_vector = sympy.Matrix(y_vector)

		# Calculate the projection and orthogonal vectors
		projection = sympy.Matrix.zeros(y_vector.shape[0], 1)
		print("\nCalculated projection weights for each vector:")
		for i in range(num_cols):
			w = matrix[:, i]
			debug_print(f"Matrix.dot(y_vector, w) {sympy.Matrix.dot(y_vector, w)}")
			debug_print(f"Matrix.dot(w, w) {sympy.Matrix.dot(w, w)}")

			weight = (sympy.Matrix.dot(y_vector, w) / sympy.Matrix.dot(w, w))
			projection += weight * w
			print(f"  Weight for vector {i+1}: {weight}")

		y_hat = projection
		orthogonal_vector = y_vector - y_hat

		print("\nCalculated vector y_hat:")
		print(format_f(y_hat))

		print("\nOrthogonal vector z = y - y_hat:")
		print(format_f(orthogonal_vector))

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
				orthogonal_vector -= orthogonal_basis[j] * (sympy.Matrix.dot(orthogonal_basis[j], w) / sympy.Matrix.dot(orthogonal_basis[j], orthogonal_basis[j]))
			orthogonal_basis.append(orthogonal_vector)

			# Compute orthonormal basis
			orthonormal_vector = orthogonal_vector.normalized()
			orthonormal_basis.append(orthonormal_vector)

		# Print the orthogonal and orthonormal bases
		print("\nOrthogonal basis:")
		orthogonal_matrix = sympy.Matrix([vector.T for vector in orthogonal_basis]).T
		self.single_matrix_print(orthogonal_matrix)
		for vector in orthogonal_basis:
			print(sympy.pretty(vector))
		orthonormal_matrix = sympy.Matrix([vector.T for vector in orthonormal_basis]).T
		print("\nOrthonormal basis:")
		self.single_matrix_print(orthonormal_matrix)
		for vector in orthonormal_basis:
			print(sympy.pretty(vector))
		input("Enter to continue")

		# Save the orthogonal basis as a new matrix if desired
		if new:
			user_input = input(f"Orthogonal ('g'), Orthonormal ('n'), or both ('b')? (: ")
			if user_input.lower() == 'g' or user_input.lower() == 'b':
				self.matrices.append([get_name(f"{matrix_title}_\"orthogonal_basis\""), orthogonal_matrix])
			if user_input.lower() == 'n' or user_input.lower() == 'b':
				self.matrices.append([get_name(f"{matrix_title}_\"orthonormal_matrix\""), orthonormal_matrix])

			return True
		else:
			self.matrices[matrix_num][1] = orthogonal_matrix
			return True



	def determinant(self, matrix_num, new=False):
		matrix_title, matrix = self.matrices[matrix_num]

		# Check if the matrix is square
		if matrix.shape[0] != matrix.shape[1]:
			print("Error: Only square matrices can have a determinant.")
			input("Enter to continue")
			return False

		try:
			# Calculate the determinant
			determinant = sympy.Matrix([matrix.det()])

			# Print the determinant
			clear_screen()
			print(f"\nDeterminant of matrix '{matrix_title}':\n")
			self.single_matrix_print(determinant)
			input("Enter to continue")

			if new:
				self.matrices.append([get_name(f"det({matrix_title})"), determinant])
				return True
			else:
				self.matrices[matrix_num][1] = determinant
				return True

		except ValueError as e:
			print("Error: ", e)
			input("Enter to continue")
			return False

	def minus_lambda(self, matrix_num, new=False, reverse_mode=False):
		
		matrix_title, matrix = self.matrices[matrix_num]
		
		if matrix.shape[0] != matrix.shape[1]:
			print("Error: This operation is only appropriate for square matrices.")
			input("Enter to continue")
			return False

		invalid = ""

		# build λ matrix
		lambda_matrix = []
		for i in range(matrix.rows):
			row = []
			for j in range(matrix.cols):
				if j == i:
					row.append(sympy.symbols('λ'))
				else:
					row.append(0)
			lambda_matrix.append(row)

		lambda_matrix = sympy.Matrix(lambda_matrix)

		while True:
			# Subtract matrices
			try:
				result = matrix - lambda_matrix  # Use SymPy's built-in subtraction

				clear_screen()
				print(f"{matrix_title} - λ: ")
				self.single_matrix_print(result)
			except ValueError:
				print("Matrix subtraction failed. Please make sure the matrices are compatible for subtraction.")
				input("Enter to continue")
				return False

			if new:
				self.matrices.append([get_name(f"{matrix_title} - λ"), result])
			else:
				self.matrices[indices[0]][1] = result

			return True

	def quadratic_form(self, matrix_num, new=False):
		matrix_title, matrix = self.matrices[matrix_num]

		# Check if the matrix is square
		if matrix.shape[0] != matrix.shape[1]:
			print("Error: Quadratic form can only be computed for square matrices.")
			input("Enter to continue")
			return False
		
		try:
			# Generate a column vector of length n using sympy symbols
			n = matrix.shape[0]
			x_symbols = sympy.Matrix([sympy.symbols(f'x_{i+1}') for i in range(n)])
			
			# Perform multiplication for the quadratic form x^T * A * x
			quadratic_form_result = x_symbols.T * matrix * x_symbols
			quadratic_form_result = quadratic_form_result.expand()

			# Print the quadratic form
			clear_screen()
			print(f"\nQuadratic form of matrix '{matrix_title}':\n")
			self.single_matrix_print(quadratic_form_result)
			input("Enter to continue")

			# Save the new matrix or update the existing one
			if new:
				self.matrices.append([get_name(f"QF({matrix_title})"), quadratic_form_result])
			else:
				self.matrices[matrix_num][1] = quadratic_form_result

			return True

		except ValueError as e:
			print("Error: ", e)
			input("Enter to continue")
			return False


	def input_value(self, i, n=None, col_mode=False):

		while True:
			user_input = input(f"  >   row {i+1}, column {n+1}: ")
			
			navigation_flag = True if (i == 0 and col_mode == False) or (n == 0 and i == 0) else False

			if user_input.lower() == '-' and not navigation_flag:
				return None, 'back'
			elif user_input.lower() == 'x':
				return None, 'quit'
			else:
				try:
					# Convert input to number or fraction
					value = sympy.sympify(user_input)
					return value, 'continue'
				except (sympy.SympifyError):
					print("Invalid input. Please enter a number or a fraction.")


	def modify_row(self, matrix_num,new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		num_cols = matrix.shape[1]
		
		row_num = self.get_row_number("Enter the row number to modify", matrix_num)
		if row_num is None:
			return False

		new_matrix = sympy.Matrix(matrix.tolist())

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
			self.matrices.append([get_name(f"{matrix_title}_\"∆row{row_num + 1}\""), new_matrix])
		else:
			self.matrices[matrix_num][1] = new_matrix
		return True


	def modify_column(self, matrix_num,new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		num_rows = matrix.shape[0]
		
		col_num = self.get_row_number("Enter the column number to modify", matrix_num,col=True)
		if col_num is None:
			return False

		new_matrix = sympy.Matrix(matrix.tolist())

		i = 0
		while i < num_rows:
			value, status = self.input_value(i, n=col_num)
			if status == 'back':
				i -= 1
			elif status == 'quit':
				return False
			elif status == 'continue':
				new_matrix[i, col_num] = value
				i += 1

		# Update the matrix
		if new:
			self.matrices.append([get_name(f"{matrix_title}_\"∆col{col_num + 1}\""), new_matrix])
		else:
			self.matrices[matrix_num][1] = new_matrix
		return True

	def modify_cell(self, matrix_num,new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		num_rows = matrix.shape[0]

		while True:
			row_num = self.get_row_number("Enter the row number of the cell to modify", matrix_num)
			if row_num is None:
				return False
			
			col_num = self.get_row_number("Enter the column number of the cell to modify", matrix_num,col=True)
			if col_num is None:
				return False

			new_matrix = sympy.Matrix(matrix.tolist())

			value, status = self.input_value(row_num, n=col_num)
			if status == 'back':
				continue
			elif status == 'quit':
				return False
			elif status == 'continue':
				new_matrix[row_num, col_num] = value

			# Update the matrix
			if new:
				self.matrices.append([get_name(f"{matrix_title}_\"∆cell{row_num + 1}{col_num + 1}\""), new_matrix])
			else:
				self.matrices[matrix_num][1] = new_matrix
			return True

	def modify_diagonal(self, matrix_num,new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		num_rows = min(matrix.shape[0],matrix.shape[1])
		
		new_matrix = sympy.Matrix(matrix.tolist())

		i = 0
		while i < num_rows:
			value, status = self.input_value(i, n=i, col_mode=True)
			if status == 'back':
				i -= 1
			elif status == 'quit':
				return False
			elif status == 'continue':
				new_matrix[i, col_num] = value
				i += 1

		# Update the matrix
		if new:
			self.matrices.append([get_name(f"{matrix_title}_\"∆diag\""), new_matrix])
		else:
			self.matrices[matrix_num][1] = new_matrix
		return True

	def modify_matrix(self, matrix_num,new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		rows, cols = matrix.shape
		
		new_matrix = sympy.Matrix(matrix.tolist())

		i = 0
		j = 0
		while i < rows:
			while j < cols:
				value, status = self.input_value(i, n=j, col_mode=True)
				if status == 'back':
					if j > 0:
						j -= 1
					elif j == 0:
						if i > 0:
							i -= 1
							j = cols - 1
						elif i == 0:
							return False

				elif status == 'quit':
					return False

				elif status == 'continue':
					new_matrix[i, j] = value
					j += 1
			i += 1
			j = 0

		# Update the matrix
		if new:
			self.matrices.append([get_name(f"{matrix_title}_\"∆\""), new_matrix])
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
		new_matrix = matrix.row_insert(matrix.rows, sympy.Matrix([new_row]))
		if new:
			self.matrices.append([get_name(f"{matrix_title}_\"add_row{new_matrix.rows}\""), new_matrix])
		else:
			self.matrices[matrix_num][1] = new_matrix
		return True


	def add_column(self, matrix_num,new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		new_column = []
		num_rows = matrix.shape[0]

		i = 0
		while i < num_rows:
			value, status = self.input_value(i, n=matrix.cols)
			if status == 'back':
				i -= 1
				new_column = new_column[:-1]  # remove the last value added
			elif status == 'quit':
				return False
			elif status == 'continue':
				new_column.append(value)
				i += 1

		# Add new column to the matrix
		new_matrix = matrix.col_insert(matrix.cols, sympy.Matrix(new_column))
		if new:
			self.matrices.append([get_name(f"{matrix_title}_\"add_col{new_matrix.cols}\""), new_matrix])
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
			self.matrices.append([get_name(f"{matrix_title}_\"del_row{new_matrix.rows}\""), new_matrix])
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
			self.matrices.append([get_name(f"{matrix_title}_\"del_col{new_matrix.cols}\""), new_matrix])
		else:
			self.matrices[matrix_num][1] = new_matrix

		return True

	def column_vector(self, matrix_num, new=False):
		matrix_title, matrix = self.matrices[matrix_num]
		
		# Ask the user for the column number to delete
		while True:
			if matrix.cols == 1:
				col_num = 0
			else:
				col_num = input(f"Select column (1-{matrix.cols}) or 'x' to cancel: ")
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
		column_vector = matrix.col(col_num)
		
		# Update or append the new matrix
		if new:
			self.matrices.append([get_name(f"{matrix_title}_\"col{col_num + 1}\""), column_vector])
		else:
			self.matrices[matrix_num][1] = column_vector

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

		format_f = expression_output_options()

		try:
			# Calculate the eigenvectors and eigenvalues
			eigenvectors = matrix.eigenvects()

			# Print the eigenvectors and eigenvalues
			clear_screen()
			print(f"\nMatrix {matrix_title}:\n")
			self.single_matrix_print(matrix) 

			print(f"\nEigenvectors and Eigenvalues of matrix '{matrix_title}':")

			result_matrix = sympy.Matrix([])  # Initialize an empty matrix

			for eigenvalue, multiplicity, vectors in eigenvectors:
				print(f"Eigenvalue:\n{format_f(eigenvalue)}, Multiplicity: {format_f(multiplicity)}")
				print("Eigenvectors:")
				for vector in vectors:
					print(format_f(vector))
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
		self.single_matrix_print(matrix)

		# Compute the characteristic polynomial
		char_polynomial = matrix.charpoly()

		# Extract the symbol used in the characteristic polynomial (usually 'lambda')
		lamda = char_polynomial.gens[0]

		format_f = expression_output_options()

		# Pretty print the characteristic polynomial
		print(f"The characteristic polynomial for matrix '{matrix_title}' is:")
		print(format_f(char_polynomial.as_expr()))
		input("Enter to continue")

		# Attempt to factor the characteristic polynomial
		factored_polynomial = sympy.factor(char_polynomial.as_expr())
		if factored_polynomial != char_polynomial.as_expr():
			print(f"The factored form of the characteristic polynomial is:")
			print(format_f(factored_polynomial))
		else:
			print("The characteristic polynomial could not be factored further.")
		input("Enter to continue")

		# Find all real and complex roots of the polynomial
		roots = sympy.solveset(char_polynomial.as_expr(), lamda, domain=sympy.S.Complexes)
		if roots:
			print("The roots of the characteristic polynomial are:")
			for root in roots:
				print(format_f(root))
		else:
			print("No roots found for the characteristic polynomial.")

		input("Enter to continue")

		if isinstance(roots, sympy.Union):
			roots_list = []
			for s in roots.args:
				if isinstance(s, sympy.FiniteSet):
					roots_list.extend(s)
		else:
			roots_list = list(roots)

		# Create diagonal matrix with the roots
		if roots_list:

			# Create the diagonal matrix
			result_matrix = sympy.diag(*roots_list)
		else:
			return False

		self.single_matrix_print(result_matrix)
		if new:
			if "\n" in sympy.pretty(char_polynomial.as_expr()):
				self.matrices.append([get_name(f"{matrix_title}, roots"), result_matrix])
			else:
				self.matrices.append([get_name(f"{matrix_title}, roots"), result_matrix])
		else:
			self.matrices[matrix_num][1] = result_matrix
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
		self.single_matrix_print(matrix)

		try:
			# Attempt to diagonalize the matrix
			P, D = matrix.diagonalize()
		except ValueError as e:
			print(f"Error: Matrix '{matrix_title}' cannot be diagonalized. {e}")
			input("Enter to continue")
			return False

		format_f = expression_output_options()

		# Print the diagonalized matrix and the transformation matrix
		print(f"Matrix '{matrix_title}' has been diagonalized:")
		print(f"Transformation matrix P:\n{format_f(P)}")
		print(f"Diagonalized matrix D:\n{format_f(D)}")

		# Overwrite the existing matrix or append a new one based on the 'new' argument
		if new:
			self.matrices.append([get_name(f"\\Lambda_\"{matrix_title}\""), D])
			self.matrices.append([get_name(f"S_\"{matrix_title}\""), P])
			user_input = input("Include S inverse? ('y' for yes)\n")
			if user_input.lower() == 'y':
				if P.det() != 0:
					self.matrices.append([get_name(f"S_\"{matrix_title}\"^-1"), P.inv()])
				else:
					input("S was not invertible\nEnter to continue")
		else:
			self.matrices[matrix_num][1] = D
			input("Enter to continue")
			
		return True

	def jordan_form(self, matrix_num, new=False):

		matrix_title, matrix = self.matrices[matrix_num]

		# Check if the matrix is square
		if matrix.shape[0] != matrix.shape[1]:
			print("Error: Only square matrices can have a Jordan form.")
			input("Enter to continue")
			return False

		clear_screen()
		print(f"\nMatrix {matrix_title}:\n")
		self.single_matrix_print(matrix)

		try:
			# Attempt to find the Jordan form of the matrix
			P, J = matrix.jordan_form()
		except ValueError as e:
			print(f"Error: Matrix '{matrix_title}' cannot be transformed to Jordan form. {e}")
			input("Enter to continue")
			return False

		format_f = expression_output_options()

		# Print the Jordan form and the transformation matrix
		print(f"Matrix '{matrix_title}' has been transformed to Jordan form:")
		print(f"Transformation matrix P:\n{format_f(P)}")
		print(f"Jordan matrix J:\n{format_f(J)}")

		# Overwrite the existing matrix or append a new one based on the 'new' argument
		if new:
			self.matrices.append([get_name(f"J_\"{matrix_title}\""), J])
			self.matrices.append([get_name(f"P_\"{matrix_title}\""), P])
			user_input = input("Include P inverse? ('y' for yes)\n")
			if user_input.lower() == 'y':
				if P.det() != 0:
					self.matrices.append([get_name(f"P_\"{matrix_title}\"^-1"), P.inv()])
				else:
					input("P was not invertible\nEnter to continue")
		else:
			self.matrices[matrix_num][1] = J
			input("Enter to continue")
			
		return True

	def svd_decomposition(self, matrix_num, new=False):
		matrix_title, matrix = self.matrices[matrix_num]

		clear_screen()
		print(f"\nMatrix {matrix_title}:\n")
		self.single_matrix_print(matrix)

		try:
			# Attempt to perform the SVD decomposition of the matrix
			U, Sigma, V = matrix.singular_value_decomposition()
		except ValueError as e:
			print(f"Error: The SVD decomposition of matrix '{matrix_title}' failed. {e}")
			input("Enter to continue")
			return False

		format_f = expression_output_options()

		# Print the SVD decomposition matrices
		print(f"SVD decomposition of matrix '{matrix_title}':")
		print(f"Matrix U (left singular vectors):\n{format_f(U)}")
		print(f"Matrix Sigma (singular values):\n{format_f(Sigma)}")
		print(f"Matrix V (right singular vectors, transposed):\n{format_f(V)}")

		# Overwrite the existing matrix or append new ones based on the 'new' argument
		if new:
			self.matrices.append([get_name(f"U_\"{matrix_title}\""), U])
			self.matrices.append([get_name(f"\\Sigma_\"{matrix_title}\""), Sigma])
			self.matrices.append([get_name(f"V_\"{matrix_title}\""), V])
		else:
			# SVD does not replace the original matrix; it's a decomposition
			print("The original matrix is not replaced by its SVD decomposition.")
			input("Enter to continue")
			
		input("Enter to continue")
		return True



	# Multi Matrix Operations
	# # # # # # # # # # # # #

	def choose_matrix(self, message, choice1=None):
		'''
		Get user selection from available matrices
		'''
		invalid = False
		while True:

			clear_screen()

			if invalid:
				print(f"Invalid matrix choice. Please choose a letter.")
				invalid = False
			
			print()
			self.print_heading(mode='choice')
			print()
			self.multi_matrix_print()
			print()

			matrix_choice = input(message)

			if matrix_choice.lower() == 'x':
				return None

			matrix_choice = ord(matrix_choice.upper()) - 65

			if matrix_choice in range(len(self.matrices)):
				return matrix_choice
			else:
				invalid = True


	def matrix_mult(self, indices, new=False, reverse_mode=False, transpose=False):
		'''
		Carries out matrix multiplication; allows iterated multiplication which
		is useful for computing final states of markov chains.
		'''

		invalid = ""
		while True:

			#clear_screen()
			if invalid:
				input(invalid)
				invalid = ""

			LHS_matrix_title, LHS_matrix = self.matrices[indices[0]]

			RHS_matrix_title, RHS_matrix = self.matrices[indices[1]]

			T_superscript = ""
			if transpose:
				LHS_matrix = LHS_matrix.transpose()
				T_superscript = "^T"

			# check for valid operands
			if RHS_matrix.rows != LHS_matrix.cols:
				invalid = "The matrices you selected are incompatible for multiplication"
				continue

			# if right operand is a square matrix: allow repeated multiplication
			if LHS_matrix.rows == RHS_matrix.cols and not transpose:
				iterations = input("Enter how many times would you like to perform this multiplication: ")
				try:
					iterations = int(iterations)
				except ValueError:
					iterations = 1
			else:
				iterations = 1

			result = LHS_matrix

			for i in range(iterations):

				# Multiply matrices
				try:
					result = result * RHS_matrix

					# another way of doing repeated multiplications
					# while seeing the result each time
					while iterations == 1:
						
						clear_screen()
						print(f"Result of multiplying {LHS_matrix_title}{T_superscript} by {RHS_matrix_title}: ")
						self.single_matrix_print(result)

						if RHS_matrix.rows != result.cols or transpose:
							input("Enter to continue")
							break

						user_input = input("Multiply again? ('y' for yes, any other key to stop)\n")

						if user_input.lower() != 'y':
							break
						else:
							result = result * RHS_matrix

				except ValueError:
					print("Matrix multiplication failed. Please make sure the matrices are compatible for multiplication.")
					input("Enter to continue")
					return False

			if new:
				self.matrices.append([get_name(f"({LHS_matrix_title}{T_superscript})({RHS_matrix_title})"),result])
			else:
				self.matrices[indices[0]][1] = result

			return True


	def matrix_sum(self, indices, new=False, reverse_mode=False):
		'''
		Carries out matrix subtraction.
		'''
		LHS_matrix = self.matrices[indices[0]][1]
		RHS_matrix = self.matrices[indices[1]][1]

		# Check if matrices are of the same size
		if LHS_matrix.rows != RHS_matrix.rows or LHS_matrix.cols != RHS_matrix.cols:
			input("The matrices you selected are incompatible for subtraction (different sizes)\nEnter to continue\n")
			return False

		invalid = ""
		while True:
			# Get first matrix scale factor
			message = "Enter scaling factor for LHS matrix"
			LHS_scale_factor = self.get_scale_factor(indices[0],message)
			if LHS_scale_factor is None:
				return False
			else:
				LHS_matrix_title = self.matrices[indices[0]][0]
				LHS_matrix = LHS_scale_factor * LHS_matrix


			# Get second matrix scale factor
			message = "Enter scaling factor for LHS matrix"
			RHS_scale_factor = self.get_scale_factor(indices[1],message)
			if RHS_scale_factor is None:
				return False
			else:
				RHS_matrix_title = self.matrices[indices[1]][0]
				RHS_matrix = RHS_scale_factor * RHS_matrix

			# Subtract matrices
			try:
				result = LHS_matrix + RHS_matrix  # Use SymPy's built-in subtraction

				clear_screen()
				print(f"Result of adding {LHS_scale_factor} x {LHS_matrix_title} with {RHS_scale_factor} x {RHS_matrix_title}: ")
				self.single_matrix_print(result)

			except ValueError:
				print("Matrix subtraction failed. Please make sure the matrices are compatible for subtraction.")
				input("Enter to continue")
				return False

			if new:
				self.matrices.append([get_name(f"({LHS_scale_factor}){LHS_matrix_title} + ({RHS_scale_factor}){RHS_matrix_title}"), result])
			else:
				self.matrices[indices[0]][1] = result

			return True

	def matrix_merge(self, indices, new=False, reverse_mode=False):
		'''
		Makes augment matrix with LHS + RHS
		'''
		LHS_matrix = self.matrices[indices[0]][1]
		RHS_matrix = self.matrices[indices[1]][1]
				
		if LHS_matrix.rows != RHS_matrix.rows:
			input("Invalid Matrices: must be the same height to be merged.\nEnter to continue\n")
			return False

		invalid = ""
		while True:
			# Get first matrix scale factor
			message = "Enter scaling factor for LHS matrix"
			LHS_scale_factor = self.get_scale_factor(indices[0],message)
			if LHS_scale_factor is None:
				return False
			else:
				LHS_matrix_title = self.matrices[indices[0]][0]
				LHS_matrix = LHS_scale_factor * self.matrices[indices[0]][1]


			# Get second matrix scale factor
			message = "Enter scaling factor for LHS matrix"
			RHS_scale_factor = self.get_scale_factor(indices[1],message)
			if RHS_scale_factor is None:
				return False
			else:
				RHS_matrix_title = self.matrices[indices[1]][0]
				RHS_matrix = RHS_scale_factor * self.matrices[indices[1]][1]

			# Subtract matrices
			try:
				result = sympy.Matrix.hstack(LHS_matrix, RHS_matrix)

				clear_screen()
				print(f"Result of merging ({LHS_scale_factor}){LHS_matrix_title} with ({RHS_scale_factor}){RHS_matrix_title}: ")
				self.single_matrix_print(result)

			except ValueError:
				print("Matrix subtraction failed. Please make sure the matrices are compatible for subtraction.")
				input("Enter to continue")
				return False

			if new:
				self.matrices.append([get_name(f"({LHS_scale_factor}){LHS_matrix_title} + ({RHS_scale_factor}){RHS_matrix_title}"), result])
			else:
				self.matrices[matrix_1_choice][1] = result

			return True

	# Printing Methods
	# # # # # # # # # 

	def print_heading(self, mode=None, mask=None):
		'''
		Display matrices in a compact list for selection
		'''
		for i in range(len(self.matrices)):

			if mask != None:
				if i in mask:
					continue

			file_name, matrix_data = self.matrices[i]

			# Print matrix number and file name
			if mode != 'choice':
				first_row = f"{i + 1}: '{file_name}' "
			else:
				first_row = f"{chr(i+65)}): '{file_name}' "

			# Calculate and print the dimensions
			num_rows, num_cols = matrix_data.shape
			first_row += f" [{num_rows} X {num_cols}] "

			# Limit the number of items printed to 6
			first_row += " ["
			for i in range(min(matrix_data.shape[1],6)):

				# Use SymPy string is pretty expression takes > 1 line
				if "\n" in sympy.pretty(matrix_data[0,i]):
					first_row += sympy.sstr(matrix_data[0,i])

				# SymPy pretty print if uses 1 line
				else:
					first_row += sympy.pretty(matrix_data[0,i])

				first_row += ", "

			# add closing bracket, remove comma
			first_row = first_row.strip(", ") + "]"

			# truncate after 6
			if matrix_data.shape[1] > 6:
				first_row += "..."

			# truncate to fit window
			first_row = first_row[:os.get_terminal_size().columns]
			
			print(first_row)


	def single_matrix_print(self, matrix):
		print(self.print_matrix_function(matrix))


	def print_matrix_indices(self,matrix_num):
		if isinstance(matrix_num,int):
			matrix_title, matrix = self.matrices[matrix_num]
			print(f"\n{matrix_title}:\n")
			self.single_matrix_print(matrix)

		elif isinstance(matrix_num,list):
			matrix_printer = Matrix_Calc()
			for i in matrix_num:
				matrix_printer.matrices.append(self.matrices[i])
			matrix_printer.mode = self.mode
			matrix_printer.multi_matrix_print()

	
	def multi_matrix_print(self):
		''' 
		method for side-by-side printing of matrices
		'''
		if self.matrices == []:
			print("No Matrices Loaded")
			return
		else:
			print("Your Loaded Matrices:\n")

		c = '@' if DEBUG else ' '

		# matrix_column_widths is the length of the longest column of the loaded matrices
		matrix_column_widths = self.get_width_arrays()

		# to avoid overflowing the terminal window, prints matrices in multiple levels (rows)
		# first: create a list of lists of beginning end indices of each levels
		first = 0
		window_size = os.get_terminal_size().columns
		current_row_size = 0
		level_indices = []
		height_counter = 0

		for i, width_array in enumerate(matrix_column_widths):
			current_row_size += min(width_array[0],5) * 3 + max(sum(width_array),len(self.matrices[i][0]))

			if current_row_size >= window_size:
				level_indices.append((first,i))
				current_row_size = min(width_array[0],5) * 3 + max(sum(width_array),len(self.matrices[i][0]))
				first = i

		level_indices.append((first,i + 1))

		for start, stop in level_indices:

			# retrieve height of tallest matrix
			if start == stop:
				stop += 1
			max_height = max([m[1].rows for m in self.matrices[start:stop]])

			if self.mode == 'latex' or self.mode == 'sci-latex':

				if stop != level_indices[-1][1]:
					height_counter += max_height
					continue
				else:
					line_counter = 0
					height_counter += max_height
					for matrix_title, matrix in self.matrices:
						if len(matrix_title + self.print_matrix_function(matrix)) // os.get_terminal_size().columns > 1:
							line_counter += 2 + len(matrix_title + self.print_matrix_function(matrix)) // os.get_terminal_size().columns
						print(matrix_title,end='=')
						self.single_matrix_print(matrix)
					while(line_counter != height_counter):
						print()
						line_counter += 1
					print("-" * window_size)
					return


			matrices = [m[1] for m in self.matrices[start:stop]]

			debug_print(f"max_height = {max_height}")

			# print the headers with column indices
			self.double_headrow(matrix_column_widths[start:stop], start, stop)
			
			
			for i in range(max_height):
				for n, matrix in enumerate(matrices):
					debug_print(f"\n{matrix}")
					if i < matrix.rows:
						
						# print row indices
						row = matrix.row(i).tolist()[0]
						margin = matrix_column_widths[start + n]
						if n == 0:
							print(f"{i:>{min(margin[0],5)}}", end="")
							print("—" * min(margin[0],5), end="")
						else:
							print(f"{i:>{min(margin[0],5)*2}}", end="")
							print("—" * min(margin[0],5), end="")

						# print actual row values
						self.print_row(i, row, margin)

					# print white space if this matrix is shorter than the tallest matrix
					if i >= matrix.rows:
						debug_print(f"i = {i}, len(matrix) = {len(matrix)}, i >= len(matrix) = {i >= len(matrix)}")
						margin = matrix_column_widths[start + n]

						# this needs to correspond to spacing in double_head_row function
						if n == 0:
							print(c*min(margin[0],5), end="")
						else:
							print(c*min(margin[0],5)*2, end="")
						print(c*min(margin[0],5), end="")
						for j in range(matrix.cols):
							margin = matrix_column_widths[start + n][j]
							print(c*margin, end="")
				print()
			print("-" * window_size)


	def print_row(self, index, row, matrix_column_widths):
		''' 
		print row with uniform spacing, output fraction forms for rational numbers
		'''
		for i, item in enumerate(row):
			margin = matrix_column_widths[i]
			item_evaluated = sympy.N(item)
			precision = 3
			
			debug_print(f"item = {item}, item_evaluated = {item_evaluated}")
			
			# if row item is a scalar
			if item_evaluated.is_number and self.mode != 'sci-pretty':

				if item_evaluated.is_complex and sympy.im(item_evaluated) != 0:
					real_part = float(item_evaluated.as_real_imag()[0])
					
					print_string = ''

					closest_fraction = Fraction.from_float(real_part).limit_denominator()
					if abs(closest_fraction - real_part) < 0.01:
						print_string += f"{str(closest_fraction)}"
					elif real_part % 1 < 1E-10 or real_part % 1 > 0.9999999999:
						print_string += f"{round(real_part)}"
					elif real_part % 10 < .0000000001 or real_part % 10 > 9.9999999999:
						print_string += f"{round(real_part, -1)}"
					else:
						print_string += f"{round(real_part, precision)}"

					imag_part = float(item_evaluated.as_real_imag()[1])

					sign = '﹢'
					if imag_part < 0:
						sign = "﹣"
						imag_part *= -1

					closest_fraction = Fraction.from_float(imag_part).limit_denominator()
					if abs(closest_fraction - imag_part) < 0.01:
						print_string += f"{sign}{str(closest_fraction)}i"
					elif imag_part % 1 < 1E-10 or imag_part % 1 > 0.9999999999:
						print_string += f"{sign}{round(imag_part)}i"
					elif imag_part % 10 < .0000000001 or imag_part % 10 > 9.9999999999:
						print_string += f"{sign}{round(imag_part, -1)}i"
					else:
						print_string += f"{sign}{round(imag_part, precision)}i"

					print("{:>{m}}".format(print_string,m=margin-1),end='')

				else:
					item_float = float(item_evaluated)

					closest_fraction = Fraction.from_float(item_float).limit_denominator()
					if abs(closest_fraction - item_float) < 0.01:
						print(f"{str(closest_fraction).rjust(margin)}", end="")
					elif item.evalf() % 1 < 1E-10 or item.evalf() % 1 > 0.9999999999:
						print(f"{round(item):>{margin}}", end="")
					elif item.evalf() % 10 < .0000000001 or item.evalf() % 10 > 9.9999999999:
						print(f"{round(item, -1):>{margin}}", end="")
					else:
						print(f"{round(item, precision):>{margin}}", end="")

			# if row item is an expression
			else:
				if self.mode == 'sci-pretty':
					print(f"{self.print_expr_function(item).strip():>{margin}}", end="")
				# if SymPy pretty tries to print multiple lines, just use string
				elif "\n" in sympy.pretty(item):
					print(f"{sympy.sstr(item).strip():>{margin}}", end="")
				else:
					print(f"{sympy.pretty(item).strip():>{margin}}", end="")

	def get_col_width(self,col):
		'''	
		find the widest entry in the column 
		'''
		max_len = 3
		for item in col:
			item = item[0]
			item_evaluated = sympy.N(item)
			precision = 3
			
			# if row item is a scalar
			if item_evaluated.is_number and self.mode != 'sci-pretty':

				if item_evaluated.is_complex and sympy.im(item_evaluated) != 0:
					real_part = float(item_evaluated.as_real_imag()[0])
					
					print_string = ''

					closest_fraction = Fraction.from_float(real_part).limit_denominator()
					if abs(closest_fraction - real_part) < 0.01:
						print_string += f"{str(closest_fraction)}"
					elif real_part % 1 < 1E-10 or real_part % 1 > 0.9999999999:
						print_string += f"{round(real_part)}"
					elif real_part % 10 < .0000000001 or real_part % 10 > 9.9999999999:
						print_string += f"{round(real_part, -1)}"
					else:
						print_string += f"{round(real_part, precision)}"

					imag_part = float(item_evaluated.as_real_imag()[1])

					sign = '##'

					closest_fraction = Fraction.from_float(imag_part).limit_denominator()
					if abs(closest_fraction - imag_part) < 0.01:
						print_string += f"{sign}{str(closest_fraction)}i"
					elif imag_part % 1 < 1E-10 or imag_part % 1 > 0.9999999999:
						print_string += f"{sign}{round(imag_part)}i"
					elif imag_part % 10 < .0000000001 or imag_part % 10 > 9.9999999999:
						print_string += f"{sign}{round(imag_part, -1)}i"
					else:
						print_string += f"{sign}{round(imag_part, precision)}i"

					max_len = max(max_len,len(print_string))
				else:
					# This crashes if item_evaluated is complex
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

			# if row item is an expression
			else:
				if self.mode == 'sci-pretty':
					max_len = max(max_len,len(f"{self.print_expr_function(item).strip():>{max_len}}"))
				# if SymPy pretty tries to print multiple lines, just use string
				elif "\n" in sympy.pretty(item):
					max_len = max(max_len,len(f"{sympy.sstr(item).strip():>{max_len}}"))
				else:
					max_len = max(max_len,len(f"{sympy.pretty(item).strip():>{max_len}}"))
		return max_len + 1

	def get_width_arrays(self):
		'''	
		finds the length of the longest row entry for each column for each of the loaded matrices
		returns a 3d array of the required width for each column to print cleanly for each matrix
		'''
		# create a list of empty lists

		max_len = [[] for _ in range(len(self.matrices))]
		
		for i, matrix in enumerate(self.matrices):
			m = 0
			for j in range(matrix[1].cols):
				
				m = max(m,self.get_col_width(matrix[1].col(j).tolist()))
				
				max_len[i].append(m)

		return max_len


	def double_headrow(self, matrix_column_widths, start, stop):
		'''	
		prints the head row with column indices lined up (hopefully) correctly
		'''
		c = '@' if DEBUG else ' '

		for i, matrix in enumerate(self.matrices[start:stop]):
			margin = min(matrix_column_widths[i][0],5)

			# white space to go before matrix title
			if i == 0:
				print(c*(margin-1), end="")
			else:
				print(c*(margin*2-1), end="")
			for j in range(matrix[1].cols):
				margin += matrix_column_widths[i][j]

			# print matrix title with margin spacing
			print(f"{matrix[0]:<{margin}}", end="")
			print(c,end="")
			
		print()  # Go to the next line			

		for i, matrix in enumerate(self.matrices[start:stop]):
			margin = matrix_column_widths[i][0]

			# white space to go before column indices
			if i == 0:
				print(c*min(margin,5)*2, end="")
			else:
				print(c*(min(margin,5))*3, end="")

			# print column indices with margin spacing
			for j in range(matrix[1].cols):
				margin = matrix_column_widths[i][j]
				print(f"{j:>{margin}}", end="")
			
		print()  # Go to the next line

		for i, matrix in enumerate(self.matrices[start:stop]):
			margin = matrix_column_widths[i][0]
			# white space to go before | pointing to column
			if i == 0:
				print(c*min(margin,5)*2, end="")
			else:
				print(c*(min(margin,5))*3, end="")

			# print | with margin spacing
			for j in range(matrix[1].cols):
				margin = matrix_column_widths[i][j]
				print(f"{'|':>{margin}}", end="")
		print()
