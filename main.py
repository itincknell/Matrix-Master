

from matrix import Matrix_Calc, clear_screen
from math import ceil
from sympy import pprint
from pyfiglet import figlet_format
import os

def main_program():
	clear_screen()
	print()
	print(figlet_format("Matrix Master",font='cybermedium'))
	print("Welcome to Matrix Master")
	input("Press Enter to continue")

	start_user_loop(Matrix_Calc())

	
def start_user_loop(Matrix_Calc):
	# Start a loop presenting options to the user
	invalid = ""
	while True:

		clear_screen()
		if invalid:
			print(invalid)
			invalid = ""
		Matrix_Calc.multi_matrix_print()

		print("\nPlease choose an option:")
		print("1) Load File")
		print("2) Save to File")
		print("3) Delete Files")
		print("4) >> Matrix Operations")
		print("5) Create Matrix")
		print("6) Delete Matrix")
		print("7) Quit")
		print(f"p) Print Mode: {Matrix_Calc.mode}")

		user_input = input("Enter your choice: ")
		if user_input == '1':
			invalid = Matrix_Calc.load_from_file()
		elif user_input == '2':
			Matrix_Calc.save_to_file()
		elif user_input == '3':
			Matrix_Calc.delete_files()
		elif user_input == '4':
			operations(Matrix_Calc)
		elif user_input == '5':
			Matrix_Calc.create_matrix()
		elif user_input == '6':
			Matrix_Calc.delete_matrix()
		elif user_input == '7':
			return
		elif user_input.lower() == 'p':
			Matrix_Calc.toggle_print_mode()
			continue
		else:
			invalid = "Invalid selection"

def operations(Matrix_Calc):
	invalid = ""
	while True:
		
		clear_screen()
		if invalid:
			print(invalid)
			invalid = ""

		if not Matrix_Calc.matrices:
			Matrix_Calc.load_from_file()
			clear_screen()
		Matrix_Calc.multi_matrix_print()

		print("\nOperations:")
		print("1) Single matrix operations")
		print("2) Two matrix operations")
		print("3) Delete a matrix")
		print("c) Clear All")
		print("x) Go back")
		print(f"p) Print Mode: {Matrix_Calc.mode}")
		user_choice = input("Please enter your choice (1-3, x): ")


		# Single Matrix Operations
		if user_choice == '1':
			invalid = ""
			while True:

				clear_screen()
				if invalid:
					print(invalid)
					invalid = ""

				# Print available matrices
				if len(Matrix_Calc.matrices) > 1:
					print("\nPlease choose a matrix:")
					Matrix_Calc.print_heading()
						
					matrix_choice = input("\nEnter your choice ('x' to go back): ")
					if matrix_choice.lower() == 'x':
						break
					if matrix_choice.isdigit() and int(matrix_choice) - 1 in range(len(Matrix_Calc.matrices)):
						matrix_choice = int(matrix_choice) - 1
						single_matrix_operations(Matrix_Calc,matrix_choice)
						break
					else:
						invalid = f"Invalid matrix choice. Please choose a number between 1 and {len(Matrix_Calc.matrices)}."
				elif len(Matrix_Calc.matrices) == 1:
					single_matrix_operations(Matrix_Calc,0)
					break

		elif user_choice == '2':

			while len(Matrix_Calc.matrices) < 2:
				print("\nYou need at least 2 matrices for this operation.")
				if not Matrix_Calc.load_from_file(overwrite=False):
					break
			if len(Matrix_Calc.matrices) >= 2:
				multi_matrix_operations(Matrix_Calc)
				
		elif user_choice == '3':
			# Delete a matrix
			if Matrix_Calc.delete_matrix():
				return

		if user_choice.lower() == 'p':
			Matrix_Calc.toggle_print_mode()
			continue

		if user_choice.lower() == 'c':
			Matrix_Calc.matrices = []
			return

		elif user_choice.lower() == 'x':
			return
		else:
			invalid = "Invalid choice. Please choose 1, 2, 3, or x."

def single_matrix_operations(Matrix_Calc, matrix_num):

	invalid = ""
	while True:

		clear_screen()
		if invalid:
			print(invalid)
			invalid = ""

		matrix_title, matrix = Matrix_Calc.matrices[matrix_num]
		print(f"\nMatrix {matrix_title}:\n")
		Matrix_Calc.single_matrix_print(matrix)  # pretty print the matrix

		space = 24

		categories = {
			'1': ("{:<{width}}".format('EDIT MATRIX:', width=space) , 'Replace variables, Add, Delete, Modify', {
					'1': ('Replace variables', Matrix_Calc.eval_variables),
					'2': ('Add a row', Matrix_Calc.add_row),
					'3': ('Add a column', Matrix_Calc.add_column),
					'4': ('Modify a row', Matrix_Calc.modify_row),
					'5': ('Modify a column', Matrix_Calc.modify_column),
					'6': ('Modify a cell', Matrix_Calc.modify_cell),
					'7': ('Modify diagonal', Matrix_Calc.modify_diagonal),
					'8': ('Modify matrix', Matrix_Calc.modify_matrix),
					'9': ('Delete a row', Matrix_Calc.delete_row),
					'10': ('Delete a column', Matrix_Calc.delete_column),
					'11': ('Create column vector', Matrix_Calc.column_vector),
					'12': ('Create Matrix', Matrix_Calc.create_matrix)
			}),
			'2': ("{:<{width}}".format('ROW OPERATIONS:', width=space) , 'Scale and Combine, Echelon Form, RREF', {
				'1': ('Scale row', Matrix_Calc.scale_row),
				'2': ('Rearrange rows', Matrix_Calc.rearrange),
				'3': ('Scale and combine rows', Matrix_Calc.scale_and_combine),
				'4': ('Transform to Echelon form', Matrix_Calc.echelon_form),
				'5': ('Transform to RREF', Matrix_Calc.rref)
			}),	
			'3': ("{:<{width}}".format('MATRIX TRANSFORMS:', width=space) , 'Scale, Tranpose, Invert, etc.', {
				'1': ('Scale matrix', Matrix_Calc.scale_matrix),
				'2': ('Transpose matrix', Matrix_Calc.transpose_matrix),
				'3': ('Raise matrix to a power', Matrix_Calc.raise_matrix_to_power),
				'4': ('Invert Matrix', Matrix_Calc.invert_matrix),
				'5': ('Cofactor Matrix', Matrix_Calc.cofactor_matrix),
				'6': ('Adjugate', Matrix_Calc.adjugate_matrix),
				'7': ('Complex Conjugate', Matrix_Calc.complex_conjugate),
				'8': ('Exponential of Matrix', Matrix_Calc.exponential_of_matrix),
			}),
			'4': ("{:<{width}}".format('DIFFERENCE ANGLES:', width=space) , 'Inner Product AᵀA, Projection onto subspace, Gram-Schmidt Process', {
				'1': ('Inner Product', lambda matrix_num, new=False :Matrix_Calc.matrix_mult((matrix_num, matrix_num), new, reverse_mode=False, transpose=True)),
				'2': ('Projection onto subspace',Matrix_Calc.projection_onto_subspace),
				'3': ('Orthogonal, orthonormal basis',Matrix_Calc.gram_schmidt),
				'4': ('Transform Vector',Matrix_Calc.transform_vector),
				
			}),
			'5': ("{:<{width}}".format('EIGENVALUES:', width=space) , 'Determinant, Characteristic Polynomial, Diagonalization', {
				'1': ('Determinant', Matrix_Calc.determinant),
				'2': ('A - λ', Matrix_Calc.minus_lambda),
				'3': ('Eigenvalues', Matrix_Calc.eigenvects),
				'4': ('Characteristic Polynomial', Matrix_Calc.char_poly),
				'5': ('Diagonalization', Matrix_Calc.diagonalization),
				'6': ('Jordan Form', Matrix_Calc.jordan_form)
			}),
			'6': ("{:<{width}}".format('SVD:', width=space) , 'Quadratric Form, Singular Value Decompoisition (SVD)', {
				'1': ('Quadratric Form', Matrix_Calc.quadratic_form),
				'2': ('Singular Value Decompoisition', Matrix_Calc.svd_decomposition),
			}),
			'x': ('Go back','', None),
			'p': (f"Print Mode: {Matrix_Calc.mode}",'', None)
		}

		print("\nCategories of Single Matrix Operations:")
		for key, (description, description_list, ops_dict) in categories.items():
			

			print(f"{key}) {description}")
			if ops_dict is not None:
				for op_descr, _ in ops_dict.values():
					print(f"\t{op_descr}")


		category_choice = input("Please enter your choice of category (1-3, x): ")

		if category_choice == 'p':
			Matrix_Calc.toggle_print_mode()
			continue

		if category_choice in categories:
			description, _, operations = categories[category_choice]
			if operations is None:
				return
			else:

				while True:

					operations.update({'p': (f"Print Mode: {Matrix_Calc.mode}", None)})

					clear_screen()
					if invalid:
						print(invalid)
						invalid = ""
					
					print(f"\nMatrix {matrix_title}:\n")
					Matrix_Calc.single_matrix_print(matrix)  # pretty print the matrix

					print(f"\nOperations in {description}:")
					for key, (op_description, _) in operations.items():
						print(f"  {key}) {op_description}")
					operation_choice = input(f"Please enter your choice of operation (1-{len(operations.values())}, x): ")

					if operation_choice == 'p':
						Matrix_Calc.toggle_print_mode()
						continue

					if operation_choice.lower() == 'x':
						break

					# some operations do not generate a new matrix
					no_matrix_output_ops = [Matrix_Calc.determinant]

					if operation_choice in operations:
						if operations[operation_choice][1] == Matrix_Calc.create_matrix:
							if operations[operation_choice][1]():
								matrix_num = len(Matrix_Calc.matrices) - 1
								matrix_title, matrix = Matrix_Calc.matrices[matrix_num]
							continue
						
						string = "Create new matrix? ('n' for new, any other key to to overwrite) "
						make_new = input(f"\n" + "* " * (len(string)//2) + f"\n{string}\n> > > ")
						make_new = True if make_new.lower() == 'n' else False

						op_description, operation = operations[operation_choice]
						if operation(matrix_num,make_new):
							if make_new:
								matrix_num = len(Matrix_Calc.matrices) - 1
							matrix_title, matrix = Matrix_Calc.matrices[matrix_num]

							if op_description == 'A - λ':

								lambda_remove = ['Eigenvalues','Characteristic Polynomial','Diagonalization']
								delete_keys = []
								for key, (op_description, _) in operations.items():
									if op_description in lambda_remove:
										delete_keys.append(key)
								for key in delete_keys:
									del operations[key]

							clear_screen()

							#if operations[operation_choice][1] not in no_matrix_output_ops: 
							print(f"{op_description} operation successfully executed.")
							print(f"\nMatrix {matrix_title}:\n")
							pprint(matrix)
							input("\nEnter to continue")
					else:
						invalid = f"Invalid operation choice. Please choose between 1 and {len(operations.values())}, or 'x' to go back."
		else:
			invalid = f"Invalid category choice. Please choose from the aviable options, or 'x' to go back."


def multi_matrix_operations(Matrix_Calc):

	invalid = ""
	reverse_mode = False

	while True:

		clear_screen()
		if invalid:
			print(invalid)
			invalid = ""

		Matrix_Calc.multi_matrix_print()

		space = 24

		categories = {
			'1': ("{:<{width}}".format('MATRIX ALGEBRA:', width=space) , 'Product, Sum, Dot Product, Merge', {
				'1': ('Matrix product', Matrix_Calc.matrix_mult),
				'2': ('Matrix sum', Matrix_Calc.matrix_sum),
				'3': ('Dot Product', lambda indices, new=False, reverse_mode=False, transpose=True: Matrix_Calc.matrix_mult(indices, new, reverse_mode, transpose=True)),
				'4': ('Merge Matrices', Matrix_Calc.matrix_merge)

			}),
			'2': ("{:<{width}}".format('ROW OPERATIONS:', width=space) , 'Replace Variables, Rearrange Rows, Scale and Combine', {
				'1': ('Replace variables', Matrix_Calc.eval_variables),
				'2': ('Scale row', Matrix_Calc.scale_row),
				'3': ('Rearrange rows', Matrix_Calc.rearrange),
				'4': ('Scale and combine rows', Matrix_Calc.scale_and_combine),
				'r': (f"Reverse mode: {reverse_mode}", None)
				}),
			'x': ('Go back', '', None),
			'p': (f"Print Mode: {Matrix_Calc.mode}", '', None)
		}
		
		print("\nCategories of Multi Matrix Operations:")
		for key, (description, description_list, _) in categories.items():
			print(f"{key}) {description}{description_list}")

		# currently only operation of is multiplication
		category_choice = input("Please enter your choice of category (1-w, x): ")

		if category_choice.lower() == 'p':
			Matrix_Calc.toggle_print_mode()
			continue

		if category_choice in categories:

			description, _, operations = categories[category_choice]
			if operations is None:
				return
			else:
				invalid = False

				indices = Matrix_Calc.get_matrix_choices(description)

				if indices == []:
					continue
				elif description == 'Matrix Algebra' and len(indices) != 2:
					continue

				while True:

					operations.update({'p': (f"Print Mode: {Matrix_Calc.mode}", None)})

					clear_screen()
					if invalid:
						print(invalid)
						invalid = ""

					Matrix_Calc.print_matrix_indices(indices)

					print(f"\nOperations in {description}:")
					for key, (op_description, _) in operations.items():
						print(f"  {key}) {op_description}")

					operation_choice = input(f"Please enter your choice of operation (1-{len(operations.values())}, x): ")
					if operation_choice.lower() == 'x':
						indices = None
						break

					if operation_choice.lower() == 'p':
						Matrix_Calc.toggle_print_mode()
						continue

					if operation_choice.lower() == 'r':
						reverse_mode = not reverse_mode
						operations.update({'r': (f"Reverse mode: {reverse_mode}", None)})
						continue

					elif operation_choice in operations:
						string = "Create new matrix? ('n' for new, any other key to to overwrite) "
						make_new = input(f"\n" + "* " * (len(string)//2) + f"\n{string}\n> > > ")
						make_new = True if make_new.lower() == 'n' else False

						op_description, operation = operations[operation_choice]
						
						result = operation(indices, new=make_new, reverse_mode=reverse_mode)

						if description == 'Matrix Algebra':
							break
						elif result:
							indices = update_indices(indices, len(Matrix_Calc.matrices))

					else:
						invalid = f"Invalid operation choice."
		else:
			invalid = "Invalid category choice."

def update_indices(indices, matrices_len):
	choices_length = len(indices)
	return list(matrices_len - choices_length + i for i in range(choices_length))



#if __name__ == "__main__":
main_program()

