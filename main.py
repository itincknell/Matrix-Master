

from matrix import Matrix_Calc, clear_screen
from math import ceil
from sympy import pprint
from pyfiglet import figlet_format

def main_program():
	clear_screen()
	print()
	print(figlet_format("        Matrix Master",font='doom',width=150))
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
		Matrix_Calc.print()

		print("\nPlease choose an option:")
		print("1) Load File")
		print("2) Save to File")
		print("3) Delete Files")
		print("4) >> Matrix Operations")
		print("5) Create Matrix")
		print("6) Delete Matrix")
		print("7) Quit")

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
		Matrix_Calc.print()

		print("\nOperations:")
		print("1) Single matrix operations")
		print("2) Two matrix operations")
		print("3) Delete a matrix")
		print("x) Go back")
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
		pprint(matrix)  # pretty print the matrix

		categories = {
			'1': ('Scale and Combine, Echelon Form, RREF', {
				'1': ('Scale row', Matrix_Calc.scale_row),
				'2': ('Rearrange rows', Matrix_Calc.rearrange),
				'3': ('Scale and combine rows', Matrix_Calc.scale_and_combine),
				'4': ('Transform to Echelon form', Matrix_Calc.echelon_form),
				'5': ('Transform to RREF', Matrix_Calc.rref)
			}),
			'2': ('Replace variables, Add, Delete, Modify', {
				'1': ('Replace variables', Matrix_Calc.eval_variables),
				'2': ('Add a row', Matrix_Calc.add_row),
				'3': ('Add a column', Matrix_Calc.add_column),
				'4': ('Modify a row', Matrix_Calc.modify_row),
				'5': ('Modify a column', Matrix_Calc.modify_column),
				'6': ('Delete a row', Matrix_Calc.delete_row),
				'7': ('Delete a column', Matrix_Calc.delete_column),
			}),
			'3': ('Scale, Tranpose, Invert, etc.', {
				'1': ('Scale matrix', Matrix_Calc.scale_matrix),
				'2': ('Transpose matrix', Matrix_Calc.transpose_matrix),
				'3': ('Raise matrix to a power', Matrix_Calc.raise_matrix_to_power),
				'4': ('Invert Matrix', Matrix_Calc.invert_matrix),
				'5': ('Adjugate',Matrix_Calc.adjugate_matrix),
			}),
			'4': ('Eigenvalues, Determinant, Quick Product', {
				'1': ('Eigenvalues', Matrix_Calc.eigenvects),
				'2': ('Determinant', Matrix_Calc.determinant),
				'3': ('Transform Vector',Matrix_Calc.transform_vector),
				'4': ('Diagonalization', Matrix_Calc.diagonalization),
				'5': ('Characteristic Polynomial', Matrix_Calc.char_poly),
				'6': ('Projection onto subspace',Matrix_Calc.projection_onto_subspace),
				'7': ('Orthogonal, orthonormal basis',Matrix_Calc.gram_schmidt),
			}),
			'x': ('Go back', None)
		}

		print("\nCategories of Single Matrix Operations:")
		for key, (description, _) in categories.items():
			print(f"{key}) {description}")

		category_choice = input("Please enter your choice of category (1-3, x): ")

		if category_choice in categories:
			description, operations = categories[category_choice]
			if operations is None:
				return
			else:
				while True:

					clear_screen()
					if invalid:
						print(invalid)
						invalid = ""
					
					print(f"\nMatrix {matrix_title}:\n")
					pprint(matrix)  # pretty print the matrix

					print(f"\nOperations in {description}:")
					for key, (op_description, _) in operations.items():
						print(f"  {key}) {op_description}")
					operation_choice = input(f"Please enter your choice of operation (1-{len(operations.values())}, x): ")
					if operation_choice.lower() == 'x':
						break
					elif operation_choice in operations:
						string = "Create new matrix? ('n' for new, any other key to to overwrite) "
						make_new = input(f"\n" + "* " * (len(string)//2) + f"\n{string}\n> > > ")
						make_new = True if make_new.lower() == 'n' else False

						op_description, operation = operations[operation_choice]
						if operation(matrix_num,make_new):
							if make_new:
								matrix_num = len(Matrix_Calc.matrices) - 1
							matrix_title, matrix = Matrix_Calc.matrices[matrix_num]

							clear_screen()
							print(f"{op_description} operation successfully executed.")
							print(f"\nMatrix {matrix_title}:\n")
							pprint(matrix)
							input("\nEnter to continue")
					else:
						invalid = f"Invalid operation choice. Please choose between 1 and {len(operations.values())}, or 'x' to go back."
		else:
			invalid = "Invalid category choice. Please choose between 1 and 3, or 'x' to go back."


def multi_matrix_operations(Matrix_Calc):

	invalid = False
	while True:

		clear_screen()
		if invalid:
			print("Invalid selection, please enter '1' or 'x' (more options coming soon)")
			invalid = False
		Matrix_Calc.print()
		print("\nOperations:")
		print("1) Multiply Matrices")
		print("x) Go back")
		user_choice = input("Please enter your choice (1 or 'x'): ")

		if user_choice == '1':
			if Matrix_Calc.matrix_mult():
				print("Matrix multiplication successfully executed.\nNew matrix:")
		elif user_choice.lower() == 'x':
			return
		else:
			invalid = True


#if __name__ == "__main__":
main_program()

