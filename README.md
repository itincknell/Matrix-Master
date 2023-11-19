# Matrix-Master
A shell for handling SymPy matrix objects

## Overview
The main data mamber of the Matrix_Calc class is a list of 2-tuples containing 0) the title of the matrix and 1) a SymPy 'MutableDenseMatrix' object. The class has extensive methods for manipulating these matrices.

The 'create matrix' methods in the main menu allow the user to populate a matrix with SymPy expression objects. 
The 'replace variables method' in the 'single matrix operations' menu can be used to replace expressions with scalars. This is the easiest method for inputing new matrices. The other option is to type your matrix into a csv file with 1 line per row.

The use of SymPy expressions allows for the calculation of arbitrary results. For example, take a 3 x 3 matrix of variables and use the determinant method in single matrix operations. This is very useful, but it can easily lead to extremely complicated expressions that do not fit in the terminal window (you've been warned!).
