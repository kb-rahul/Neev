# Neev
Assignment 1
Implement the Gauss elimination algorithm for computing the solution to a linear system of equations Ax = b

Requirements
The program should take as argument a single file name. The file will contain the pickle'd numpy matrix A and vector b
You need to implement the algorithm from scratch, not use a numpy routine
The output should be a numpy array containing the solution x
Your code should detect whether the system of equations is consistent or not
In case multiple solutions are possible, any one should be returned.
In case no solutions are possible, an error should be thrown


******** 
Assigment Details:
Usage:
python gausselem.py --inputfile <File Name> --outfile <File Name>

Output:
Pickle object consists of the dictionary Data with all solution related to the question
Shows: Row Echelon form, solution matrix, Consistency, input matrix
