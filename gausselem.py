import numpy as np
import argparse
import copy as cp
import pickle

def echelon(Ab):
    """
    The Gauss Elimination method for solving Ax = b
    Args:
    Augmented matix [A b]
    Returns:
    Row Echelon Matrix [R f]
    """
    Ab = cp.deepcopy(Ab).astype(np.float64)
    m, n = Ab.shape
    # n - 1 : Because Ab is a augmented matix and consists of extra column
    for ith_pivot in np.arange(min(m, n - 1)):
        # Find the pivot for the ith row:
        imax = np.argmax(np.abs(Ab[ith_pivot:, ith_pivot])) + ith_pivot
        # Checking for the condition where all the rows are below pivot element are zero
        if Ab[imax, ith_pivot] == 0:
            continue
        # Swap with the max element
        Ab[[ith_pivot, imax], :] = Ab[[imax, ith_pivot], :]
        # Populating all the multiplication scale factor
        for ith_row in np.arange(ith_pivot + 1, m):
            scale_factor = Ab[ith_row, ith_pivot] / Ab[ith_pivot, ith_pivot]
            Ab[ith_row, :] -= Ab[ith_pivot, :] * scale_factor
    return Ab


def back_substitute(Ab):
    """
    Performing back sub and indicates consistency
    Args:
    Echelon Augmented matrix [R f]
    Returns:
    Consistency: -1 ---> Inconsistent
    1 ---> Consistent
    Soln: Solution Vector
    """
    Ab = cp.deepcopy(Ab)
    mat_stat = 1
    m, n = Ab.shape
    Ab_rev = np.flipud(Ab)
    nonzero_ind = Ab_rev[:, (n - 2):]
    l = get_last_non_zero_index(nonzero_ind[:, 0]) - 1
    k = get_last_non_zero_index(nonzero_ind[:, 1]) - 1
    if l != k:
        mat_stat = -1
    sol_dim = l
    solutions = np.zeros((m - sol_dim, 1))
    for idx, row in enumerate(Ab_rev[sol_dim:, :(n - sol_dim)]):
        cur_soln = 0
        for idy, col in enumerate(row):
            if idy == len(row) - 1:
                continue
            cur_soln += solutions[idy, 0] * col
        cur_soln = (row[-1] - cur_soln) / row[-idx - 2]
        solutions[-idx - 1] = cur_soln
    return mat_stat, solutions


def unpickle_objects(inputfile):
    Ab = pickle.load(open(inputfile, "rb"))
    return Ab


def arg_parser_get():
    parser = argparse.ArgumentParser(description='Finding the Echelon Form of the matrix and solution of the equations, Moreover comment about Consistency.')
    parser.add_argument('--inputfile', help="Enter the path of input file of [A b]")
    parser.add_argument('--outfile', help="Enter the path for saving the solution", default="./pickleout")
    return parser.parse_args()


def get_last_non_zero_index(d, default=None):
    rev = (idx for idx, item in enumerate(d, 1) if item)
    return next(rev, default)


if __name__ == '__main__':
    args = arg_parser_get()
    Ab = unpickle_objects(args.inputfile)
    Rf = echelon(Ab)
    consistent, soln = back_substitute(Rf)
    Data = {}
    Data["input_matrix"] = Ab
    Data["Echelon_Matrix"] = Rf
    if consistent == -1:
        Data["Is_Consistent"] = False
    else:
        Data["Is_Consistent"] = True
    Data["Solution"] = soln
    with open(args.outfile, "w") as fp:
        pickle.dump(Data, fp)
