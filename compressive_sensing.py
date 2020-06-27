import numpy as np
import cvxpy as cp
import scipy

def compress_sensing(A, Y, norm):
    '''
    returns a sparse signal X, subject to Y=A@X
    '''
    if norm == 'L0':
        return cs_l0(A, Y)
    elif norm == 'L1':
        return cs_l1(A, Y)
    else:
        raise ValueError
    
    
def cs_l0(A, Y):
    '''
    arguments:
     - A:          np.ndarray of shape (M, 100)
                   sampling coefficient matrix whose rows are orthonormal
                   you may assume M>=3 is always true
     - Y:          np.ndarray of shape (M,)
                   sampling result
    returns:
     - X:          np.ndarray of shape (100,)
                   sparse signal; X must have at most 3 non-zero elements 
    '''

    # Get the 100,3 combinations using the combs_idx function
    combs = combs_idx(A.shape[1], 3)

    # Initialize variables
    min_residual = 1
    min_residual_index = 0
    residuals = []
    xvals = []
    X = np.zeros((100))

    for i in range(combs.shape[0]):
        #For all combinations, get the value for vector x and the residual using least squares
        xval, res, _, _ = np.linalg.lstsq(A[:, combs[i]], Y, rcond=None)
        xvals.append(xval)
        residuals.append(res)
        if res < min_residual:
            # Get the minimum residual value from least squares
            min_residual = res
            min_residual_index = i

    residuals = np.array(residuals)
    minresid = combs[residuals.argmin()]
    #Return the best vector x with same indices as in the combinations given
    X[minresid] = xvals[min_residual_index]
    return X


def cs_l1(A, Y):
    '''
    arguments:
     - A:          np.ndarray of shape (M, 100)
                   sampling coefficient matrix whose rows are orthonormal
                   you may assume M>=3 is always true
     - Y:          np.ndarray of shape (M,)
                   sampling result
    returns:
     - X:          np.ndarray of shape (100,)
                   sparse signal, whose L1 norm is minimised
    '''

    # Initialize the variable to be estimated
    X = cp.Variable(shape=A.shape[1])

    # Define the constraints and the objective for compressive sensing by l1 norm
    constraints = [A * X == Y]
    objective = cp.Minimize(cp.norm(X, 1))

    # Solve the sparse recovery problem
    sparse_recovery = cp.Problem(objective, constraints)
    sparse_recovery.solve()
    return X.value

def combs_idx(N, k):
    '''
    returns an index array of N chooses k.

    arguments:
        -N: integer
        -k: integer

    returns:
        np array of shape (X,k)  where X is number of combinations. each row has k integers in [0,N),
        representing indices of a k-combination.
    '''

    assert N>k
    combs = []
    comb = np.arange(k)
    while comb[-1] < N:
        combs.append(comb.copy())
        for i in range(1,k+1):
            if comb[-i] != N-i:
                break # find last occurance of non-maximum elem
        # reset this last part in increasing order
        comb[-i:] = np.arange(1+comb[-i], i+1+comb[-i])

    return np.array(combs)

if __name__ == '__main__':
    
    def print_sparse(x, threshold=1e-6):
        # we will say any number with absolute value
        # below threshold is effectively zero
        idx = np.argwhere(np.abs(x)>=threshold)
        for i, e in zip(idx, x[tuple(zip(*idx))]):
            print(f'X{i}={e}')
        print("")
            
    # load the first test case
    test1 = np.load('hw3-data-test-1.npy', allow_pickle=True).item()
    A = test1['A']
    Y = test1['Y']
    print("Test case 1 with L1:")
    print_sparse(compress_sensing(A, Y, 'L1'))
    print("Test case 1 with L0:")
    print_sparse(compress_sensing(A, Y, 'L0'))
    
    
    # load the second test case
    test2 = np.load('hw3-data-test-2.npy', allow_pickle=True).item()
    A = test2['A']
    Y = test2['Y']
    print("Test case 2 with L1:")
    print_sparse(compress_sensing(A, Y, 'L1'))
    print("Test case 2 with L0:")
    print_sparse(compress_sensing(A, Y, 'L0'))
    
