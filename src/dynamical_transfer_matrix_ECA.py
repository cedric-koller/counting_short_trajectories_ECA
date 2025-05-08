import numpy as np
import itertools
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import ArpackNoConvergence


def evolve(rule, config_1, other):
    '''
    Evolves a configuration of the Elementary Cellular Automaton (ECA) by one time step.

    Param: rule (1D array of 8 elements): the rule of the ECA in binary form (inverse of binary used by Wolfram, first 000, then 001, ...)
    Param: config_1 (1D array of 2 elements): the left and middle state of the ECA
    Param: other (int): the right state of the ECA

    Return: int: the new state of the ECA 
    '''
    code=other+config_1[1]*2+config_1[0]*4
    return rule[code]


def transfer_matrix_sparse(rule, allowed_indexes, p=0, c=2, mu=0):
    '''
    Compute the transfer matrix as a sparse dok matrix.

    Param: rule (1D array of 8 elements): the rule of the ECA in binary form (inverse of binary used by Wolfram, first 000, then 001, ...)
    Param: allowed_indexes (2D array of 2 elements): the indexes of the compatible permutations (i.e. middle element must be the same)
    Param: p (int): the size of the transient
    Param: c (int): the size of the cycle
    Param: mu (float): the Lagrange multiplier for the density of the initial state

    Return: dok_matrix: the transfer matrix
    '''

    permutations=np.array(list(itertools.product([0,1], repeat=2*(p+c))))
    permutations=permutations.reshape(permutations.shape[0],int(permutations.shape[1]/2),2)
    T=sp.dok_matrix((permutations.shape[0],permutations.shape[0]), dtype=np.double)
    T_deriv=sp.dok_matrix((permutations.shape[0],permutations.shape[0]), dtype=np.double)
    for allowed_idx in allowed_indexes:
        i=allowed_idx[0]
        j=allowed_idx[1]
        # Check cycle:
        if evolve(rule, permutations[i,-1], permutations[j,-1,1])==permutations[i,p,1]:
            # Check respect evolution:
            respect_rule=True
            for k in range(permutations.shape[1]-1):
                if evolve(rule, permutations[i,k], permutations[j,k,1])!=permutations[i,k+1,1]:
                    respect_rule=False
                    break
            if respect_rule:
                T[i,j]=np.exp(mu*permutations[i,0,1])
                T_deriv[i,j]=permutations[i,0,1]*np.exp(mu*permutations[i,0,1])
    return T, T_deriv

def transfer_matrix_sparse_ending_in_0(rule, allowed_indexes, p=0, c=2, mu=0):
    '''
    Compute the transfer matrix as a sparse dok matrix.

    Param: rule (1D array of 8 elements): the rule of the ECA in binary form (inverse of binary used by Wolfram, first 000, then 001, ...)
    Param: allowed_indexes (2D array of 2 elements): the indexes of the compatible permutations (i.e. middle element must be the same)
    Param: p (int): the size of the transient
    Param: c (int): the size of the cycle
    Param: mu (float): the Lagrange multiplier for the density of the initial state

    Return: dok_matrix: the transfer matrix
    '''

    permutations=np.array(list(itertools.product([0,1], repeat=2*(p+c))))
    permutations=permutations.reshape(permutations.shape[0],int(permutations.shape[1]/2),2)
    T=sp.dok_matrix((permutations.shape[0],permutations.shape[0]), dtype=np.double)
    T_deriv=sp.dok_matrix((permutations.shape[0],permutations.shape[0]), dtype=np.double)
    for allowed_idx in allowed_indexes:
        i=allowed_idx[0]
        j=allowed_idx[1]
        # Check end configuration all 0
        if permutations[i, p, 1]!=0:
            continue
        # Check cycle:
        if evolve(rule, permutations[i,-1], permutations[j,-1,1])==permutations[i,p,1]:
            # Check respect evolution:
            respect_rule=True
            for k in range(permutations.shape[1]-1):
                if evolve(rule, permutations[i,k], permutations[j,k,1])!=permutations[i,k+1,1]:
                    respect_rule=False
                    break
            if respect_rule:
                T[i,j]=np.exp(mu*permutations[i,0,1])
                T_deriv[i,j]=permutations[i,0,1]*np.exp(mu*permutations[i,0,1])
    return T, T_deriv


def transfer_matrix_sparse_translation_left(rule, allowed_indexes, p=1, c=1, mu=0):
    '''
    Compute the transfer matrix as a sparse dok matrix.

    Param: rule (1D array of 8 elements): the rule of the ECA in binary form (inverse of binary used by Wolfram, first 000, then 001, ...)
    Param: allowed_indexes (2D array of 2 elements): the indexes of the compatible permutations (i.e. middle element must be the same)
    Param: p (int): the size of the transient
    Param: c (int): the size of the cycle - 1
    Param: mu (float): the Lagrange multiplier for the density of the initial state

    Return: dok_matrix: the transfer matrix
    '''
    permutations=np.array(list(itertools.product([0,1], repeat=2*(p+c))))
    permutations=permutations.reshape(permutations.shape[0],int(permutations.shape[1]/2),2)
    T=sp.dok_matrix((permutations.shape[0],permutations.shape[0]), dtype=np.double)
    T_deriv=sp.dok_matrix((permutations.shape[0],permutations.shape[0]), dtype=np.double)
    for allowed_idx in allowed_indexes:
        i=allowed_idx[0]
        j=allowed_idx[1]
        # Check translation:
        if permutations[i,p,1]==permutations[i,p+c-1,1]: # To check for translation, we need at least 2 time steps
            # Check respect evolution translated:
            respect_rule=True
            for k in range(permutations.shape[1]-1):
                if evolve(rule, permutations[i,k], permutations[j,k,1])!=permutations[j,k+1,1]: # We translate left, so we compare to the second element of j
                    respect_rule=False
                    break
            if respect_rule:
                T[i,j]=np.exp(mu*permutations[i,0,1])
                T_deriv[i,j]=permutations[i,0,1]*np.exp(mu*permutations[i,0,1])
    return T, T_deriv


def transfer_matrix_sparse_translation_right(rule, allowed_indexes, p=1, c=1, mu=0):
    '''
    Compute the transfer matrix as a sparse dok matrix.

    Param: rule (1D array of 8 elements): the rule of the ECA in binary form (inverse of binary used by Wolfram, first 000, then 001, ...)
    Param: allowed_indexes (2D array of 2 elements): the indexes of the compatible permutations (i.e. middle element must be the same)
    Param: p (int): the size of the transient
    Param: c (int): the size of the cycle - 1
    Param: mu (float): the Lagrange multiplier for the density of the initial state

    Return: dok_matrix: the transfer matrix
    '''
    permutations=np.array(list(itertools.product([0,1], repeat=2*(p+c))))
    permutations=permutations.reshape(permutations.shape[0],int(permutations.shape[1]/2),2)
    T=sp.dok_matrix((permutations.shape[0],permutations.shape[0]), dtype=np.double)
    T_deriv=sp.dok_matrix((permutations.shape[0],permutations.shape[0]), dtype=np.double)
    for allowed_idx in allowed_indexes:
        i=allowed_idx[0]
        j=allowed_idx[1]
        # Check translation:
        if permutations[i,p,1]==permutations[i,p+c-1,1]:
            # Check respect evolution translated:
            respect_rule=True
            for k in range(permutations.shape[1]-1):
                if evolve(rule, permutations[i,k], permutations[j,k,1])!=permutations[i,k+1,0]: # We translate right, so we compare to the first element of i
                    respect_rule=False
                    break
            if respect_rule:
                T[i,j]=np.exp(mu*permutations[i,0,1])
                T_deriv[i,j]=permutations[i,0,1]*np.exp(mu*permutations[i,0,1])
    return T, T_deriv


def transfer_matrix_sparse_translation_right_2(rule, allowed_indexes, p=1, c=1, mu=0):
    '''
    Compute the transfer matrix as a sparse dok matrix.

    Param: rule (1D array of 8 elements): the rule of the ECA in binary form (inverse of binary used by Wolfram, first 000, then 001, ...)
    Param: allowed_indexes (2D array of 2 elements): the indexes of the compatible permutations (i.e. middle element must be the same)
    Param: p (int): the size of the transient
    Param: c (int): the size of the cycle - 1
    Param: mu (float): the Lagrange multiplier for the density of the initial state

    Return: dok_matrix: the transfer matrix
    '''
    permutations=np.array(list(itertools.product([0,1], repeat=2*(p+c))))
    permutations=permutations.reshape(permutations.shape[0],int(permutations.shape[1]/2),2)
    T=sp.dok_matrix((permutations.shape[0],permutations.shape[0]), dtype=np.double)
    T_deriv=sp.dok_matrix((permutations.shape[0],permutations.shape[0]), dtype=np.double)
    for allowed_idx in allowed_indexes:
        i=allowed_idx[0]
        j=allowed_idx[1]
        # Check translation:
        if permutations[i,p,1]==permutations[i,p+c-1,1]:
            # Check respect evolution translated:
            respect_rule=True
            for k in range(permutations.shape[1]-1):
                if k%2==0:
                    if evolve(rule, permutations[i,k], permutations[j,k,1])!=permutations[i,k+1,1]: # if k=0,2,4 we check the second element of i (no translation)
                        respect_rule=False
                        break
                else:
                    if evolve(rule, permutations[i,k], permutations[j,k,1])!=permutations[i,k+1,0]: # if k=1,3,5 we check the first element of i (translation)
                        respect_rule=False
                        break
            if respect_rule:
                T[i,j]=np.exp(mu*permutations[i,0,1])
                T_deriv[i,j]=permutations[i,0,1]*np.exp(mu*permutations[i,0,1])
    return T, T_deriv

def transfer_matrix_sparse_translation_left_2(rule, allowed_indexes, p=1, c=1, mu=0):
    '''
    Compute the transfer matrix as a sparse dok matrix.

    Param: rule (1D array of 8 elements): the rule of the ECA in binary form (inverse of binary used by Wolfram, first 000, then 001, ...)
    Param: allowed_indexes (2D array of 2 elements): the indexes of the compatible permutations (i.e. middle element must be the same)
    Param: p (int): the size of the transient
    Param: c (int): the size of the cycle - 1
    Param: mu (float): the Lagrange multiplier for the density of the initial state

    Return: dok_matrix: the transfer matrix
    '''
    permutations=np.array(list(itertools.product([0,1], repeat=2*(p+c))))
    permutations=permutations.reshape(permutations.shape[0],int(permutations.shape[1]/2),2)
    T=sp.dok_matrix((permutations.shape[0],permutations.shape[0]), dtype=np.double)
    T_deriv=sp.dok_matrix((permutations.shape[0],permutations.shape[0]), dtype=np.double)
    for allowed_idx in allowed_indexes:
        i=allowed_idx[0]
        j=allowed_idx[1]
        # Check translation:
        if permutations[i,p,1]==permutations[i,p+c-1,1]:
            # Check respect evolution translated:
            respect_rule=True
            for k in range(permutations.shape[1]-1):
                if k%2==0:
                    if evolve(rule, permutations[i,k], permutations[j,k,1])!=permutations[i,k+1,1]: # if k=0,2,4 we check the second element of i (no translation)
                        respect_rule=False
                        break
                else:
                    if evolve(rule, permutations[i,k], permutations[j,k,1])!=permutations[j,k+1,1]: # if k=1,3,5 we check the second element of j (translation)
                        respect_rule=False
                        break
            if respect_rule:
                T[i,j]=np.exp(mu*permutations[i,0,1])
                T_deriv[i,j]=permutations[i,0,1]*np.exp(mu*permutations[i,0,1])
    return T, T_deriv



def transfer_matrix_sparse_no_cycle(rule, allowed_indexes, p=2, mu=0):
    '''
    Compute the transfer matrix as a sparse dok matrix.

    Param: rule (1D array of 8 elements): the rule of the ECA in binary form (inverse of binary used by Wolfram, first 000, then 001, ...)
    Param: allowed_indexes (2D array of 2 elements): the indexes of the compatible permutations (i.e. middle element must be the same)
    Param: p (int): the size of the transient
    Param: c (int): the size of the cycle
    Param: mu (float): the Lagrange multiplier for the density of the initial state

    Return: dok_matrix: the transfer matrix
    '''

    permutations=np.array(list(itertools.product([0,1], repeat=2*(p))))
    permutations=permutations.reshape(permutations.shape[0],int(permutations.shape[1]/2),2)
    T=sp.dok_matrix((permutations.shape[0],permutations.shape[0]), dtype=np.double)
    T_deriv=sp.dok_matrix((permutations.shape[0],permutations.shape[0]), dtype=np.double)
    for allowed_idx in allowed_indexes:
        i=allowed_idx[0]
        j=allowed_idx[1]
        # Check respect evolution:
        respect_rule=True
        for k in range(permutations.shape[1]-1):
            if evolve(rule, permutations[i,k], permutations[j,k,1])!=permutations[i,k+1,1]:
                respect_rule=False
                break
        if respect_rule:
            T[i,j]=np.exp(mu*permutations[i,0,1])
            T_deriv[i,j]=permutations[i,0,1]*np.exp(mu*permutations[i,0,1])
    return T, T_deriv

def derivative_transfer_matrix_sparse_no_cycle(rule, allowed_indexes, p=2, mu=0):
    '''
    Compute the transfer matrix as a sparse dok matrix.

    Param: rule (1D array of 8 elements): the rule of the ECA in binary form (inverse of binary used by Wolfram, first 000, then 001, ...)
    Param: allowed_indexes (2D array of 2 elements): the indexes of the compatible permutations (i.e. middle element must be the same)
    Param: p (int): the size of the transient
    Param: c (int): the size of the cycle
    Param: mu (float): the Lagrange multiplier for the density of the initial state

    Return: dok_matrix: the transfer matrix
    '''

    permutations=np.array(list(itertools.product([0,1], repeat=2*(p))))
    permutations=permutations.reshape(permutations.shape[0],int(permutations.shape[1]/2),2)
    T=sp.dok_matrix((permutations.shape[0],permutations.shape[0]), dtype=np.double)
    for allowed_idx in allowed_indexes:
        i=allowed_idx[0]
        j=allowed_idx[1]
        # Check respect evolution:
        respect_rule=True
        for k in range(permutations.shape[1]-1):
            if evolve(rule, permutations[i,k], permutations[j,k,1])!=permutations[i,k+1,1]:
                respect_rule=False
                break
        if respect_rule:
            T[i,j]=permutations[i,0,1]*np.exp(mu*permutations[i,0,1])
    return T


def entropy_and_density(rule, p=0, c=1, mu=0, rs=42, method_derivative='Hellmann-Feynman'):
    '''
    Compute the entropy and density of the ECA using the transfer matrix method.

    Param: rule (1D array of 8 elements): the rule of the ECA in binary form (inverse of binary used by Wolfram, first 000, then 001, ...)
    Param: p (int): the size of the transient
    Param: c (int): the size of the cycle
    Param: mu (float): the Lagrange multiplier for the density of the initial state
    Param: rs (int): the random seed for the initial vector

    Return: float, float: the entropy and density of the ECA
    '''
    np.random.seed(rs) # set the random seed for reproducibility
    v0=np.random.rand(4**(p+c),1) # initial vector for the sparse matrix
    with open('index_compatible_permutations/indexes_compatible_permutations_'+str(p+c)+'.npy','rb') as f:
        compatible_permutations=np.load(f)
    if c!=0:
        T, T_deriv=transfer_matrix_sparse(rule, compatible_permutations, p,c,mu)
        try:
            if method_derivative=='finite difference':
                eig=np.real((spla.eigs(T, k=1, which='LR', v0=v0, maxiter=10*4**(p+c), return_eigenvectors=False)))[0]
                if eig<=0:
                    return -np.inf, -np.nan
            elif method_derivative=='Hellmann-Feynman':
                eigvals, eigvecs = spla.eigs(T, k=1, which='LR', v0=v0, maxiter=10*4**(p+c), return_eigenvectors=True)
                eig_=eigvals[0].real
                vec_right=eigvecs[:,0]  
                if eig_<=0:
                    return -np.inf, -np.nan
            else:
                raise ValueError("Method must be 'finite difference' or 'Hellmann-Feynman'")
        except ArpackNoConvergence as e:
            print("Warning: ARPACK did not converge. Returning NaN.")
            return np.nan, np.nan
        
        try:
            if method_derivative=='finite difference':
                eig2=np.real((spla.eigs(transfer_matrix_sparse(rule, compatible_permutations, p,c,mu+1e-6)[0], k=1, which='LR', maxiter=10*4**(p+c), v0=v0,return_eigenvectors=False)))[0]
            elif method_derivative=='Hellmann-Feynman':
                eigvals, eigvecs = spla.eigs(T.T, k=1, which='LR', v0=v0, maxiter=10*4**(p+c), return_eigenvectors=True)
                vec_left=eigvecs[:,0]
            else:
                raise ValueError("Method must be 'finite difference' or 'Hellmann-Feynman'")
        except ArpackNoConvergence as e:
            print("Warning: ARPACK did not converge. Returning NaN.")
            return np.nan, np.nan

        if method_derivative=='finite difference': 
            phi=np.log(eig)
            rho=(np.log(eig2)-np.log(eig))/1e-6
            return phi-mu*rho, rho
        elif method_derivative=='Hellmann-Feynman':
            norm =vec_left.conj().T@vec_right
            phi_new = np.log(eig_)
            rho_new = (vec_left.conj().T@T_deriv@vec_right/(eig_*norm)).real
            return phi_new-mu*rho_new, rho_new
        else:
            raise ValueError("Method must be 'finite difference' or 'Hellmann-Feynman'")
    else:
        T, T_deriv=transfer_matrix_sparse_no_cycle(rule, compatible_permutations, p,mu)
        try:
            if method_derivative=='finite difference':
                eig=np.real((spla.eigs(T, k=1, which='LR', v0=v0, maxiter=10*4**(p+c), return_eigenvectors=False)))[0]
                if eig<=0:
                    return -np.inf, -np.nan
            elif method_derivative=='Hellmann-Feynman':
                eigvals, eigvecs = spla.eigs(T, k=1, which='LR', v0=v0, maxiter=10*4**(p+c), return_eigenvectors=True)
                eig_=eigvals[0].real
                vec_right=eigvecs[:,0]  
                if eig_<=0:
                    return -np.inf, -np.nan
            else:
                raise ValueError("Method must be 'finite difference' or 'Hellmann-Feynman'")
        except ArpackNoConvergence as e:
            print("Warning: ARPACK did not converge. Returning NaN.")
            return np.nan, np.nan
        
        try:
            if method_derivative=='finite difference':
                eig2=np.real((spla.eigs(transfer_matrix_sparse_no_cycle(rule, compatible_permutations, p,mu+1e-6)[0], k=1, which='LR', maxiter=10*4**(p+c), v0=v0,return_eigenvectors=False)))[0]
            elif method_derivative=='Hellmann-Feynman':
                eigvals, eigvecs = spla.eigs(T.T, k=1, which='LR', v0=v0, maxiter=10*4**(p+c), return_eigenvectors=True)
                vec_left=eigvecs[:,0]
            else:
                raise ValueError("Method must be 'finite difference' or 'Hellmann-Feynman'")
        except ArpackNoConvergence as e:
            print("Warning: ARPACK did not converge. Returning NaN.")
            return np.nan, np.nan

        if method_derivative=='finite difference': 
            phi=np.log(eig)
            rho=(np.log(eig2)-np.log(eig))/1e-6
            return phi-mu*rho, rho
        elif method_derivative=='Hellmann-Feynman':
            norm =vec_left.conj().T@vec_right
            phi_new = np.log(eig_)
            rho_new = (vec_left.conj().T@T_deriv@vec_right/(eig_*norm)).real
            return phi_new-mu*rho_new, rho_new
        else:
            raise ValueError("Method must be 'finite difference' or 'Hellmann-Feynman'")
        
def entropy_and_density_ending_all_0(rule, p=0, c=1, mu=0, rs=42, method_derivative='Hellmann-Feynman'):
    '''
    Compute the entropy and density of the ECA using the transfer matrix method.

    Param: rule (1D array of 8 elements): the rule of the ECA in binary form (inverse of binary used by Wolfram, first 000, then 001, ...)
    Param: p (int): the size of the transient
    Param: c (int): the size of the cycle
    Param: mu (float): the Lagrange multiplier for the density of the initial state
    Param: rs (int): the random seed for the initial vector

    Return: float, float: the entropy and density of the ECA
    '''
    np.random.seed(rs) # set the random seed for reproducibility
    v0=np.random.rand(4**(p+c),1) # initial vector for the sparse matrix
    with open('index_compatible_permutations/indexes_compatible_permutations_'+str(p+c)+'.npy','rb') as f:
        compatible_permutations=np.load(f)
    if c!=0:
        T, T_deriv=transfer_matrix_sparse_ending_in_0(rule, compatible_permutations, p,c,mu)
        try:
            if method_derivative=='finite difference':
                eig=np.real((spla.eigs(T, k=1, which='LR', v0=v0, maxiter=10*4**(p+c), return_eigenvectors=False)))[0]
                if eig<=0:
                    return -np.inf, -np.nan
            elif method_derivative=='Hellmann-Feynman':
                eigvals, eigvecs = spla.eigs(T, k=1, which='LR', v0=v0, maxiter=10*4**(p+c), return_eigenvectors=True)
                eig_=eigvals[0].real
                vec_right=eigvecs[:,0]  
                if eig_<=0:
                    return -np.inf, -np.nan
            else:
                raise ValueError("Method must be 'finite difference' or 'Hellmann-Feynman'")
        except ArpackNoConvergence as e:
            print("Warning: ARPACK did not converge. Returning NaN.")
            return np.nan, np.nan
        
        try:
            if method_derivative=='finite difference':
                eig2=np.real((spla.eigs(transfer_matrix_sparse_ending_in_0(rule, compatible_permutations, p,c,mu+1e-6)[0], k=1, which='LR', maxiter=10*4**(p+c), v0=v0,return_eigenvectors=False)))[0]
            elif method_derivative=='Hellmann-Feynman':
                eigvals, eigvecs = spla.eigs(T.T, k=1, which='LR', v0=v0, maxiter=10*4**(p+c), return_eigenvectors=True)
                vec_left=eigvecs[:,0]
            else:
                raise ValueError("Method must be 'finite difference' or 'Hellmann-Feynman'")
        except ArpackNoConvergence as e:
            print("Warning: ARPACK did not converge. Returning NaN.")
            return np.nan, np.nan

        if method_derivative=='finite difference': 
            phi=np.log(eig)
            rho=(np.log(eig2)-np.log(eig))/1e-6
            return phi-mu*rho, rho
        elif method_derivative=='Hellmann-Feynman':
            norm =vec_left.conj().T@vec_right
            phi_new = np.log(eig_)
            rho_new = (vec_left.conj().T@T_deriv@vec_right/(eig_*norm)).real
            return phi_new-mu*rho_new, rho_new
        else:
            raise ValueError("Method must be 'finite difference' or 'Hellmann-Feynman'")
    else:
        raise ValueError("The size of the cycle must be greater than 0")


def entropy_and_density_translation_left(rule, p=1, c=1, mu=0, rs=42, method_derivative='Hellmann-Feynman'):
    '''
    Compute the entropy and density of the ECA using the transfer matrix method.

    Param: rule (1D array of 8 elements): the rule of the ECA in binary form (inverse of binary used by Wolfram, first 000, then 001, ...)
    Param: p (int): the size of the transient
    Param: c (int): the size of the cycle
    Param: mu (float): the Lagrange multiplier for the density of the initial state
    Param: rs (int): the random seed for the initial vector

    Return: float, float: the entropy and density of the ECA
    '''
    c=c+1 # so that the c is indeed the size of the cycle

    np.random.seed(rs) # set the random seed for reproducibility
    v0=np.random.rand(4**(p+c),1) # initial vector for the sparse matrix

    with open('index_compatible_permutations/indexes_compatible_permutations_'+str(p+c)+'.npy','rb') as f:
        compatible_permutations=np.load(f)

    T, T_deriv=transfer_matrix_sparse_translation_left(rule, compatible_permutations, p,c,mu)
    try:
        if method_derivative=='finite difference':
            eig=np.real((spla.eigs(T, k=1, which='LR', v0=v0, maxiter=10*4**(p+c), return_eigenvectors=False)))[0]
            if eig<=0:
                return -np.inf, -np.nan
        elif method_derivative=='Hellmann-Feynman':
            eigvals, eigvecs = spla.eigs(T, k=1, which='LR', v0=v0, maxiter=10*4**(p+c), return_eigenvectors=True)
            eig_=eigvals[0].real
            vec_right=eigvecs[:,0]  
            if eig_<=0:
                return -np.inf, -np.nan
        else:
            raise ValueError("Method must be 'finite difference' or 'Hellmann-Feynman'")
    except ArpackNoConvergence as e:
        print("Warning: ARPACK did not converge. Returning NaN.")
        return np.nan, np.nan
    
    try:
        if method_derivative=='finite difference':
            eig2=np.real((spla.eigs(transfer_matrix_sparse_translation_left(rule, compatible_permutations, p,c,mu+1e-6)[0], k=1, which='LR', maxiter=10*4**(p+c), v0=v0,return_eigenvectors=False)))[0]
        elif method_derivative=='Hellmann-Feynman':
            eigvals, eigvecs = spla.eigs(T.T, k=1, which='LR', v0=v0, maxiter=10*4**(p+c), return_eigenvectors=True)
            vec_left=eigvecs[:,0]
        else:
            raise ValueError("Method must be 'finite difference' or 'Hellmann-Feynman'")
    except ArpackNoConvergence as e:
        print("Warning: ARPACK did not converge. Returning NaN.")
        return np.nan, np.nan

    if method_derivative=='finite difference': 
        phi=np.log(eig)
        rho=(np.log(eig2)-np.log(eig))/1e-6
        return phi-mu*rho, rho
    elif method_derivative=='Hellmann-Feynman':
        norm =vec_left.conj().T@vec_right
        phi_new = np.log(eig_)
        rho_new = (vec_left.conj().T@T_deriv@vec_right/(eig_*norm)).real
        return phi_new-mu*rho_new, rho_new
    else:
        raise ValueError("Method must be 'finite difference' or 'Hellmann-Feynman'")

def entropy_and_density_translation_right(rule, p=1, c=1, mu=0, rs=42, method_derivative='Hellmann-Feynman'):
    '''
    Compute the entropy and density of the ECA using the transfer matrix method.

    Param: rule (1D array of 8 elements): the rule of the ECA in binary form (inverse of binary used by Wolfram, first 000, then 001, ...)
    Param: p (int): the size of the transient
    Param: c (int): the size of the cycle
    Param: mu (float): the Lagrange multiplier for the density of the initial state
    Param: rs (int): the random seed for the initial vector

    Return: float, float: the entropy and density of the ECA
    '''
    c=c+1 # so that the c is indeed the size of the cycle

    np.random.seed(rs) # set the random seed for reproducibility
    v0=np.random.rand(4**(p+c),1) # initial vector for the sparse matrix

    with open('index_compatible_permutations/indexes_compatible_permutations_'+str(p+c)+'.npy','rb') as f:
        compatible_permutations=np.load(f)

    T, T_deriv=transfer_matrix_sparse_translation_right(rule, compatible_permutations, p,c,mu)
    try:
        if method_derivative=='finite difference':
            eig=np.real((spla.eigs(T, k=1, which='LR', v0=v0, maxiter=10*4**(p+c), return_eigenvectors=False)))[0]
            if eig<=0:
                return -np.inf, -np.nan
        elif method_derivative=='Hellmann-Feynman':
            eigvals, eigvecs = spla.eigs(T, k=1, which='LR', v0=v0, maxiter=10*4**(p+c), return_eigenvectors=True)
            eig_=eigvals[0].real
            vec_right=eigvecs[:,0]  
            if eig_<=0:
                return -np.inf, -np.nan
        else:
            raise ValueError("Method must be 'finite difference' or 'Hellmann-Feynman'")
    except ArpackNoConvergence as e:
        print("Warning: ARPACK did not converge. Returning NaN.")
        return np.nan, np.nan
    
    try:
        if method_derivative=='finite difference':
            eig2=np.real((spla.eigs(transfer_matrix_sparse_translation_right(rule, compatible_permutations, p,c,mu+1e-6)[0], k=1, which='LR', maxiter=10*4**(p+c), v0=v0,return_eigenvectors=False)))[0]
        elif method_derivative=='Hellmann-Feynman':
            eigvals, eigvecs = spla.eigs(T.T, k=1, which='LR', v0=v0, maxiter=10*4**(p+c), return_eigenvectors=True)
            vec_left=eigvecs[:,0]
        else:
            raise ValueError("Method must be 'finite difference' or 'Hellmann-Feynman'")
    except ArpackNoConvergence as e:
        print("Warning: ARPACK did not converge. Returning NaN.")
        return np.nan, np.nan

    if method_derivative=='finite difference': 
        phi=np.log(eig)
        rho=(np.log(eig2)-np.log(eig))/1e-6
        return phi-mu*rho, rho
    elif method_derivative=='Hellmann-Feynman':
        norm =vec_left.conj().T@vec_right
        phi_new = np.log(eig_)
        rho_new = (vec_left.conj().T@T_deriv@vec_right/(eig_*norm)).real
        return phi_new-mu*rho_new, rho_new
    else:
        raise ValueError("Method must be 'finite difference' or 'Hellmann-Feynman'")

def entropy_and_density_translation_right_2(rule, p=1, c=1, mu=0, rs=42, method_derivative='Hellmann-Feynman'):
    '''
    Compute the entropy and density of the ECA using the transfer matrix method.

    Param: rule (1D array of 8 elements): the rule of the ECA in binary form (inverse of binary used by Wolfram, first 000, then 001, ...)
    Param: p (int): the size of the transient
    Param: c (int): the size of the cycle
    Param: mu (float): the Lagrange multiplier for the density of the initial state
    Param: rs (int): the random seed for the initial vector

    Return: float, float: the entropy and density of the ECA
    '''
    c=c+1 # so that the c is indeed the size of the cycle

    np.random.seed(rs) # set the random seed for reproducibility
    v0=np.random.rand(4**(p+c),1) # initial vector for the sparse matrix
    with open('index_compatible_permutations/indexes_compatible_permutations_'+str(p+c)+'.npy','rb') as f:
        compatible_permutations=np.load(f)

    T, T_deriv=transfer_matrix_sparse_translation_right_2(rule, compatible_permutations, p,c,mu)
    try:
        if method_derivative=='finite difference':
            eig=np.real((spla.eigs(T, k=1, which='LR', v0=v0, maxiter=10*4**(p+c), return_eigenvectors=False)))[0]
            if eig<=0:
                return -np.inf, -np.nan
        elif method_derivative=='Hellmann-Feynman':
            eigvals, eigvecs = spla.eigs(T, k=1, which='LR', v0=v0, maxiter=10*4**(p+c), return_eigenvectors=True)
            eig_=eigvals[0].real
            vec_right=eigvecs[:,0]  
            if eig_<=0:
                return -np.inf, -np.nan
        else:
            raise ValueError("Method must be 'finite difference' or 'Hellmann-Feynman'")
    except ArpackNoConvergence as e:
        print("Warning: ARPACK did not converge. Returning NaN.")
        return np.nan, np.nan
    
    try:
        if method_derivative=='finite difference':
            eig2=np.real((spla.eigs(transfer_matrix_sparse_translation_right_2(rule, compatible_permutations, p,c,mu+1e-6)[0], k=1, which='LR', maxiter=10*4**(p+c), v0=v0,return_eigenvectors=False)))[0]
        elif method_derivative=='Hellmann-Feynman':
            eigvals, eigvecs = spla.eigs(T.T, k=1, which='LR', v0=v0, maxiter=10*4**(p+c), return_eigenvectors=True)
            vec_left=eigvecs[:,0]
        else:
            raise ValueError("Method must be 'finite difference' or 'Hellmann-Feynman'")
    except ArpackNoConvergence as e:
        print("Warning: ARPACK did not converge. Returning NaN.")
        return np.nan, np.nan

    if method_derivative=='finite difference': 
        phi=np.log(eig)
        rho=(np.log(eig2)-np.log(eig))/1e-6
        return phi-mu*rho, rho
    elif method_derivative=='Hellmann-Feynman':
        norm =vec_left.conj().T@vec_right
        phi_new = np.log(eig_)
        rho_new = (vec_left.conj().T@T_deriv@vec_right/(eig_*norm)).real
        return phi_new-mu*rho_new, rho_new
    else:
        raise ValueError("Method must be 'finite difference' or 'Hellmann-Feynman'")

def entropy_and_density_translation_left_2(rule, p=1, c=1, mu=0, rs=42, method_derivative='Hellmann-Feynman'):
    '''
    Compute the entropy and density of the ECA using the transfer matrix method.

    Param: rule (1D array of 8 elements): the rule of the ECA in binary form (inverse of binary used by Wolfram, first 000, then 001, ...)
    Param: p (int): the size of the transient
    Param: c (int): the size of the cycle
    Param: mu (float): the Lagrange multiplier for the density of the initial state
    Param: rs (int): the random seed for the initial vector

    Return: float, float: the entropy and density of the ECA
    '''
    c=c+1 # so that the c is indeed the size of the cycle

    np.random.seed(rs) # set the random seed for reproducibility
    v0=np.random.rand(4**(p+c),1) # initial vector for the sparse matrix

    with open('index_compatible_permutations/indexes_compatible_permutations_'+str(p+c)+'.npy','rb') as f:
        compatible_permutations=np.load(f)

    T, T_deriv=transfer_matrix_sparse_translation_left_2(rule, compatible_permutations, p,c,mu)
    try:
        if method_derivative=='finite difference':
            eig=np.real((spla.eigs(T, k=1, which='LR', v0=v0, maxiter=10*4**(p+c), return_eigenvectors=False)))[0]
            if eig<=0:
                return -np.inf, -np.nan
        elif method_derivative=='Hellmann-Feynman':
            eigvals, eigvecs = spla.eigs(T, k=1, which='LR', v0=v0, maxiter=10*4**(p+c), return_eigenvectors=True)
            eig_=eigvals[0].real
            vec_right=eigvecs[:,0]  
            if eig_<=0:
                return -np.inf, -np.nan
        else:
            raise ValueError("Method must be 'finite difference' or 'Hellmann-Feynman'")
    except ArpackNoConvergence as e:
        print("Warning: ARPACK did not converge. Returning NaN.")
        return np.nan, np.nan
    
    try:
        if method_derivative=='finite difference':
            eig2=np.real((spla.eigs(transfer_matrix_sparse_translation_left_2(rule, compatible_permutations, p,c,mu+1e-6)[0], k=1, which='LR', maxiter=10*4**(p+c), v0=v0,return_eigenvectors=False)))[0]
        elif method_derivative=='Hellmann-Feynman':
            eigvals, eigvecs = spla.eigs(T.T, k=1, which='LR', v0=v0, maxiter=10*4**(p+c), return_eigenvectors=True)
            vec_left=eigvecs[:,0]
        else:
            raise ValueError("Method must be 'finite difference' or 'Hellmann-Feynman'")
    except ArpackNoConvergence as e:
        print("Warning: ARPACK did not converge. Returning NaN.")
        return np.nan, np.nan

    if method_derivative=='finite difference': 
        phi=np.log(eig)
        rho=(np.log(eig2)-np.log(eig))/1e-6
        return phi-mu*rho, rho
    elif method_derivative=='Hellmann-Feynman':
        norm =vec_left.conj().T@vec_right
        phi_new = np.log(eig_)
        rho_new = (vec_left.conj().T@T_deriv@vec_right/(eig_*norm)).real
        return phi_new-mu*rho_new, rho_new
    else:
        raise ValueError("Method must be 'finite difference' or 'Hellmann-Feynman'")

