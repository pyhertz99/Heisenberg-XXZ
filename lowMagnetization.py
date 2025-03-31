import numpy as np
from scipy import sparse
from scipy.special import comb
from itertools import combinations
import heisenbergXXZ as xxz


def findIndex(vec, spin_basis):
    """
    Search for index of binary represented vector
    in binary basis array.

    Parameters
    ----------
    vec : array of 0/1 (N)
        Binary represented vector to find.
    spin_basis : array of 0/1 (M,N)
        Binary represented basis.

    Raises
    ------
    ValueError
        If such vector not found in basis.

    Returns
    -------
    index : int
        Index of given vector.

    """
    
    M = spin_basis.shape[0]
    
    for index in range(M):
        basis_vec = np.copy(spin_basis[index])
        if np.array_equal(vec, basis_vec):
            return index
    
    print("vector not found in basis")
    raise ValueError
    
    
def flipBasisVecIndex(i,j,basis_vec,spin_basis):
    """
    Search for index in spin_basis
    of basis vector with flipped i-th and j-th
    binary position.
    
    Parameters
    ----------
    i : int
        First binary position index.
    j : int
        Second binary position index.
    basis_vec : array of 0/1 (N)
        Binary represented basis vector.
    spin_basis : array of 0/1 (M,N)
        Binary represented basis.

    Returns
    -------
    l : int
        Index of flipped vector in binary basis.

    """
    
    flip_basis_vec = np.copy(basis_vec)
    
    if flip_basis_vec[i] == 0:
        flip_basis_vec[i] = 1
        flip_basis_vec[j] = 0
    else:    
        flip_basis_vec[i] = 0
        flip_basis_vec[j] = 1
    
    l = findIndex(flip_basis_vec, spin_basis)
    
    return l
    

def basisVectorImage(i,j,k,J_xy,J_z,spin_basis):
    """
    Computes image of basis vector (represented in 
    binary basis) with respect to pauli matrices
    applied at i-th and j-th site.

    Parameters
    ----------
    i : int
        Index of first site.
    j : int
        Index of second site.
    k : int
        Index of basis vector in binary basis.
    J_xy : float
        J_xy sites coupling constant.
    J_z : float
        J_z sites coupling constant.
    basis_vec : array of 0/1 (N)
        Binary represented basis vector.
    spin_basis : array of 0/1 (M,N)
        Binary represented basis.

    Raises
    ------
    ValueError
        If binary vector is in wrong format.

    Returns
    -------
    y : array (M)
        Image in binary basis.

    """
    
    basis_vec = np.copy(spin_basis[k])
    M = spin_basis.shape[0]
    w = (basis_vec[i],basis_vec[j])
    
    y = np.zeros((M),dtype=complex)
    if w == (0,0) or w == (1,1):
        y[k] = J_z
        
    elif w == (1,0):
        l = flipBasisVecIndex(i, j, basis_vec, spin_basis)
        y[k] = -J_z
        y[l] = 2*J_xy
        
    elif w == (0,1):
        l = flipBasisVecIndex(i, j, basis_vec, spin_basis)
        y[k] = -J_z
        y[l] = 2*J_xy
        
    else:
        raise ValueError
        
    return y 
    

def createHamiltonian(N,L,M_xy,M_z,B_array):
    """
    Creates L-subspace hamiltonian for given
    connectivity matrices and magnetic field array.
    Designated for low magnetization states.

    Parameters
    ----------
    N : integer
        Number of sites.
    L : integer
        Number of excited states.
    M_xy : array (N,N) of float 
        J_xy symmetric tridiagonal connectivity matrix.
    M_z : array (N,N) of float
        J_z symmetric tridiagonal connectivity matrix.
    B_array : array (N) of float
        External magnetic field.

    Returns
    -------
    H : array of real (M,M)
        Subspace hamiltonian.

    """
    
    spin_basis = xxz.subspaceBasisBinary(N,L)
    M = spin_basis.shape[0]
    
    H = np.zeros((M,M),dtype=complex)
    
    #interaction
    for i in range(N):
        for j in range(N):
            H_ij = np.zeros((M,M),dtype=complex)

            J_xy = M_xy[i,j]
            J_z = M_z[i,j]
            if J_xy == 0.0 and J_z == 0.0:
                continue
            
            for k in range(M):
                H_ij[:,k] = basisVectorImage(i, j, k, J_xy, J_z, spin_basis)
                
            H += H_ij
            
    #B-field
    for i in range(N):
        H_i = np.zeros((M,M),dtype=complex)
        B = B_array[i]
        
        for k in range(M):
            basis_vec = np.copy(spin_basis[k])
            if basis_vec[i] == 0:
                H_i[k,k] = B
            else:
                H_i[k,k] = -B
        
        H += H_i 
        
    return np.real(H)


def localMagnetizationOperators(N,M,spin_basis):
    """
    Creates array of operators on L-magnetization
    subspace each corresponding to local magnetization
    on i-th site.

    Parameters
    ----------
    N : integer
        Number of sites.
    M : integer
        L-magnetization subspace dimension.
    spin_basis : array of 0/1 (M,N)
        Binary represented basis.

    Returns
    -------
    As : array of complex (N,M,M)
        Array of subspace operators for each site.

    """
    
    As = np.zeros((N,M,M),dtype=complex)
    
    ys = np.full(N,0.0)
    
    for i in range(N):
        A_i = np.zeros((M,M),dtype=complex)
        
        for k in range(M):
            basis_vec = np.copy(spin_basis[k])
            if basis_vec[i] == 0:
                A_i[k,k] = 1
            else:
                A_i[k,k] = -1
                
        As[i] = A_i
        
    return As


def contourImg(psi_array,N,M,spin_basis,t_steps):
    """
    Creates a contour plot for given array of state
    vectors.

    Parameters
    ----------
    psi_array : array (t_steps,M)
        Array of state vectors.
    N : integer
        Number of sites.
    M : integer
        L-magnetization subspace dimension.
    spin_basis : array of 0/1 (M,N)
        Binary represented basis.
    t_steps : int
        Number of time steps.
    

    Returns
    -------
    array of real (t_steps,N)
        Contour plot.

    """
    
    img = np.full((t_steps, N), 0.0)

    As = localMagnetizationOperators(N,M,spin_basis)
    
    for i in range(N):
        A_i = As[i]
        
        for j in range(t_steps):   
            psi = psi_array[j]     
            img[j, i] = np.real(psi.conjugate() @ A_i @ psi)
               
    return np.flip(img)


def partialTraceBasis(sites,index1,index2,spin_basis,M):
    """
    Computes partial trace of tensor product
    of basis vectors.

    Parameters
    ----------
    sites : numpy array of int (not work for lists)
        Sites which remain. Runs from 1 to N !!!
    index1 : int
        Index of first basis vector in spin_basis.
    index2 : int
        Index of second basis vector in spin_basis.
    spin_basis : array of 0/1 (M,N)
        Binary represented basis.
    M : integer
        L-magnetization subspace dimension.

    Returns
    -------
    rho : array of complex (2**n,2**n) n = #sites
        Traced density matrix.

    """

    rho = np.array([1],dtype='complex')
    
    e1 = np.copy(spin_basis[index1])
    e2 = np.copy(spin_basis[index2])

    for site in sites:
        site_index = site-1
        b1 = e1[site_index]
        b2 = e2[site_index]
        
        if b1 == 0:
            ket = np.array([1,0],dtype="complex")
        else:
            ket = np.array([0,1],dtype="complex")
        
        if b2 == 0:
            bra = np.array([1,0],dtype="complex")
        else:
            bra = np.array([0,1],dtype="complex")
            
        rho_site = np.outer(ket,bra)
        rho = np.kron(rho,rho_site)
        
    return rho


def partialTraceBinaryMatrix(sites,spin_basis,M):
    """
    Creates matrix with entry 1 iff tensor product
    of basis vectors i and j are involved in
    traced density matrix.

    Parameters
    ----------
    sites : numpy array of int (not work for lists)
        Sites which remain. Runs from 1 to N !!!
    spin_basis : array of 0/1 (M,N)
        # Binary represented basis.
    M : integer
        L-magnetization subspace dimension.

    Returns
    -------
    A : array of 0/1 (M,M)
        Matrix for basis vectors.

    """
    
    n = sites.size
    
    A = np.zeros((M,M,2**n,2**n),dtype=complex)
    
    for i in range(M):
        for j in range(M):
            e1 = np.copy(spin_basis[i])
            e2 = np.copy(spin_basis[j])
            e1 = np.delete(e1,sites-1)
            e2 = np.delete(e2,sites-1)
            if np.array_equal(e1,e2):
                A[i,j] = partialTraceBasis(sites,i,j,spin_basis,M)
    return A


def partialTrace(psi,sites,spin_basis,binary_array,M):
    """
    Computes partial trace over sites not included
    in sites array.

    Parameters
    ----------
    psi : array of complex (M)
        State vector.
    sites : numpy array of int (not work for lists)
        Sites which remain. Runs from 1 to N !!!
    spin_basis : array of 0/1 (M,N)
        Binary represented basis.
    binary_array : array of 0/1 (M,M)
        Array created with partialTraceBinaryMatrix.
    M : integer
        L-magnetization subspace dimension.

    Returns
    -------
    rho : array of complex (2**n,2**n) n = #sites
        Traced density matrix.

    """
    
    n = len(sites)
    rho = np.zeros((2**n,2**n),dtype=complex)
    
    for i in range(M):
        for j in range(M):
            rho_traced = binary_array[i,j]
            rho += psi[i]*np.conjugate(psi[j])*rho_traced
    
    return rho


def entanglementEntropyArray(psi_array,site,spin_basis,M,t_steps):
    """
    Computes entanglement entropy on
    given site at each time step. Designated
    for low magnetization states.

    Parameters
    ----------
    psi_array : array (t_steps,M)
        Array of state vectors in spin basis
        at each time step.
    site : int
        Site to compute entanglement.
    spin_basis : array of 0/1 (M,N)
        Binary represented basis.
    M : integer
        L-magnetization subspace dimension.
    t_steps : int
        Number of time steps.

    Returns
    -------
    entropy_array : array of float (t_steps)
        Entanglement entropy at each time step.

    """
    
    entropy_array = np.full(t_steps,0.0)
    
    binary_array = partialTraceBinaryMatrix(np.array([site]),spin_basis,M)
    
    for i in range(t_steps):
        psi = psi_array[i]
        rho_tr = partialTrace(psi, [site], spin_basis, binary_array, M)
        rho_tr = sparse.coo_array(rho_tr)
        entropy_array[i] = xxz.entanglementEntropy(rho_tr, 2)
        
    return entropy_array


def mutualInformationArray(psi_array,sites1,sites2,spin_basis,M,t_steps):
    """
    Computes mutual information on
    given site at each time step. Designated
    for low magnetization states.

    Parameters
    ----------
    psi_array : array (t_steps,M)
        Array of state vectors in spin basis
        at each time step.
    spin_basis : array of 0/1 (M,N)
        Binary represented basis.
    M : integer
        L-magnetization subspace dimension.
    t_steps : int
        Number of time steps.
    sites1 : array of int
        first set if sites.
    sites2 : array of int
        second set of sites.

    

    Returns
    -------
    entropy_array : array of float (t_steps)
        Entropy array at each time step.
    mutual_array : array of float (t_steps)
        Mutual information array at each time step.

    """
    
    sites_all = np.union1d(sites1,sites2)
    
    n1 = sites1.size
    n2 = sites2.size
    n3 = sites_all.size
    
    mutual_array = np.full(t_steps,0.0)
    entropy_array = np.full(t_steps,0.0)
    
    binary_array1 = partialTraceBinaryMatrix(sites1,spin_basis,M)
    binary_array2 = partialTraceBinaryMatrix(sites2,spin_basis,M)
    binary_array12 = partialTraceBinaryMatrix(sites_all,spin_basis,M)
    
    for i in range(t_steps):
        psi = psi_array[i]
        
        rho1 = partialTrace(psi, sites1, spin_basis, binary_array1, M)
        rho2 = partialTrace(psi, sites2, spin_basis, binary_array2, M)
        rho12 = partialTrace(psi, sites_all, spin_basis, binary_array12, M)
        
        rho1 = sparse.coo_array(rho1)
        rho2 = sparse.coo_array(rho2)
        rho12 = sparse.coo_array(rho12)
        S1 = xxz.entanglementEntropy(rho1, 2**n1)
        S2 = xxz.entanglementEntropy(rho2, 2**n2)
        S12 = xxz.entanglementEntropy(rho12, 2**n3)
        
        entropy_array[i] = S2
        mutual_array[i] = S1 + S2 - S12
        
    return entropy_array, mutual_array
        