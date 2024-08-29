import numpy as np
from scipy import sparse
from scipy.special import comb
from itertools import combinations

#identity matrix
I = sparse.csc_matrix(np.array([[1,0],
                                [0,1]]), dtype=complex)

#pauli spin matrices
sigma_x = sparse.csc_matrix(np.array([[0,1],
                                      [1,0]]), dtype=complex)

sigma_y = sparse.csc_matrix(np.array([[0,-1j],
                                      [1j,0]]), dtype=complex)

sigma_z = sparse.csc_matrix(np.array([[1, 0],
                                      [0,-1]]), dtype=complex)


def createConnectivityMatrices(N, J_xy, J_z, boundary=0.0):
    """
    Creates J_xy and J_z connectivity matrices
    for 1D chain of sites with constant coupling
    constants J_xy and J_z.
    
    Parameters
    ----------
    N : integer
        Number of sites.
    J_xy : float
        J_xy sites coupling constant.
    J_z : float
        J_z sites coupling constant.
    boundary: float, optional
        Connection at the edges of the chain; 0 for open boundary conditions, 1 for periodic boundary conditions.

    Returns
    -------
    M_xy : array (N,N) of float 
        J_xy symmetric tridiagonal connectivity matrix.
    M_z : array (N,N) of float
        J_z symmetric tridiagonal connectivity matrix.

    """
    
    M_xy = np.full((N,N),0.0)
    M_z = np.full((N,N),0.0)
    
    for i in range(N-1):
        M_xy[i,i+1] = J_xy/2
        M_z[i,i+1] = J_z/2
    for i in range(1,N):
        M_xy[i,i-1] = J_xy/2
        M_z[i,i-1] = J_z/2
        
    M_xy[0,N-1] = boundary * J_xy/2
    M_xy[N-1,0] = boundary * J_xy/2
    M_z[0,N-1] = boundary * J_z/2
    M_z[N-1,0] = boundary * J_z/2

    return M_xy, M_z
    
    
def createOperator(N, op_array, indices):
    """
    Creates operator on complete Hilbert space
    from single-spin operators.

    Parameters
    ----------
    N : integer
        Number of cells.
    op_array : array of 2x2 matrices in CSC form
        Single spin operators.
    indices : array of integers
        Sorted indices of cells of the operators.

    Returns
    -------
    A : sparse matrix (2^N,2^N) in CSC form
        Operator on complete Hilbert space.

    """
    
    A = sparse.csc_matrix(np.array(1))
    j = 0
    j_max = len(indices)
    for index in range(N):
        if j == j_max or index != indices[j]:
            A = sparse.kron(A,I,format="csc")
        else:
            A = sparse.kron(A,op_array[j],format="csc")
            j += 1
            
    return A

def createCompleteHamiltonian(N, M_xy, M_z, B_array):
    """
    Creates Hamiltonian on complete hilbert space
    from connectivity matrices and magnetic field array.

    Parameters
    ----------
    N : integer
        Number of sites.
    M_xy : array (N,N) of float 
        J_xy symmetric tridiagonal connectivity matrix.
    M_z : array (N,N) of float
        J_z symmetric tridiagonal connectivity matrix.
    B_array : array (N) of float
        External magnetic field.

    Returns
    -------
    H : sparse matrix (2^N,2^N) in CSC form
        Hamiltonian on complete Hilbert space.

    """
    
    H = sparse.csc_array((2**N,2**N),dtype=complex)
    
    #external magnetic field
    for j in range(N):
        H += B_array[j] * createOperator(N,[sigma_z],[j])
        
    #sites coupling
    for i in range(N):
        for j in range(N):
            if i < j:
                J_xy = 2*M_xy[i,j]
                J_z = 2*M_z[i,j]
                H += J_xy * createOperator(N,[sigma_x,sigma_x], [i,j])
                H += J_xy * createOperator(N,[sigma_y,sigma_y], [i,j])
                H += J_z * createOperator(N,[sigma_z,sigma_z], [i,j])
                
    return H

def subspaceTransformationMatrix(N,L,M):
    """
    Creates transformation matrix from complete hilbert
    space to L-subspace.

    Parameters
    ----------
    N : integer
        Number of sites.
    L : integer
        Number of excited states.

    Returns
    -------
    PI: sparse array (D,M)
        Transformation matrix from complete space
        to subspace.
    spin_indices: array (M)
        indices of corresponding spin states in
        complete hilbert space

    """
    
    D = 2**N
    
    #array of binary represented basis vectors
    spin_basis = np.full((M,N),0)
    #corresponding complete vector indices
    spin_indices = np.full(M,0)

    i = 0
    for c in combinations(range(N),L):
        for j in c:
            spin_basis[i,j] = 1
        i += 1

    vectorIndex = lambda spin_state : int("".join(str(x) for x in spin_state), 2)

    for i in range(M):
        spin_indices[i] = vectorIndex(spin_basis[i])

    #construct transformation matrix
    rows = np.arange(M)
    cols = spin_indices
    data = np.full(M,1)

    PI = sparse.coo_array((data,(rows,cols)),shape=(M,D))
    PI = PI.tocsc()

    return PI, spin_indices


def createHamiltonian(N, PI, M_xy, M_z, B_array):
    """
    Create L-subspace hamiltonian.

    Parameters
    ----------
    N : integer
        Number of sites.
    PI : array (D,M)
        Transformation matrix from complete space
        to subspace.
    M_xy : array (N,N) of float 
        J_xy symmetric tridiagonal connectivity matrix.
    M_z : array (N,N) of float
        J_z symmetric tridiagonal connectivity matrix.
    B_array : array (N) of float
        External magnetic field.

    Returns
    -------
    H_L : complex array (M,M)
        L-subspace hamiltonian.

    """
    
    H = createCompleteHamiltonian(N, M_xy, M_z, B_array)
    H_L = PI @ H @ PI.transpose()
    H_L = H_L.toarray()
    
    return np.real(H_L)

def diagonalizeHamiltonian(H):
    """
    Computes spectrum and eigenvectors of hamiltonian.

    Parameters
    ----------
    H : complex array (M,M)
        L-subspace hamiltonian.

    Returns
    -------
    array (M) of float, array (M,M) of complex 
        Spectrum and eigenvectors of hamiltonian.

    """
    
    return np.linalg.eigh(H)


def evolveState(t_max,t_steps,spectrum,eigvecs,eigvecs_herm,psi_0,M):
    """
    Evolves state in time.

    Parameters
    ----------
    t_max : float
        Maximum time of evolution.
    t_steps : int
        Number of time steps.
    spectrum : array (M)
        Spectrum of hamiltonian.
    eigvecs : array (M,M)
        Corresponding eigenvectors.
    psi_0 : array (M)
        Initial state vector in spin basis.
    M : int
        Size of corresponding hilbert space.

    Returns
    -------
    psi_array : array (t_steps,M)
        Array of state vectors in spin basis
        at each time step.

    """

    c_0 = eigvecs_herm @ psi_0
    
    ts = np.linspace(0,t_max,t_steps)
    psi_array = np.full((t_steps,M),0j) # array of spin basis states in time

    c_t = np.full(M,0j)

    for j in range(t_steps):
        t = ts[j]
        c_t = c_0 * np.exp(-1j*spectrum*t)
        psi_array[j] = eigvecs @ c_t
        
    return psi_array

def stateOverlap(psi_A,psi_B):
    """
    Computes overlap of state vectors.

    Parameters
    ----------
    psi_A : array (M)
        First state vector.
    psi_B : array (M)
        Second state vector.

    Returns
    -------
    float
        Overlap of states |<a|b>|^2.

    """
    k = np.dot(psi_A.conjugate(),psi_B)
    return np.absolute(k)**2
    
def survivalProbability(psi_array, t_steps):
    """
    psi_array : array (t_steps,M)
        Array of state vectors in spin basis for each time step.
    t_steps : int
        Number of time steps to compute.

    Returns: array (t_steps)
        Array of survival probability for each time step.

    """
    
    survival_array = np.full(t_steps,0.0)
    psi_0 = psi_array[0]
    
    for i in range(t_steps):
        survival_array[i] = stateOverlap(psi_array[i], psi_0)
    
    return survival_array
    
def localMagnetization(psi,PI,N):
    """
    Computes local magnetization on all sites.

    Parameters
    ----------
    cs : array (M)
        State vector in stationary basis.
    eigvecs : array (M,M)
        Corresponding eigenvectors.
    PI: sparse array (D,M)
        Transformation matrix from complete space
        to subspace.
    N : integer
        Number of sites.

    Returns
    -------
    ys : TYPE
        DESCRIPTION.

    """
    
    ys = np.full(N,0.0)
    
    for i in range(N):
        A = createOperator(N, [sigma_z], [i])
        A_L = PI @ A @ PI.transpose()
        A_L = A_L.toarray()
        ys[i] = np.real(psi.conjugate() @ A_L @ psi)
        
    return ys

def contourImg(psi_array,t_steps,PI,N):
    """
    Creates contour plot.

    Parameters
    ----------
    c_array : array (t_steps,M)
        Array of state vectors in stationary basis for each time step.
    t_steps : int
        Number of time steps to compute.
    eigvecs : array (M,M)
        Corresponding eigenvectors.
    PI: sparse array (D,M)
        Transformation matrix from complete space
        to subspace.
    N : integer
        Number of sites.

    Returns
    -------
    array (t_steps,N)
        Contour plot.

    """
    
    img = np.full((t_steps, N), 0.0)

    for i in range(N):
        A = createOperator(N, [sigma_z], [i])
        A_L = PI @ A @ PI.transpose()
        
        for j in range(t_steps):   
            psi = psi_array[j]     
            img[j, i] = np.real(psi.conjugate() @ A_L @ psi)
               
    return np.flip(img)

def domainWall(M):
    """
    Creates domain wall state.

    Parameters
    ----------
    N : integer
        Number of sites.

    Returns
    -------
    psi_0 : array (M)
        Domain wall state.

    """
    
    psi_0 = np.full(M,0.0)
    psi_0[-1] = 1
    
    return psi_0

def neel(N,M,spin_indices):
    """
    Creates Neel state.

    Parameters
    ----------
    N : integer
        Number of sites.
    M : int
        Size of corresponding hilbert space.
    spin_indices : TYPE
        DESCRIPTION.

    Returns
    -------
    psi_0 : array (M)
        Neel state.

    """
    
    psi_0 = np.full(M,0.0)
    
    neel_state = np.full(N,1)
    for i in range(1,N,2):
        neel_state[i] = 0
    
    neel_index = int("".join(str(x) for x in neel_state), 2)
    
    for idx in range(M):
        if spin_indices[idx] == neel_index:
            break
    
    psi_0[idx] = 1
    
    return psi_0
        
def singleSpin(N,site):
    """
    Creates single spin state.

    Parameters
    ----------
    N : integer
        Number of sites.
    site : integer
        Site to excite (1 to N).

    Returns
    -------
    psi_0 : array (M)
        Single spin state.

    """
    
    psi_0 = np.full(N,0.0)
    
    psi_0[site-1] = 1
    
    return psi_0

def rightPartialTrace(rho,dim_A,dim_B):
    """
    Performs partial trace over system B.

    Parameters
    ----------
    rho : sparse array (D,D) in CSC form
        Density matrix.
    dim_A : int
        Dimension of system A.
    dim_B : int
        Dimension of system B.

    Returns
    -------
    rho_A : sparse array (dim_A,dim_A) in CSC form
        Traced density matrix.

    """
    
    rho_A = sparse.csc_matrix((dim_A, dim_A), dtype=complex)
    
    for b in range(dim_B):
        indices = np.arange(b, dim_A * dim_B, dim_B)
        rho_A_block = rho[indices, :]
        rho_A_block = rho_A_block[:, indices]
        rho_A += rho_A_block

    return rho_A

def leftPartialTrace(rho,dim_A,dim_B):
    """
    Performs partial trace over system A.

    Parameters
    ----------
    rho : sparse array (D,D) in CSC form
        Density matrix.
    dim_A : int
        Dimension of system A.
    dim_B : int
        Dimension of system B.

    Returns
    -------
    rho_A : sparse array (dim_B,dim_B) in CSC form
        Traced density matrix.

    """
    
    rho_B = sparse.csc_matrix((dim_B, dim_B), dtype=complex)
    
    for a in range(dim_A):
        indices = np.arange(a * dim_B, (a + 1) * dim_B)
        rho_B_block = rho[indices, :][:, indices]
        rho_B += rho_B_block
    
    return rho_B

def siteDensityMatrix(psi_L, PI, site, N):
    """
    Computes reduced density matrix for chosen site.

    Parameters
    ----------
    psi_L : array (M) of complex
        State vector in L-subspace.
    PI: sparse array (D,M)
        Transformation matrix from complete space
        to subspace.
    site : int
        Site to calculate density matrix for.
    N : int
        Number of sites.

    Returns
    -------
    array (2,2)
        Reduced density matrix.

    """
    
    psi_L = sparse.csc_matrix(psi_L)
    
    psi = PI.T @ psi_L.T
    rho = psi.conjugate() * psi.T
    
    dim_A = 2**site
    dim_B = 2**(N-site)
    rho_tr = rightPartialTrace(rho, dim_A, dim_B)
    rho_tr = leftPartialTrace(rho_tr, dim_A//2, 2)
    
    return rho_tr.toarray()
    
def xlogx(x):
    """
    Computes x * log(x) and checks for small values.
    """
    
    if x < 10**(-100):
        return 0
    else:
        return x * np.log(x)
    
        
def entanglementEntropy(rho,dim):
    """
    Computes entanglement entropy for given
    density matrix.

    Parameters
    ----------
    rho : array (2,2)
        Density matrix.
    dim : int
        Dimension of density matrix.

    Returns
    -------
    s : float
        Entanglement entropy.

    """
    
    rho_diag = np.linalg.eigvalsh(rho)
    
    s = 0
    for i in range(dim):
        lamb = rho_diag[i]
        s -= xlogx(lamb)
    
    return s

def entanglementEntropyArray(psi_array, t_steps, PI, site, N):
    """
    Computes entanglement entropy on
    given site at each time step.

    Parameters
    ----------
    psi_array : array (t_steps,M)
        Array of state vectors in spin basis
        at each time step.
    t_steps : int
        Number of time steps.
    PI: sparse array (D,M)
        Transformation matrix from complete space
        to subspace.
    site : int
        Site to compute entanglement.
    N : int
        Number of sites.

    Returns
    -------
    entropy_array : array (t_steps)
        Entanglement entropy at each time step.

    """
    
    entropy_array = np.full(t_steps,0.0)
    
    for i in range(t_steps):
        psi = psi_array[i]
        rho_tr = siteDensityMatrix(psi, PI, site, N)
        entropy_array[i] = entanglementEntropy(rho_tr, 2)
        
    return entropy_array

def domainWallEntropyArray(psi_array,t_steps,PI,N):
    """
    

    Parameters
    ----------
    psi_array : array (t_steps,M)
        Array of state vectors in spin basis
        at each time step.
    t_steps : int
        Number of time steps.
    PI: sparse array (D,M)
        Transformation matrix from complete space
        to subspace.
    N : int
        Number of sites.

    Returns
    -------
    entropy_array : array (t_steps)
        Array with half-spins entropy at each time step.

    """
    
    entropy_array = np.full(t_steps,0.0)
    
    for i in range(t_steps):
        psi_L = sparse.csc_matrix(psi_array[i])
        
        psi = PI.T @ psi_L.T
        rho = psi.conjugate() * psi.T
        
        dim_half = 2**(N//2)
        rho_tr = rightPartialTrace(rho, dim_half, dim_half)
        
        rho_tr = rho_tr.toarray()
        
        entropy_array[i] = entanglementEntropy(rho_tr, 2**(N//2))
        
    return entropy_array