import numpy as np
from scipy import sparse
from scipy.special import comb
from itertools import combinations

# identity matrix
I = sparse.csc_matrix(np.array([[1,0],
                                [0,1]]), dtype=complex)

# pauli spin matrices
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
    J_z : TYPE
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


def subspaceBasisBinary(N,L):
    """
    Creates array of binary representation
    of subspace basis vectors.
    
    Parameters
    ----------
    N : integer
        Number of sites.
    L : integer
        Number of excited states.

    Returns
    -------
    spin_basis : array (M,N)
        Binary representation of subspace basis.

    """
    
    M = int(comb(N,L)) #subspace dimension
    
    i = 0
    spin_basis = np.full((M,N),0)
    for c in combinations(range(N),L):
        for j in c:
            spin_basis[i,j] = 1
        i += 1
    
    return spin_basis


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
    spin_basis = subspaceBasisBinary(N, L)
    #corresponding complete vector indices
    spin_indices = np.full(M,0)

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
    
    if t_steps == 1:
        ts = np.array([t_max])
    else:
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
    psi_array : array (t_steps,M)
        Array of state vectors.
    t_steps : int
        Number of time steps to compute.
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

def randomState(M):
    """
    Cretes random state of dimension M.

    Parameters
    ----------
    M : int
        Hilbert space dimension.

    Returns
    -------
    psi_0 : array of complex (M)
        Normalized state vector.

    """
    
    psi_0 = np.zeros(M,dtype="complex")
    
    re = np.random.uniform(low=-1.0,high=1.0,size=M)
    im = np.random.uniform(low=-1.0,high=1.0,size=M)
    
    for i in range(M):
        psi_0[i] = re[i] + 1j*im[i]
        
    dot = np.conjugate(psi_0) @ psi_0
    psi_0 *= 1 / np.sqrt(np.real(dot))
    
    return psi_0


def XZRotationState(k,N,PI):
    """
    Creates a state with k-rotated spin.

    Parameters
    ----------
    k : int
        Number of rotations.
    N : integer
        Number of sites.
    PI: sparse array (D,M)
        Transformation matrix from complete space
        to subspace.

    Returns
    -------
    psi_0 : array (M)
        k-rotated state.

    """
    
    theta = (k/N)*2*np.pi
    R = np.array([[np.cos(theta/2),-np.sin(theta/2)],
                  [np.sin(theta/2),np.cos(theta/2)]])
    
    spin_state = np.array([1.0,0.0])
    complete_state = 1
    for i in range(N):
        complete_state = np.kron(complete_state,spin_state)
        spin_state = R @ spin_state
    
    return PI @ complete_state.T


#code for entanglement and information

def densityMatrix(psi_L,PI):
    """
    Computes complete-space density matrix of
    given subspace vector in CSC form.

    Parameters
    ----------
    psi_L : array of complex (M)
        Subspace state vector.
    PI: sparse array (D,M)
        Transformation matrix from complete space
        to subspace.

    Returns
    -------
    rho : CSC array of complex (D,D)
        Complete-space density matrix.

    """
    
    psi_L = sparse.csc_matrix(psi_L)
    
    psi = PI.T @ psi_L.T
    rho = psi.conjugate() * psi.T
    
    return rho
    

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


def siteDensityMatrix(rho,site,N):
    """
    Computes reduced density matrix on single site.

    Parameters
    ----------
    rho : CSC array of complex (D,D)
        Complete-space density matrix.
    site : int
        Site to compute density matrix on.
    N : int
        Number of sites.

    Returns
    -------
    rho_tr :  CSC array of complex (2,2)
        Site denstiy matrix.

    """
    
    dim_A = 2**site
    dim_B = 2**(N-site)
    rho_tr = rightPartialTrace(rho, dim_A, dim_B)
    rho_tr = leftPartialTrace(rho_tr, dim_A//2, 2)
    
    return rho_tr
    

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
    rho : CSC array of complex (2,2)
        Density matrix.
    dim : int
        Dimension of density matrix.

    Returns
    -------
    s : float
        Entanglement entropy.

    """
    
    rho = rho.toarray()
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
        rho = densityMatrix(psi,PI)
        rho_tr = siteDensityMatrix(rho,site,N)
        entropy_array[i] = entanglementEntropy(rho_tr,2)
        
    return entropy_array


def domainWallEntropyArray(psi_array,t_steps,PI,N):
    """
    Computes entanglement-entropy array between
    left and right half of the sites.

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
        rho = densityMatrix(psi_array[i],PI)
        
        dim_half = 2**(N//2)
        rho_tr = rightPartialTrace(rho, dim_half, dim_half)
        
        entropy_array[i] = entanglementEntropy(rho_tr, 2**(N//2))
        
    return entropy_array


def leftEntropyLattice(rho,l):
    """
    Computes entropy left-sublattice.

    Parameters
    ----------
    rho : CSC array of complex (2,2)
        Density matrix.
    l : int
        Number of layer (0 to N-1).

    Returns
    -------
    I_list : list (l+1)
        Entropy left-sublattice.
    """
    
    I_list = []
    
    if l == 0:
        I = entanglementEntropy(rho,2**(l+1))
        I_list.append(I)
        
        return I_list
    
    I = entanglementEntropy(rho,2**(l+1))
    
    rho_tr = rightPartialTrace(rho,2**l,2)
    I_sub = leftEntropyLattice(rho_tr,l-1)
    
    I_list.append(I)
    I_list = I_list + I_sub
    
    return I_list


def rightEntropyLattice(rho,l):
    """
    Computes entropy right-sublattice.

    Parameters
    ----------
    rho : CSC array of complex (2,2)
        Density matrix.
    l : int
        Number of layer (0 to N-1).

    Returns
    -------
    I_list : list (l+1)
        Entropy right-sublattice.
    """
    
    I_list = []
    
    if l == 0:
        I = entanglementEntropy(rho,2**(l+1))
        I_list.append([I])
        
        return I_list
    
    I = entanglementEntropy(rho,2**(l+1))
    
    rho_tr_left = rightPartialTrace(rho,2**l,2)
    rho_tr_right = leftPartialTrace(rho,2,2**l)
    
    I_sub_left = leftEntropyLattice(rho_tr_left,l-1)
    I_sub_right = rightEntropyLattice(rho_tr_right,l-1)
    
    I_list.append([I])
    
    for i in range(1,l+1):
        list_i = [I_sub_left[i-1]]
        list_i = list_i + I_sub_right[i-1]

        I_list.append(list_i)
        
    return I_list


def entropyLattice(rho,N):
    """
    Computes complete entropy lattice.

    Parameters
    ----------
    rho : CSC array of complex (2,2)
        Density matrix.
    N : int
        Number of sites.

    Returns
    -------
    staircase list
        Entropy lattice.

    """
    
    return rightEntropyLattice(rho,N-1)
        

def latticePoints(N):
    """
    Computes (x,y) positions of triangular lattice.

    Parameters
    ----------
    N : int
        Number of sites.

    Returns
    -------
    xs : array of float (N(N+1)//2)
        X coordinates.
    ys : array of float (N(N+1)//2)
        y coordinates
    """
    
    xs = np.array([],dtype=float)
    ys = np.array([],dtype=float)
    
    for i in range(N):
        y = (N-i-1)*np.sqrt(3)/2
        
        xs_i = np.linspace(-0.5*i,0.5*i,i+1)
        ys_i = np.full(i+1,y)
        
        xs = np.concatenate((xs,xs_i))
        ys = np.concatenate((ys,ys_i))
        
    return xs, ys
    
    
    
def informationLattice(rho,N):
    """
    Computes infromation lattice values.

    Parameters
    ----------
    rho : CSC array of complex (2,2)
        Density matrix.
    N : int
        Number of sites.

    Returns
    -------
    informationLat : list
        Ordered list of rows of entropy lattice.

    """
    
    entropyLat = entropyLattice(rho,N)
    informationLat = []
    
    for i in range(N):
        l = N-i-1
        lat_i = []
            
        for j in range(i+1):
            if i >= N-2:
                I_int = 0
            else:
                S_int = entropyLat[i+2][j+1]
                I_int = l-1 - S_int
            
            if i == N-1:
                I_A = 0
                I_B = 0
            else:
                S_A = entropyLat[i+1][j]
                S_B = entropyLat[i+1][j+1]
                I_A = l - S_A
                I_B = l - S_B
            
            S_AB = entropyLat[i][j]
            I_AB = l+1 - S_AB
            
            lat_i.append(I_AB - I_A - I_B + I_int)
        
        informationLat.append(lat_i)
    
    return informationLat
            
def areaLaw(rho,N):
    """
    Computes mutual information between connected
    region of sites and its complement.

    Parameters
    ----------
    rho : CSC array of complex (2,2)
        Density matrix.
    N : int
        Number of sites.

    Returns
    -------
    I : array (N-1)
        Mutual information for each subsystem.

    """
    
    I = np.zeros(N-1,dtype=float)
    
    for i in range(0,N-1):
        dim_L = 2**(i+1)
        dim_R = 2**(N-i-1)
        rho_L = rightPartialTrace(rho, dim_L, dim_R)
        rho_R = leftPartialTrace(rho, dim_L, dim_R)
        
        I_L = i+1 - entanglementEntropy(rho_L, dim_L)
        I_R = N-i-1 - entanglementEntropy(rho_R, dim_R)
        
        I[i] = N - I_L - I_R
    
    return I

#code for Peres lattice

def peresLattice(A,spectrum,eigvecs_herm):
    """
    Creates Peres lattice for given subspace
    operator A.

    Parameters
    ----------
    A : csc array of complex (M,M)
        L-magnetization subspace operator.
    spectrum : array of float (M)
        Spectrum of subspace hamiltonian.
    eigvecs_herm : array of complex (M,M)
        Normalized eigenvectors.

    Returns
    -------
    lattice : array of float (M,2)
        Points (E,<A>) of Peres lattice.

    """
    
    M = spectrum.size
    lattice = np.zeros((M,2),dtype="float")
    
    for i in range(M):
        E = spectrum[i]
        eigvec = eigvecs_herm[i]
        
        A_mean = np.conjugate(eigvec) @ A @ eigvec
        
        lattice[i] = [E,np.real(A_mean)]
        
    return lattice
    