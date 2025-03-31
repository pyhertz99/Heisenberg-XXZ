import numpy as np


def chain(N, J_xy, J_z, boundary=0.0):
    """
    Creates M_xy and M_z connectivity matrices
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
        M_xy symmetric tridiagonal connectivity matrix.
    M_z : array (N,N) of float
        M_z symmetric tridiagonal connectivity matrix.

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


def lipkin(N, J_xy, J_z):
    """
    Creates M_xy and M_z connectivity matrices
    for Lipkin topology with constant coupling
    constants J_xy and J_z.
    
    Parameters
    ----------
    N : integer
        Number of sites.
    J_xy : float
        J_xy sites coupling constant.
    J_z : TYPE
        J_z sites coupling constant.

    Returns
    -------
    M_xy : array (N,N) of float 
        M_xy symmetric connectivity matrix.
    M_z : array (N,N) of float
        M_z symmetric connectivity matrix.

    """
    
    M_xy = np.full((N,N),J_xy/2)
    M_z = np.full((N,N),J_z/2)
    
    for i in range(N):
        M_xy[i,i] = 0
        M_z[i,i] = 0

    return M_xy, M_z

def chainFiniteRange(N, J_xy, J_z, q=1.0):
    """
    Creates M_xy and M_z connectivity matrices
    for finite range Heisenberg chain
    with constant coupling constants J_xy and J_z.
    
    Parameters
    ----------
    N : integer
        Number of sites.
    J_xy : float
        J_xy sites coupling constant.
    J_z : float
        J_z sites coupling constant.
    q : float
        degree of connectivity (0 local, 1 lipkin)

    Returns
    -------
    M_xy : array (N,N) of float 
        M_xy symmetric connectivity matrix.
    M_z : array (N,N) of float
        M_z symmetric connectivity matrix.

    """
        
    if q == 0.0:
        return chain(N,J_xy,J_z,1.0)
    
    gamma = 1 / np.tan(np.pi*q/2)
    
    M_xy = np.zeros((N,N))
    M_z = np.zeros((N,N))
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = min(np.abs(i-j),N-np.abs(i-j))
            M_xy[i,j] = J_xy / (2*d**gamma)
            M_z[i,j] = J_z / (2*d**gamma)
            
    return M_xy, M_z

def grid2D(n, m, J_xy, J_z, twist_n=False, twist_m=False):
    """
    Creates M_xy and M_z connectivity matrices
    for m*n periodic (torus) grid with adjacent
    iteractions with constant coupling
    constants J_xy and J_z.
    
    Parameters
    ----------
    n : integer
        Number of sites in x axis.
    m : integer
        Number of sites in y axis.
    J_xy : float
        J_xy sites coupling constant.
    J_z : float
        J_z sites coupling constant.
    twist_n : Bool
        twist boundary condition in x axis
    twist_m : Bool
        twist boundary condition in y axis

    Returns
    -------
    M_xy : array (N,N) of float 
        M_xy symmetric connectivity matrix.
    M_z : array (N,N) of float
        M_z symmetric connectivity matrix.

    """
    
    M = np.zeros((n*m,n*m))
    
    for i in range(n):
        for j in range(m):
            
            t_index = -1
            b_index = -1
            l_index = -1
            r_index = -1
            
            if j == 0:
                if twist_m:
                    t_index = (m-1)*n + (n-i-1)
                else:
                    t_index = (m-1)*n + i
            if j == m-1:
                if twist_m:
                    b_index = n-i-1
                else:
                    b_index = i
            if i == 0:
                if twist_n:
                    l_index = (n-j-1)*n + n-1
                else:
                    l_index = j*n + n-1
            if i == n-1:
                if twist_n:
                    r_index = (n-j-1)*n
                else:
                    r_index = j*n
            
            c_index = j*n + i
            if t_index == -1:
                t_index = (j-1)*n + i
            if b_index == -1:
                b_index = (j+1)*n + i
            if l_index == -1:
                l_index = j*n + i-1
            if r_index == -1:
                r_index = j*n + i+1
            
            M[c_index,t_index] = 1/2
            M[c_index,b_index] = 1/2
            M[c_index,l_index] = 1/2
            M[c_index,r_index] = 1/2
            
    return J_xy*M, J_z*M
      

def discreteFiniteRange(N,K,J_xy,J_z):
    """
    Creates M_xy and M_z connectivity matrices for
    1st to Kth neighbour connection with coupling
    constants J_xy and J_z. For K=1 equals chain
    CM, for K=N/2 equals Lipkin CM.

    Parameters
    ----------
    N : integer
        Number of sites.
    K : integer
        Number of neighbours to connect. Must be
        1 <= K <= N/2.
    J_xy : float
        J_xy sites coupling constant.
    J_z : TYPE
        J_z sites coupling constant.

    Returns
    -------
    M_xy : array (N,N) of float 
        M_xy symmetric connectivity matrix.
    M_z : array (N,N) of float
        M_z symmetric connectivity matrix.

    """
    
    M = np.zeros((N,N))
    
    
    for k in range(1,K+1):
        M_k = np.zeros((N,N))
        
        for i in range(N):
            for j in range(N):
                M_k[i,(i+k) % N] = 0.5
                M_k[i,i-k] = 0.5
        
        M += M_k
        
    return J_xy*M, J_z*M
                