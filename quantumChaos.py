import numpy as np
from scipy.special import gamma
from scipy.optimize import curve_fit


def integratedLevelDensity(spectrum):
    """
    Computes integrated (cumulative) level density.
    
    Parameters
    ----------
    spectrum : array of float
        Sorted (increasing) array of eigenvalues.

    Returns
    -------
    int_density : array of float
        Integrated spectrum.
    """
    
    points = spectrum.size
    int_density = np.linspace(1,points,points)
        
    return int_density


def unfoldSpectrum(spectrum,deg,cut_edges):
    """
    Computes unfolded spectrum
    using polynomial unfolding.

    Parameters
    ----------
    spectrum : array of float
        Sorted (increasing) array of eigenvalues.
    deg : int
        Degree of unfolding polynomial.
    cut_edges : int
        Number of spectrum points to cut at both edges.

    Returns
    -------
    unfolded_spectrum : array of float
        Unfolded spectrum.
    polynom : np.polynom
        Polynom which fits the integrated spectrum.

    """
    
    points = spectrum.size
    
    int_density = integratedLevelDensity(spectrum)
    
    #cut edges
    int_density = int_density[cut_edges:points-cut_edges]
    spectrum = spectrum[cut_edges:points-cut_edges]
    
    polynom_coefs = np.polyfit(spectrum, int_density, deg)
    polynom = np.poly1d(polynom_coefs)
    
    #unfold spectrum
    unfolded_spectrum = np.zeros(points-2*cut_edges)
    for i in range(points-2*cut_edges):
        unfolded_spectrum[i] = polynom(spectrum[i])
    
    return unfolded_spectrum, polynom


def levelDifference(unfolded_spectrum):
    """
    Computes difference between adjacent energy levels.

    Parameters
    ----------
    unfolded_spectrum : array of float
        Unfolded spectrum.

    Returns
    -------
    level_dif : array of float
        Differences of adjacent energy levels.

    """
    
    points = unfolded_spectrum.size
    level_dif = np.zeros(points-1)
    
    for i in range(points-1):
        L_1 = unfolded_spectrum[i]
        L_2 = unfolded_spectrum[i+1]
        #avoid numerical error due to small numbers
        if L_2 - L_1 < 0:
            level_dif[i] = 0
        else:
            if L_2 - L_1 < 6.0:
                level_dif[i] = L_2 - L_1
    
    return level_dif


def brodyDistribution(x,beta):
    """
    Brody distribution function.

    Parameters
    ----------
    x : positive float
        Independent variable.
    beta : float
        Beta parameter.
    w : float
        Parameter which scales the x axis.

    Returns
    -------
    float
        Value of Brody distribution at point x.

    """
    
    #prevent from inserting negative x
    if np.any(x < 0):
        return np.inf
    
    b = gamma((beta+2)/(beta+1))**(beta+1)
    a = (beta+1)*b
    
    return  a * x**beta * np.exp(-b*x**(beta+1))


def brodyFit(level_dif,bins):
    """
    Computes histogram from level differences
    and fits with Brody distribution.

    Parameters
    ----------
    level_dif : array of float
        Differences of adjacent energy levels.
    bins : int
        Number of bins to use for histogram.

    Returns
    -------
    hist : array of float
        Y values of histogram.
    bin_centres : array of float
        X values (centres of bins) of histogram.
    beta : float
        Fitted beta parameter.
    cov_beta : float
        Error of fitted beta parameter.

    """
    
    hist, bin_edges = np.histogram(level_dif,bins=bins,density=True)
    
    #scale histogram
    
    #compute centres of bins
    bin_centres = [(bin_edges[i]+bin_edges[i+1])/2.0
                  for i in range(len(bin_edges)-1)]
    
    #set bounds for parameters (-0.95 is optimal minimum)
    bounds = ([-1.0,1.0])
    
    fit_result = curve_fit(brodyDistribution,
                           bin_centres, hist, bounds=bounds)
    
    return hist, bin_centres, fit_result[0], fit_result[1]


def ratioConsecutiveLevels(spectrum):
    """
    Performs statistical chaoticity test
    on not-unfolded spectrum.

    Parameters
    ----------
    spectrum : array of float
        Sorted (increasing) array of eigenvalues.

    Returns
    -------
    mean_r : float
        Mean value of ratio of consecutive levels.
    sigma_r : float
        Standard deviation of consecutive levels.

    """
    
    points = spectrum.size
    
    R = np.zeros(points-2)
    for i in range(1,points-1):
        s_1 = spectrum[i+1] - spectrum[i]
        s_2 = spectrum[i] - spectrum[i-1]
        
        #if values are too small, return 0
        try:
            r = s_1/s_2
            R[i-1] = min(r, 1/r)
        except FloatingPointError:
            R[i-1] = 0
            
    return np.mean(R), np.std(R)
    
    
    
    

    
    