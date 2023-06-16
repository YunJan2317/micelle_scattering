import numpy as np


"""
https://pubs.acs.org/action/showCitFormats?doi=10.1021/acs.jpcb.0c04120&ref=pdf
https://scripts.iucr.org/cgi-bin/paper?cj2032
"""

def q_range(q_start: float, q_end: float, q_step: float) -> np.ndarray:
    
    return np.arange(q_start, q_end, q_step)


def corona_density(R: float, R_g: float, N_agg: int) -> float:
    """
    R: the radius of the micellar core
    R_g: the radius of gyration of the corona chain
    N_agg: the aggregation number
    """
    
    return (R_g**2)*N_agg/(4*((R + R_g)**2))


def Phi(q_range: np.ndarray, R: float) -> np.ndarray:
    """
    q_range: a range of q values in an ascending order
    R: the radius of the micellar core
    """
    
    qR = q_range*R
    
    return 3*(np.sin(qR) - qR*np.cos(qR))/np.power(qR, 3)
    

def F_core(q_range: np.ndarray, R: float) -> np.ndarray:
    """
    q_range: a range of q values in an ascending order
    R: the radius of the micellar core
    """
    
    Phi_ = Phi(q_range=q_range, R=R)

    return np.power(Phi_, 2)


def F_Debye(q_range: np.ndarray, R_g: float) -> np.ndarray:
    """
    q_range: a range of q values in an ascending order
    R_g: the radius of gyration of the corona chain
    """
    
    qR_sq = np.power(q_range*R_g, 2)
    
    return 2*(np.exp(-qR_sq) - 1 + qR_sq)/np.power(qR_sq, 2)
    

def psi(q_range: np.ndarray, R_g: float) -> np.ndarray:
    """
    q_range: a range of q values in an ascending order
    R_g: the radius of gyration of the corona chain
    """
    
    qR_sq = np.power(q_range*R_g, 2)

    return (1 - np.exp(-qR_sq))/qR_sq


def S_corona_corona(q_range: np.ndarray, R: float, R_g: float) -> np.ndarray:
    """
    q_range: a range of q values in an ascending order
    R: the radius of the micellar core
    R_g: the radius of gyration of the corona chain
    """
    
    qRR = q_range*(R + R_g)
    
    psi_ = psi(q_range=q_range, R_g=R_g)
    
    return np.power(psi_, 2)*np.power(np.sin(qRR)/qRR, 2)


def S_core_corona(q_range: np.ndarray, R: float, R_g: float) -> np.ndarray:
    """
    q_range: a range of q values in an ascending order
    R: the radius of the micellar core
    R_g: the radius of gyration of the corona chain
    """
    
    qRR = q_range*(R + R_g)
    
    Phi_ = Phi(q_range=q_range, R=R)
    psi_ = psi(q_range=q_range, R_g=R_g)

    return Phi_*psi_*np.sin(qRR)/qRR


def F_RPA(q_range: np.ndarray, R_g: float, beta: float) -> np.ndarray:
    """
    q_range: a range of q values in an ascending order
    R_g: the radius of gyration of the corona chain
    beta: an empirical coefficient
    """
    
    F_Debye_ = F_Debye(q_range=q_range, R_g=R_g)
    
    return F_Debye_/(1 + beta*F_Debye_)


def F_corona(q_range: np.ndarray, R: float, R_g: float, N_agg: int, beta: float) -> np.ndarray:
    """
    q_range: a range of q values in an ascending order
    R: the radius of the micellar core
    R_g: the radius of gyration of the corona chain
    N_agg: the aggregation number
    beta: an empirical coefficient
    """
    
    F_RPA_ = F_RPA(q_range=q_range, R_g=R_g, beta=beta)
    F_RPA_0 = F_RPA_[0]
    S_corona_corona_ = S_corona_corona(q_range=q_range, R=R, R_g=R_g)
    
    return F_RPA_/N_agg + (N_agg - F_RPA_0)*S_corona_corona_/N_agg


def F_micelle(
    q_range: np.ndarray, 
    R: float, 
    R_g: float, 
    N_agg: int, 
    rho_core: float, 
    rho_corona: float, 
    beta: float
) -> np.ndarray:

    """
    q_range: a range of q values in an ascending order
    R: the radius of the micellar core
    R_g: the radius of gyration of the corona chain
    N_agg: the aggregation number
    rho_core: the excess scattering length of the core
    rho_corona: the excess scattering length of the corona
    beta: an empirical coefficient
    """
    
    F_core_ = F_core(q_range=q_range, R=R)
    F_corona_ = F_corona(q_range=q_range, R=R, R_g=R_g, N_agg=N_agg, beta=beta)
    S_core_corona_ = S_core_corona(q_range=q_range, R=R, R_g=R_g)
    
    return ((rho_core**2)*F_core_ + (rho_corona**2)*F_corona_ + (2*rho_core*rho_corona)*S_core_corona_)\
            /((rho_core + rho_corona)**2)


def I_micelle(
    q_range: np.ndarray, 
    R: float, 
    R_g: float, 
    N_agg: int, 
    rho_core: float, 
    rho_corona: float, 
) -> np.ndarray:

    """
    q_range: a range of q values in an ascending order
    R: the radius of the micellar core
    R_g: the radius of gyration of the corona chain
    N_agg: the aggregation number
    rho_core: the excess scattering length of the core
    rho_corona: the excess scattering length of the corona
    """
    
    corona_density_ = corona_density(R=R, R_g=R_g, N_agg=N_agg)
    
    beta = 1.42*corona_density_**1.04
    
    F_micelle_ = F_micelle(
        q_range=q_range, 
        R=R, 
        R_g=R_g, 
        N_agg=N_agg, 
        rho_core=rho_core, 
        rho_corona=rho_corona, 
        beta=beta
    )

    return np.power(F_micelle_, 2)


def main():
    pass


if __name__ == "__main__":
    main()
