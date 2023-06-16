import numpy as np

from scipy.integrate import quad
from scipy.special import j0, j1


def psi(x: float) -> float:
    
    return (1 - np.exp(-x))/x


def Psi(alpha: float, q: float, R: float, L: float) -> float:
    
    qRsin = q*R*np.sin(alpha)
    qLcos = q*L*np.cos(alpha/2)
    
    return (2*j1(qRsin)/qRsin)*(np.sin(qLcos)/qLcos)


def F_s(q_range: np.ndarray, R: float, L: float) -> np.ndarray:
    
    def F_s_scalar(q: float, R: float, L: float) -> float:
        
        def func(alpha: float, q: float, R: float, L: float):
            
            Psi_ = Psi(alpha=alpha, q=q, R=R, L=L)
            
            return np.power(Psi_, 2)*np.sin(alpha)
        
        return quad(func=func, a=0, b=np.pi/2, args=(q, R, L))
    
    F_s_vec = np.vectorize(F_s_scalar)
    
    return F_s_vec(q_range, R, L)


def F_c(q_range: np.ndarray, R_g: float) -> np.ndarray:
    
    qR_sq = np.power(q_range*R_g, 2)
    
    return 2*(np.exp(-qR_sq) - 1 + qR_sp)/np.power(qR_sq, 2)


def Xi(alpha: float, q: float, R: float, L: float) -> float:
    
    qRsin = q*R*np.sin(alpha)
    qLcos = q*L*np.cos(alpha/2)
    
    B_0 = j0(qRsin)
    B_1 = j1(qRsin)
    
    return (R/(R + L))*2*B_1*np.cos(qLcos)/qRsin + (L/(R + L))*B_0*np.sin(qLcos)/qLcos


def S_sc(q_range: np.ndarray, R: float, R_g: float, L: float) -> np.ndarray:
    
    def S_sc_scalar(q: float, R: float, R_g: float, L: float) -> float:
        
        def func(alpha: float, q: float, R: float, R_g: float, L: float) -> float:
            
            Psi_ = Psi(alpha=alpha, q=q, R=R, L=L)
            Xi_ = Xi(alpha=alpha, q=q, R=R+R_g, L=L+2*R_g)
            
            return Psi_*Xi_*np.sin(alpha)
        
        psi_ = psi(q*R_g)
        
        return psi_*quad(func=func, a=0, b=np.pi/2, args=(q, R, R_g, L))
    
    S_sc_vec = np.vectorize(S_sc_scalar)
    
    return S_sc_vec(q_range)


def S_cc(q_range: np.ndarray, R: float, R_g: float, L: float) -> np.ndarray:
    
    def S_cc_scalar(q: float, R: float, R_g: float, L: float) -> float:
        
        def func(alpha: float, q: float, R: float, R_g: float, L: float) -> float:
            
            Xi_ = Xi(alpha=alpha, q=q, R=R+R_g, L=L+2*R_g)
            
            return np.power(Xi_, 2)*np.sin(alpha)
        
        psi_ = psi(q*R_g)
        
        return np.power(psi_, 2)*quad(func=func, a=0, b=np.pi/2, args=(q, R, R_g, L))
    
    S_cc_vec = np.vectorize(S_cc_scalar)
    
    return S_cc_vec(q_range, R, R_g, L)


def F_micelle(
    q_range: np.ndarray, 
    R: float, 
    R_g: float, 
    L: float, 
    N_agg: int, 
    beta_s: float, 
    beta_c: float
) -> np.ndarray:
    
    F_s_ = F_s(q_range=q_range, R=R, L=L)
    F_c_ = F_c(q_range=q_range, R_g=R_g)
    S_sc_ = S_sc(q_range=q_range, R=R, R_g=R_g, L=L)
    S_cc_ = S_cc(q_range=q_range, R=R, R_g=R_g, L=L)
    
    F_abs = (N_agg**2)*(beta_s**2)*F_s_ + N*(beta_c**2)*F_c_ + /
            2*(N_agg**2)*beta_s*beta_c*S_sc + N_agg*(N_agg - 1)*(beta_c**2)*S_cc_
    
    return F_abs/((N_agg**2)*((beta_s + beta_c)**2))


def I_micelle(
    q_range: np.ndarray, 
    R: float, 
    R_g: float, 
    L: float, 
    N_agg: int, 
    beta_s: float, 
    beta_c: float, 
) -> np.ndarray:
    
    F_micelle_ = F_micelle(
        q_range=q_range, 
        R=R, 
        R_g=R_g, 
        L=L, 
        N_agg=N_agg, 
        beta_s=beta_s, 
        beta_c=beta_c, 
    )

    return np.power(F_micelle_, 2)


def main(*args, **kwargs):
    pass


if __name__ == "__main__":
    main()
