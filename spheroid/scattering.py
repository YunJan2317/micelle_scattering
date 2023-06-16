import numpy as np

from scipy.integrate import quad


def Phi(x: float) -> float:
    
    return 3*(np.sin(x) - q*R*np.cos(x))/np.power(x, 3)


def psi(x: float) -> float:
    
    return (1 - np.exp(-x))/x


def r_spheroid(alpha: float, R: float, epsilon: float) -> float:
    
    return R*np.power(np.power(np.sin(alpha), 2) + np.power(epsilon*np.cos(alpha), 2), 1/2)


def F_s(q_range: np.ndarray, R: float, epsilon: float) -> np.ndarray:
    
    def F_s_scalar(q: float, R: float, epsilon: float) -> float:
        
        def func(alpha: float, q: float, R: float, epsilon: float) -> float:
            
            r_ = r_spheroid(alpha=alpha, R=R, epsilon=epsilon)
            Phi_ = Phi(q*r_)
            
            return np.power(Phi_, 2)*np.sin(alpha)
        
        return quad(func=func, a=0, b=np.pi/2, args=(q, R, epsilon))
    
    F_s_vec = np.vectorize(F_s_scalar)
    
    return F_s_vec(q_range, R, epsilon)


def F_c(q_range: np.ndarray, R_g: float) -> np.ndarray:
    
    qR_sq = np.power(q_range*R_g, 2)
    
    return 2*(np.exp(-qR_sq) - 1 + qR_sp)/np.power(qR_sq, 2)


def S_sc(q_range: np.ndarray, R: float, R_g: float, epsilon: float) -> np.ndarrray:
    
    def S_sc_scalar(q: float, R: float, R_g: float, epsilon: float) -> float:
        
        def func(alpha: float, q: float, R: float, R_g: float, epsilon: float) -> float:
            
            r_ = r_spheroid(alpha=alpha, R=R, epsilon=epsilon)
            Phi_ = Phi(q*r_)
            
            return Phi_*(np.sin(q*(r_ + R_g))/(q*(r_ + R_g)))*np.sin(alpha)
        
        psi_ = psi(q*R_g)
        
        return psi_*quad(func=func, a=0, b=np.pi/2, args=(q, R, R_g, epsilon))
    
    S_sc_vec = np.vectorize(S_sc_scalar)
    
    return S_sc_vec(q_range, R, R_g, epsilon)


def S_cc(q_range: np.ndarray, R: float, R_g: float, epsilon: float) -> np.ndarray: 
    
    def S_cc_scalar(q: float, R: float, R_g: float, epsilon: float) -> float:
        
        def func(alpha: float, q: float, R: float, R_g: float, epsilon: float) -> float:
            
            r_ = r_spheroid(alpha=alpha, R=R, epsilon=epsilon)
            
            return np.power(np.sin(q*(r_ + R_g))/(q*(r_ + R_g)), 2)*np.sin(alpha)
        
        psi_ = psi(q*R_g)
        
        return np.power(psi, 2)*quad(func=func, a=0, b=np.pi/2, args=(q, R, R_g, epsilon))
    
    S_cc_vec = np.vectorize(S_cc_scalar)
    
    return S_cc_vec(q_range, R, R_g, epsilon)


def F_micelle(
    q_range: np.ndarray, 
    R: float, 
    R_g: float, 
    epsilon: float, 
    N_agg: int, 
    beta_s: float, 
    beta_c: float
) -> np.ndarray:
    
    F_s_ = F_s(q_range=q_range, R=R, epsilon=epsilon)
    F_c_ = F_c(q_range=q_range, R_g=R_g)
    S_sc_ = S_sc(q_range=q_range, R=R, R_g=R_g, epsilon=epsilon)
    S_cc_ = S_cc(q_range=q_range, R=R, R_g=R_g, epsilon=epsilon)
    
    F_abs = (N_agg**2)*(beta_s**2)*F_s_ + N*(beta_c**2)*F_c_ + /
            2*(N_agg**2)*beta_s*beta_c*S_sc + N_agg*(N_agg - 1)*(beta_c**2)*S_cc_
    
    return F_abs/((N_agg**2)*((beta_s + beta_c)**2))


def I_micelle(
    q_range: np.ndarray, 
    R: float, 
    R_g: float, 
    epsilon: float, 
    N_agg: int, 
    beta_s: float, 
    beta_c: float, 
) -> np.ndarray:
    
    F_micelle_ = F_micelle(
        q_range=q_range, 
        R=R, 
        R_g=R_g, 
        epsilon=epsilon, 
        N_agg=N_agg, 
        beta_s=beta_s, 
        beta_c=beta_c, 
    )

    return np.power(F_micelle_, 2)


def main(*args, **kwargs):
    pass


if __name__ == "__main__":
    main()
