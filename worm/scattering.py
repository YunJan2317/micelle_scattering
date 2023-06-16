import numpy as np

from scipy.special import j0, j1, sici


def F_cs(q_range: np.ndarray, R: float) -> np.ndarray:
    
    return np.power(2*j1(q_range*R)/(q_range*R))


def F_L(q_range: np.ndarray, L: float) -> np.ndarray:
    
    qL = q_range*L
    
    Si_, _ = sici(qL)
    
    return 2*Si_/qL - 4*np.power(np.sin(qL/2)/qL, 2)


def F_s(q_range: np.ndarray, R: float, L: float) -> np.ndarray:
    
    F_cs_ = F_cs(q_range=q_range, R=R)
    F_L_ = F_L(q_range=q_range, L=L)
    
    return F_cs_*F_L_


def F_c(q_range: np.ndarray, R_g: float) -> np.ndarray:
    
    qR_sq = np.power(q_range*R_g, 2)
    
    return 2*(np.exp(-qR_sq) - 1 + qR_sp)/np.power(qR_sq, 2)


def psi(x: np.ndarray) -> np.ndarray:
    
    return (1 - np.exp(-x))/x


def S_sc(q_range: np.ndarray, R: float, R_g: float, L: float) -> np.ndarray:
    
    psi_ = psi(q_range*R_g)
    B_1_ = j1(q_range*R)
    B_0_ = j0(q_range*(R + R_g))
    F_L_ = F_L(q_range=q_range, L=L)
    
    return psi_*(2*B_1_/(q_range*R))*B_0_*F_L_


def S_cc(q_range: np.ndarray, R: float, R_g: float, L: float) -> np.ndarray:
    
    psi_ = psi(q_range*R_g)
    B_0_ = j0(q_range*(R + R_g))
    F_L_ = F_L(q_range=q_range, L=L)
    
    return np.power(psi_, 2)*np.power(B_0_, 2)*F_L_


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
