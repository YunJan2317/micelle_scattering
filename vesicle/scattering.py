import numpy as np


def F_cs(q_range: np.ndarray, L: float) -> np.ndarray:
    
    qL = q_range*L/2
    
    return np.power(np.sin(qL)/qL, 2)


def F_R(q_range: np.ndarray, R: float) -> np.ndarray:
    
    return np.power(np.sin(q_range*R)/(q_range*R), 2)


def F_s(q_range, R: float, L: float) -> np.ndarray:
    
    F_cs_ = F_cs(q_range=q_range, L=L)
    F_R_ = F_R(q_range=q_range, R=R)
    
    return F_cs_*F_R_


def F_c(q_range: np.ndarray, R_g: float) -> np.ndarray:
    
    qR_sq = np.power(q_range*R_g, 2)
    
    return 2*(np.exp(-qR_sq) - 1 + qR_sp)/np.power(qR_sq, 2)


def psi(x: np.ndarray) -> np.ndarray:
    
    return (1 - np.exp(-x))/x


def S_sc(q_range: np.ndarray, R: float, R_g: float, L: float) -> np.ndarray:
    
    psi_ = psi(q_range*R_g)
    sin_ = np.sin(q_range*L/2)/(q_range*L/2)
    cos_ = np.cos(q_range*(L/2 + R_g))
    F_R_ = F_R(q_range=q_range, R=R)
    
    return psi_*sin_*cos_*F_R_


def S_cc(q_range: np.ndarray, R: float, R_g: float, L: float) -> np.ndarray:
    
    psi_ = psi(q_range*R_g)
    cos_ = np.cos(q_range*(L/2 + R_g))
    F_R_ = F_R(q_range=q_range, R=R)
    
    return np.power(psi_, 2)*np.power(cos_, 2)*F_R_


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
