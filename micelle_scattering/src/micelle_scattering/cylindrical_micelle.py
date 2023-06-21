import numpy as np

from scipy.integrate import quad
from scipy.special import j0, j1


class Cylindrical_Micelle:
    
    def __init__(
            self, 
            q_range: float, 
            R: float, 
            epsilon: float, 
            R_g: float, 
            N_agg: int, 
            r_beta: float
            ) -> None:
        
        """
        q_range: the values of q in an ascending order
        R: the radius of the core
        epsilon: the elongation coefficient
        R_g: the radius of gyration of the corona
        N_agg: the aggregation number
        r_beta: the ratio between the corona and core scattering lengths
        """
        
        self.q_range = q_range
        self.R = R
        self.epsilon = epsilon
        self.R_g = R_g
        self.N_agg = N_agg
        self.r_beta = r_beta

        self.L = epsilon*R


    def psi(self, x: float) -> float:
        
        return (1 - np.exp(-x))/x
    
    
    def Psi(self, alpha: float, q: float, R: float, L: float) -> float:
        
        qRsin = q*R*np.sin(alpha)
        qLcos = q*L*np.cos(alpha/2)
        
        return (2*j1(qRsin)/qRsin)*(np.sin(qLcos)/qLcos)


    def F_s(self) -> np.ndarray:
        
        q_range = self.q_range
        R = self.R        
        L = self.L
        
        def F_s_scalar(q: float, R: float, L: float) -> float:
            
            def func(alpha: float, q: float, R: float, L: float):
                
                Psi_ = self.Psi(alpha=alpha, q=q, R=R, L=L)
                
                return np.power(Psi_, 2)*np.sin(alpha)
            
            return quad(func=func, a=0, b=np.pi/2, args=(q, R, L))[0]
        
        F_s_vec = np.vectorize(F_s_scalar)
        
        return F_s_vec(q_range, R, L)


    def F_c(self) -> np.ndarray:
        
        q_range = self.q_range
        R_g = self.R_g
        
        qR_sq = np.power(q_range*R_g, 2)
        
        return 2*(np.exp(-qR_sq) - 1 + qR_sq)/np.power(qR_sq, 2)


    def Xi(self, alpha: float, q: float, R: float, L: float) -> float:
        
        qRsin = q*R*np.sin(alpha)
        qLcos = q*L*np.cos(alpha/2)
        
        B_0 = j0(qRsin)
        B_1 = j1(qRsin)
        
        return (R/(R + L))*2*B_1*np.cos(qLcos)/qRsin + (L/(R + L))*B_0*np.sin(qLcos)/qLcos


    def S_sc(self) -> np.ndarray:
        
        q_range = self.q_range
        R = self.R
        R_g = self.R_g
        L = self.L
        
        def S_sc_scalar(q: float, R: float, R_g: float, L: float) -> float:
            
            def func(alpha: float, q: float, R: float, R_g: float, L: float) -> float:
                
                Psi_ = self.Psi(alpha=alpha, q=q, R=R, L=L)
                Xi_ = self.Xi(alpha=alpha, q=q, R=R+R_g, L=L+2*R_g)
                
                return Psi_*Xi_*np.sin(alpha)
            
            psi_ = self.psi(q*R_g)
            
            return psi_*quad(func=func, a=0, b=np.pi/2, args=(q, R, R_g, L))[0]
        
        S_sc_vec = np.vectorize(S_sc_scalar)
        
        return S_sc_vec(q_range, R, R_g, L)


    def S_cc(self) -> np.ndarray:
        
        q_range = self.q_range
        R = self.R
        R_g = self.R_g
        L = self.L
        
        def S_cc_scalar(q: float, R: float, R_g: float, L: float) -> float:
            
            def func(alpha: float, q: float, R: float, R_g: float, L: float) -> float:
                
                Xi_ = self.Xi(alpha=alpha, q=q, R=R+R_g, L=L+2*R_g)
                
                return np.power(Xi_, 2)*np.sin(alpha)
            
            psi_ = self.psi(q*R_g)
            
            return np.power(psi_, 2)*quad(func=func, a=0, b=np.pi/2, args=(q, R, R_g, L))[0]
        
        S_cc_vec = np.vectorize(S_cc_scalar)
        
        return S_cc_vec(q_range, R, R_g, L)


    def F_micelle(self) -> np.ndarray:
        
        N_agg = self.N_agg
        r_beta = self.r_beta
        
        F_s_ = self.F_s()
        F_c_ = self.F_c()
        S_sc_ = self.S_sc()
        S_cc_ = self.S_cc()
        
        return (F_s_ + (r_beta**2)*F_c_/N_agg + 2*r_beta*S_sc_ + (N_agg - 1)*(r_beta**2)*S_cc_/N_agg)/((1 + r_beta)**2)


    def scattering_intensity(self) -> np.ndarray:
        
        F_micelle_ = self.F_micelle()
    
        return np.power(F_micelle_, 2)


def main(*args, **kwargs):
    pass


if __name__ == "__main__":
    main()
