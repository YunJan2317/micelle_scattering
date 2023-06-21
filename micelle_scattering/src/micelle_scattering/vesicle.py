import numpy as np


class Vesicle:
    
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


    def F_cs(self) -> np.ndarray:
        
        q_range = self.q_range
        L = self.L
        
        qL = q_range*L/2
        
        return np.power(np.sin(qL)/qL, 2)
    
    
    def F_R(self) -> np.ndarray:
        
        q_range = self.q_range
        R = self.R
        
        qR = q_range*R
        
        return np.power(np.sin(qR)/(qR), 2)
    
    
    def F_s(self) -> np.ndarray:
        
        q_range = self.q_range
        R = self.R
        L = self.L
        
        F_cs_ = self.F_cs(q_range=q_range, L=L)
        F_R_ = self.F_R(q_range=q_range, R=R)
        
        return F_cs_*F_R_
    
    
    def F_c(self) -> np.ndarray:
        
        q_range = self.q_range
        R_g = self.R_g
        
        qR_sq = np.power(q_range*R_g, 2)
        
        return 2*(np.exp(-qR_sq) - 1 + qR_sq)/np.power(qR_sq, 2)
    
    
    def psi(self, x: np.ndarray) -> np.ndarray:
        
        return (1 - np.exp(-x))/x
    
    
    def S_sc(self) -> np.ndarray:
        
        q_range = self.q_range
        R = self.R
        R_g = self.R_g
        L = self.L
        
        psi_ = self.psi(q_range*R_g)
        sin_ = np.sin(q_range*L/2)/(q_range*L/2)
        cos_ = np.cos(q_range*(L/2 + R_g))
        F_R_ = self.F_R(q_range=q_range, R=R)
        
        return psi_*sin_*cos_*F_R_
    
    
    def S_cc(self) -> np.ndarray:
        
        q_range = self.q_range
        R = self.R
        R_g = self.R_g
        L = self.L
        
        psi_ = self.psi(q_range*R_g)
        cos_ = np.cos(q_range*(L/2 + R_g))
        F_R_ = self.F_R(q_range=q_range, R=R)
        
        return np.power(psi_, 2)*np.power(cos_, 2)*F_R_


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
