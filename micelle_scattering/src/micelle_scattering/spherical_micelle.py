import numpy as np


class Spherical_Micelle:
    
    def __init__(
            self, 
            q_range: np.ndarray, 
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
        beta: the correction coefficient for corona correlations
        """
        
        self.q_range = q_range
        self.R = R
        self.epsilon = 1
        self.R_g = R_g
        self.N_agg = N_agg
        self.r_beta = r_beta
        
        sigma = (R_g**2)*N_agg/(4*((R + R_g)**2))
        
        self.beta = 1.42*sigma**1.04
    
    
    def corona_density(self) -> float:
        
        R = self.R
        R_g = self.R_g
        N_agg = self.N_agg
        
        return (R_g**2)*N_agg/(4*((R + R_g)**2))


    def Phi(self) -> np.ndarray:
        
        q_range = self.q_range
        R = self.R
        
        qR = q_range*R
        
        return 3*(np.sin(qR) - qR*np.cos(qR))/np.power(qR, 3)
        

    def F_core(self) -> np.ndarray:
                
        Phi_ = self.Phi()

        return np.power(Phi_, 2)


    def F_Debye(self) -> np.ndarray:
        
        q_range = self.q_range
        R_g = self.R_g
        
        qR_sq = np.power(q_range*R_g, 2)
        
        return 2*(np.exp(-qR_sq) - 1 + qR_sq)/np.power(qR_sq, 2)
        

    def psi(self) -> np.ndarray:
        
        q_range = self.q_range
        R_g = self.R_g
        
        qR_sq = np.power(q_range*R_g, 2)

        return (1 - np.exp(-qR_sq))/qR_sq


    def S_corona_corona(self) -> np.ndarray:

        q_range = self.q_range
        R = self.R
        R_g = self.R_g
        
        qRR = q_range*(R + R_g)
        
        psi_ = self.psi()
        
        return np.power(psi_, 2)*np.power(np.sin(qRR)/qRR, 2)


    def S_core_corona(self) -> np.ndarray:
        
        q_range = self.q_range
        R = self.R
        R_g = self.R_g
        
        qRR = q_range*(R + R_g)
        
        Phi_ = self.Phi()
        psi_ = self.psi()

        return Phi_*psi_*np.sin(qRR)/qRR


    def F_RPA(self) -> np.ndarray:
        
        F_Debye_ = self.F_Debye()
        beta = self.beta
        
        return F_Debye_/(1 + beta*F_Debye_)


    def F_corona(self) -> np.ndarray:
        
        N_agg = self.N_agg
        
        F_RPA_ = self.F_RPA()
        F_RPA_0 = F_RPA_[0]
        S_corona_corona_ = self.S_corona_corona()
        
        return F_RPA_/N_agg + (N_agg - F_RPA_0)*S_corona_corona_/N_agg


    def F_micelle(self) -> np.ndarray:
        
        r_beta = self.r_beta
        
        F_core_ = self.F_core()
        F_corona_ = self.F_corona()
        S_core_corona_ = self.S_core_corona()
        
        return (F_core_ + (r_beta**2)*F_corona_ + (2*r_beta)*S_core_corona_)/((1 + r_beta)**2)


    def scattering_intensity(self) -> np.ndarray:
        
        F_micelle_ = self.F_micelle()

        return np.power(F_micelle_, 2)


def main():
    pass


if __name__ == "__main__":
    main()
