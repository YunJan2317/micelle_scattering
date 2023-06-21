import numpy as np

from scipy.integrate import quad


class Spheroidal_Micelle:
    
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
        """
        
        self.q_range = q_range
        self.R = R
        self.epsilon = epsilon
        self.R_g = R_g
        self.N_agg = N_agg
        self.r_beta = r_beta
    
    
    def Phi(self, x: float) -> float:
        
        return 3*(np.sin(x) - x*np.cos(x))/np.power(x, 3)
    
    
    def psi(self, x: float) -> float:
        
        return (1 - np.exp(-x))/x


    def r_spheroid(self, alpha: float) -> float:
        
        R = self.R
        epsilon = self.epsilon
        
        return R*np.power(np.power(np.sin(alpha), 2) + np.power(epsilon*np.cos(alpha), 2), 1/2)
    
    
    def F_s(self) -> np.ndarray:
        
        q_range = self.q_range
        R = self.R
        epsilon = self.epsilon
        
        def F_s_scalar(q: float, R: float, epsilon: float) -> float:
            
            def func(alpha: float, q: float, R: float, epsilon: float) -> float:
                
                r_ = self.r_spheroid(alpha=alpha, R=R, epsilon=epsilon)
                Phi_ = self.Phi(q*r_)
                
                return np.power(Phi_, 2)*np.sin(alpha)
            
            return quad(func=func, a=0, b=np.pi/2, args=(q, R, epsilon))[0]
        
        F_s_vec = np.vectorize(F_s_scalar)
        
        return F_s_vec(q_range, R, epsilon)


    def F_c(self) -> np.ndarray:
        
        q_range = self.q_range
        R_g = self.R_g
        
        qR_sq = np.power(q_range*R_g, 2)
        
        return 2*(np.exp(-qR_sq) - 1 + qR_sq)/np.power(qR_sq, 2)


    def S_sc(self) -> np.ndarray:
        
        q_range = self.q_range
        R = self.R
        epsilon = self.epsilon
        R_g = self.R_g
        
        def S_sc_scalar(q: float, R: float, epsilon: float, R_g: float) -> float:
            
            def func(alpha: float, q: float, R: float, epsilon: float, R_g: float) -> float:
                
                r_ = self.r_spheroid(alpha=alpha, R=R, epsilon=epsilon)
                Phi_ = self.Phi(q*r_)
                
                return Phi_*(np.sin(q*(r_ + R_g))/(q*(r_ + R_g)))*np.sin(alpha)
            
            psi_ = self.psi(q*R_g)
            
            return psi_*quad(func=func, a=0, b=np.pi/2, args=(q, R, R_g, epsilon))[0]
        
        S_sc_vec = np.vectorize(S_sc_scalar)
        
        return S_sc_vec(q_range, R, epsilon, R_g)


    def S_cc(self) -> np.ndarray: 
        
        q_range = self.q_range
        R = self.R
        epsilon = self.epsilon
        R_g = self.R_g
        
        def S_cc_scalar(q: float, R: float, epsilon: float, R_g: float) -> float:
            
            def func(alpha: float, q: float, R: float, epsilon: float, R_g: float) -> float:
                
                r_ = self.r_spheroid(alpha=alpha, R=R, epsilon=epsilon)
                
                return np.power(np.sin(q*(r_ + R_g))/(q*(r_ + R_g)), 2)*np.sin(alpha)
            
            psi_ = self.psi(q*R_g)
            
            return np.power(psi_, 2)*quad(func=func, a=0, b=np.pi/2, args=(q, R, R_g, epsilon))[0]
        
        S_cc_vec = np.vectorize(S_cc_scalar)
        
        return S_cc_vec(q_range, R, epsilon, R_g)


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
