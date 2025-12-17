import numpy as np
from scipy import signal

def rand_range(x, y):
    return np.random.uniform(x, y)

def rand_list(x):
    return x[np.random.randint(0, len(x))]

class RawBoost:
    def __init__(self, algo_id=[0, 1, 2, 3, 4], fs=16000):
        self.algo_id = algo_id
        self.fs = fs

    def process(self, x):
        # Randomly select an augmentation algorithm
        algo = rand_list(self.algo_id)

        if algo == 0:
            # No augmentation
            return x
        elif algo == 1:
            return self.lnl_convolutive_noise(x)
        elif algo == 2:
            return self.isd_additive_noise(x)
        elif algo == 3:
            return self.stationary_noise(x)
        elif algo == 4:
            # Combined: Convolutive + Impulsive
            x = self.lnl_convolutive_noise(x)
            return self.isd_additive_noise(x)
        else:
            return x

    # Algo 1: Linear and non-linear convolutive noise
    def lnl_convolutive_noise(self, x, N_f=5, n_list=[1, 2, 3, 4, 5], a_min=10, a_max=100):
        if len(x.shape) > 1: x = x.flatten()
        
        n = rand_list(n_list)
        a = rand_list(range(a_min, a_max))
        
        # Linear part
        b = np.array([1.0])
        for i in range(N_f):
            b_i = np.array([1.0, rand_range(-1, 1)])
            b = np.convolve(b, b_i)
            
        a_poly = np.array([1.0])
        for i in range(n):
            a_poly = np.convolve(a_poly, np.array([1.0, rand_range(-0.1, 0.1)]))
            
        x_linear = signal.lfilter(b, a_poly, x)
        
        # Non-linear part
        f = np.random.randn()
        x_nonlinear = x_linear + f * np.square(x_linear)
        
        # Gain adjustment
        rms_x = np.sqrt(np.mean(x**2))
        rms_aug = np.sqrt(np.mean(x_nonlinear**2))
        if rms_aug == 0: return x
        
        return x_nonlinear * (rms_x / rms_aug)

    # Algo 2: Impulsive signal dependent additive noise
    def isd_additive_noise(self, x, P=10, g_sd=2):
        if len(x.shape) > 1: x = x.flatten()
        
        beta = rand_list(range(5, P))
        
        noise = np.random.randn(len(x))
        noise_mask = np.random.choice([0, 1], size=len(x), p=[1 - 1/beta, 1/beta])
        noise = noise * noise_mask
        
        x_aug = x + g_sd * noise * x
        return x_aug

    # Algo 3: Stationary signal independent additive noise
    def stationary_noise(self, x, SNR_min=10, SNR_max=40):
        if len(x.shape) > 1: x = x.flatten()
        
        noise = np.random.randn(len(x))
        
        # Calculate SNR
        signal_power = np.sum(x ** 2)
        noise_power = np.sum(noise ** 2)
        
        target_snr = rand_range(SNR_min, SNR_max)
        target_snr_linear = 10 ** (target_snr / 10)
        
        required_noise_power = signal_power / target_snr_linear
        scale_factor = np.sqrt(required_noise_power / (noise_power + 1e-9))
        
        x_aug = x + noise * scale_factor
        return x_aug



