import matplotlib.pyplot as plt
import numpy as np

def plot_series(time, series, format='-', start=0, end=None):
    plt.plot(time[start:end],series[start:end], format)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    return np.where(season_time < 0.4, np.cos(season_time * 2 * np.pi),
                    1/np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeasts the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


time = np.arange(4*365+1, dtype="float32")
baseline = 10
series = trend(time, 0.05)
baseline = 10
amplitude = 15
slop = 0.09
noise_level = 6

series = baseline + trend(time, slop) + seasonality(time, period=365, amplitude=amplitude)
series += noise(time, noise_level, seed=42)

plot_series(time=time, series=series)