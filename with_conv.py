import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras_tuner

def plot_series(time, series,prediction_series, format='-', start=0, end=None):
    plt.plot(time[start:end],series[start:end], format)
    plt.plot(prediction_series[start:end],prediction_series[start:end], format)
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

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1],window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset

def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(30, activation="relu", input_shape=[window_size]))
    model.add(tf.keras.layers.Dense(10, activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(
        hp.Choice('momentum', values=[.9,.8])
    )) # optimizer=tf.keras.optimizers.SGD(hp.Choice('momentum', values=[0.9,0.7,0.5,0.3]), learning_rate=1e-5)
    return model

def conv_model(dataset, window_size):
    model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv1D(filters=128,kernel_size=3, strides=1, padding="causal", activation="relu",
                               input_shape=[None,1]),
        tf.keras.layers.Dense(28, input_shape=[window_size], activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1)
    ]
    )
    model.compile(loss="mse", optimizer= tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.5))

    model.fit(dataset, epochs=100, verbose=1)

    return model

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

time = np.arange(4*365+1, dtype="float32")
baseline = 10
series = trend(time, 0.05)
baseline = 10
amplitude = 15
slop = 0.09
noise_level = 6

series = baseline + trend(time, slop) + seasonality(time, period=365, amplitude=amplitude)
series += noise(time, noise_level, seed=42)

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]

time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

dataset = windowed_dataset(series, window_size, batch_size, shuffle_buffer_size)

model = conv_model(dataset, window_size)

forcast = model_forecast(model, series[...,np.newaxis], window_size)
result = forcast[split_time - window_size : -1, -1, 0]

# plot_series(time,series,result)
print(result)




