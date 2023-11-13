import numpy as np
import tensorflow as tf

def get_data():
    data_file = 'data/hefei.csv'
    f = open(data_file)
    data = f.read()
    f.close()
    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]
    temperatures=[]

    for line in lines:
        if line:
            linedata = line.split(',')
            linedata = linedata[1:13]
            for item in linedata:
                if item:
                    temperatures.append(float(item))
    
    series = np.asarray(temperatures)
    time = np.arange(len(temperatures), dtype="float32")

    return time,series

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1],window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset

def rnn_model(dataset, valid_dataset):
    model = tf.keras.models.Sequential([
        tf.keras.layers.SimpleRNN(100, return_sequences=True, input_shape=[None,1]),
        tf.keras.layers.SimpleRNN(100),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=1.5e-6, momentum=0.9)
    model.compile(loss= tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
    history = model.fit(dataset, epochs=500,verbose=1, validation_data=valid_dataset)
    print(history)

def gru_model(dataset, valid_dataset):
    model = tf.keras.models.Sequential([
        tf.keras.layers.GRU(100, return_sequences=True, input_shape=[None,1]),
        tf.keras.layers.GRU(100),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=1.5e-6, momentum=0.9)
    model.compile(loss= tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
    history = model.fit(dataset, epochs=500,verbose=1, validation_data=valid_dataset)
    print(history)

def lstm_model(dataset, valid_dataset):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(100, return_sequences=True, input_shape=[None,1]),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=1.5e-6, momentum=0.9)
    model.compile(loss= tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
    history = model.fit(dataset, epochs=500,verbose=1, validation_data=valid_dataset)
    print(history)


time,series = get_data()
print("Size of Data {}".format(len(series)))
mean = series.mean(axis=0)
print('mean {}'.format(mean))
print("before normalization {}".format(series[0]))
series -= mean
std = series.std(axis=0)
series /= std

number_of_years =12*5
split = len(series) - number_of_years

time_train = time[:split]
x_train = series[:split]

time_valid = time[split:]
x_valid = series[split:]

window_size = 24
batch_size = 12
shuffle_buffer_size = 48

train_dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
valid_dataset = windowed_dataset(x_valid, window_size, batch_size, shuffle_buffer_size)

rnn_model(train_dataset, valid_dataset)

