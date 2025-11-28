# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 08:58:48 2025

@author: jesus
"""


import tensorflow as tf
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
    
df = pd.read_csv( "C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/CTA_-_Ridership_-_Daily_Boarding_Totals.csv", parse_dates=["service_date"] )
df.columns = ["date", "day_type", "bus", "rail", "total"]
df = df.sort_values("date").set_index("date")
dr = df.drop("total", axis=1)
df = df.drop_duplicates()

df.head()

df["2019-03":"2019-05"].plot(grid=True, marker=".", figsize=(8, 3.5))
plt.show()


diff_7 = df[["bus", "rail"]].diff(7)["2019-03":"2019-05"]

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
df.plot(ax=axs[0], legend=False, marker=".") 
df.shift(7).plot(ax=axs[0], grid=True, legend=False, linestyle=":")  
diff_7.plot(ax=axs[1], grid=True, marker=".")  
axs[0].set_ylim([170_000, 900_000])  
plt.show()


list(df.loc["2019-05-25":"2019-05-27"]["day_type"])



diff_7.abs().mean()



targets = df[["bus", "rail"]]["2019-03":"2019-05"]
(diff_7 / targets).abs().mean()


period = slice("2001", "2019")
try:
    df_monthly = df.select_dtypes(include="number").resample('ME').mean()  # compute the mean for each month
    rolling_average_12_months = df_monthly.loc[period].rolling(window=12).mean()
except ValueError as ex:
    try:
        df_monthly = df.select_dtypes(include="number").resample('M').mean()  # compute the mean for each month
        rolling_average_12_months = df_monthly.loc[period].rolling(window=12).mean()
    except ValueError as ex:
        df_monthly = df.resample('M').mean()  # compute the mean for each month
        rolling_average_12_months = df_monthly[period].rolling(window=12).mean()

fig, ax = plt.subplots(figsize=(8, 4))
df_monthly[period].plot(ax=ax, marker=".")
rolling_average_12_months.plot(ax=ax, grid=True, legend=False)
plt.show()



df_monthly.diff(12)[period].plot(grid=True, marker=".", figsize=(8, 3))
plt.show()



origin, today = "2019-01-01", "2019-05-31"
rail_series = df.loc[origin:today]["rail"].asfreq("D")
model = ARIMA( rail_series, order=(1,0,0), seasonal_order=(0,1,1,7) )
model = model.fit()
y_pred = model.forecast() 

origin, start_date, end_date = "2019-01-01", "2019-03-01", "2019-05-31"
time_period = pd.date_range(start_date, end_date)
rail_series = df.loc[origin:end_date]["rail"].asfreq("D")
y_preds = []

for today in time_period.shift(-1):
    model = ARIMA( rail_series[origin:today], order=(1,0,0), seasonal_order=(0,1,1,7) )
    model = model.fit()
    y_pred = model.forecast()[0]
    y_preds.append(y_pred)
    
    
y_preds = pd.Series( y_preds, index=time_period )
mae = (y_preds - rail_series[time_period]).abs().mean()

my_series = [0,1,2,3,4,5]
my_dataset = tf.keras.utils.timeseries_dataset_from_array( my_series, targets=my_series[3:], sequence_length=3, batch_size=2 )
list(my_dataset)

for window_dataset in tf.data.Dataset.range(6).window(4, shift=1):
    for element in window_dataset:
        print(f"{element}", end=" ")
    print()

dataset = tf.data.Dataset.range(6).window(4, shift=1, drop_remainder=True)
dataset = dataset.flat_map( lambda window_dataset: window_dataset.batch(4) )
for window_tensor in dataset:
    print(f"{window_tensor}")


def to_window(dataset, length):
    dataset = dataset.window( length, shift=1, drop_remainder=True )
    return dataset.flat_map( lambda window_ds: window_ds.batch(length) )


dataset = to_window(tf.data.Dataset.range(6), 4)
dataset = dataset.map( lambda window: ( window[:-1], window[-1] ))
list(dataset.batch(2))


rail_train = df["rail"]["2016-01":"2018-12"] / 1e6
rail_valid = df["rail"]["2019-01":"2019-05"] / 1e6
rail_test = df["rail"]["2019-06":] / 1e6

seq_length = 56
train_ds = tf.keras.utils.timeseries_dataset_from_array( rail_train.to_numpy(), targets=rail_train[seq_length:],
                                                        sequence_length=seq_length, batch_size=32, shuffle=True, seed=42)
valid_ds = tf.keras.utils.timeseries_dataset_from_array( rail_valid.to_numpy(), targets=rail_valid[seq_length:],
                                                        sequence_length=seq_length, batch_size=32)



tf.random.set_seed(42)
model = tf.keras.Sequential([ tf.keras.layers.Dense(1, input_shape=[seq_length])  ])
early_stopping_cb = tf.keras.callbacks.EarlyStopping( monitor="val_mae", patience=50, restore_best_weights=True )
opt = tf.keras.optimizers.SGD( learning_rate=0.02, momentum=0.9 )
model.compile( loss=tf.keras.losses.Huber(), optimizer=opt, metrics=["mae"] )
history = model.fit( train_ds, validation_data=valid_ds, epochs=500, callbacks=[early_stopping_cb] )
                              
model = tf.keras.Sequential( [tf.keras.layers.SimpleRNN( 1, input_shape=[None, 1] )] )

univar_model = tf.keras.Sequential([ tf.keras.layers.SimpleRNN( 32, input_shape=[None, 1] ), 
                                    tf.keras.layers.Dense(1)])

deep_model = tf.keras.Sequential( [ tf.keras.layers.SimpleRNN(32, return_sequences=True, input_shape=[None, 1]),
                                   tf.keras.layers.SimpleRNN(32, return_sequences=True),
                                   tf.keras.layers.SimpleRNN(32),
                                   tf.keras.layers.Dense(1)] )

df_mulvar = df[["bus", "rail"]] / 1e6 
df_mulvar["next_day_type"] = df["day_type"].shift(-1) 
df_mulvar = pd.get_dummies(df_mulvar, dtype=float)

mulvar_train = df_mulvar["2016-01":"2018-12"]
mulvar_valid = df_mulvar["2019-01":"2019-05"]
mulvar_test = df_mulvar["2019-06":]

tf.random.set_seed(42) 
train_mulvar_ds = tf.keras.utils.timeseries_dataset_from_array( mulvar_train.to_numpy(), targets=mulvar_train["rail"][seq_length:],
                                                                sequence_length=seq_length,
                                                                batch_size=32,
                                                                shuffle=True,
                                                                seed=42)


valid_mulvar_ds = tf.keras.utils.timeseries_dataset_from_array( mulvar_valid.to_numpy(), targets=mulvar_valid["rail"][seq_length:],
                                                               sequence_length=seq_length,
                                                               batch_size=32)

mulvar_model = tf.keras.Sequential([ tf.keras.layers.SimpleRNN(32, input_shape=[None, 5]), tf.keras.layers.Dense(1) ])




def fit_and_evaluate(model, train_set, valid_set, learning_rate, epochs=500):
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_mae", patience=50, restore_best_weights=True)
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=["mae"])
    history = model.fit(train_set, validation_data=valid_set, epochs=epochs,
                        callbacks=[early_stopping_cb])
    valid_loss, valid_mae = model.evaluate(valid_set)
    return valid_mae * 1e6


fit_and_evaluate(mulvar_model, train_mulvar_ds, valid_mulvar_ds,
                 learning_rate=0.05)


X = rail_valid.to_numpy()[ np.newaxis, :seq_length, np.newaxis]
for step_ahead in range(14):
    y_pred_one = univar_model.predict(X)
    X = np.concatenate([X, y_pred_one.reshape(1,1,1)], axis=1)
    
    
Y_pred = pd.Series(X[0, -14:, 0],
                   index=pd.date_range("2019-02-26", "2019-03-11"))

fig, ax = plt.subplots(figsize=(8, 3.5))
(rail_valid * 1e6)["2019-02-01":"2019-03-11"].plot(
    label="True", marker=".", ax=ax)
(Y_pred * 1e6).plot(
    label="Predictions", grid=True, marker="x", color="r", ax=ax)
ax.vlines("2019-02-25", 0, 1e6, color="k", linestyle="--", label="Today")
ax.set_ylim([200_000, 800_000])
plt.legend(loc="center left")
plt.show()


tf.random.set_seed(42)  # extra code – ensures reproducibility

def split_inputs_and_targets(mulvar_series, ahead=14, target_col=1):
    return mulvar_series[:, :-ahead], mulvar_series[:, -ahead:, target_col]

ahead_train_ds = tf.keras.utils.timeseries_dataset_from_array(
    mulvar_train.to_numpy(),
    targets=None,
    sequence_length=seq_length + 14,
    batch_size=32,
    shuffle=True,
    seed=42
).map(split_inputs_and_targets)
ahead_valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    mulvar_valid.to_numpy(),
    targets=None,
    sequence_length=seq_length + 14,
    batch_size=32
).map(split_inputs_and_targets)



tf.random.set_seed(42)

ahead_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=[None, 5]),
    tf.keras.layers.Dense(14)
])

fit_and_evaluate(ahead_model, ahead_train_ds, ahead_valid_ds,
                 learning_rate=0.02)




X = mulvar_valid.to_numpy()[np.newaxis, :seq_length]
Y_pred = ahead_model.predict(X)

def to_windows(dataset, length):
    dataset = dataset.window(length, shift=1, drop_remainder=True)
    return dataset.flat_map(lambda window_ds: window_ds.batch(length))

my_series = tf.data.Dataset.range(7)
dataset = to_windows(to_windows(my_series, 3), 4)
list(dataset)


dataset = dataset.map( lambda S: (S[:0], S[:, 1:]) )
list(dataset)


def to_seq2seq_dataset(series, seq_length=56, ahead=14, target_col=1,
                       batch_size=32, shuffle=False, seed=None):
    ds = to_windows(tf.data.Dataset.from_tensor_slices(series), ahead + 1)
    ds = to_windows(ds, seq_length).map(
        lambda S: (S[:, 0], S[:, 1:, target_col]))
    if shuffle:
        ds = ds.shuffle(8 * batch_size, seed=seed)
    return ds.batch(batch_size)


seq2seq_train = to_seq2seq_dataset(mulvar_train, shuffle=True, seed=42)
seq2seq_valid = to_seq2seq_dataset(mulvar_valid)
    

tf.random.set_seed(42)
seq2seq_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, return_sequences=True, input_shape=[None, 5]),
    tf.keras.layers.Dense(14)
    # equivalent: tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(14))
    # also equivalent: tf.keras.layers.Conv1D(14, kernel_size=1)
])


fit_and_evaluate(seq2seq_model, seq2seq_train, seq2seq_valid,
                 learning_rate=0.1)



class LNSimpleRNNCell(tf.keras.layers.Layer):
    def __init__(self, units, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.state_size = units
        self.state_output_size = units
        self.simple_rnn_cell = tf.keras.layers.SimpleRNN( units, activation=None )
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.activation = tf.keras.activations.get(activation)
        
    def call(self, inputs, states):
        outputs, new_states = self.simple_rnn_cell(inputs, states)
        norm_outputs = self.activation(self.layer_norm(outputs))
        return norm_outputs, [norm_outputs]



custom_ln_model = tf.keras.Sequential([tf.keras.layers.RNN(LNSimpleRNNCell(32), return_sequences=True, input_shape=[None, 5]),
                                       tf.keras.layers.Dense(14)])

model = tf.keras.Sequential( [tf.keras.layers.LSTM(32, return_sequences=True, input_shape=[None, 5]), 
                              tf.keras.layers.Dense(14)] )



conv_rnn_model = tf.keras.Sequential([tf.keras.layers.Conv1D(filters=32, kernel_size=4, strides=2, activation="relu", input_shape=[None, 5]),
                                      tf.keras.layers.GRU(32, return_sequences=True),
                                      tf.keras.layers.Dense(14)])


longer_train = to_seq2seq_dataset(mulvar_train, seq_length=112, shuffle=True, seed=42)
longer_valid = to_seq2seq_dataset(mulvar_valid, seq_length=112, shuffle=True)
downsampled_train = longer_train.map( lambda X,Y: (X,Y[:,3::2]) )
downsampled_valid = longer_valid.map( lambda X,Y: (X,Y[:,3::2]) )

fit_and_evaluate(conv_rnn_model, downsampled_train, downsampled_valid,
                 learning_rate=0.1, epochs=5)

wavenet_model = tf.keras.Sequential()
wavenet_model.add( tf.keras.layers.Input( shape=[None, 5] ) )
for rate in (1,2,3,4,8) * 2:
    wavenet_model.add( tf.keras.layers.Conv1D( filters=32, kernel_size=2, padding="causal", activation="relu", dilation_rate=rate ) )

wavenet_model.add( tf.keras.layers.Conv1D(filters=14, kernel_size=1) )
    







