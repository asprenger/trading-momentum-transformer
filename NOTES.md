# Notes

## Links

Model saving and loading:

  * [Save and load models](https://www.tensorflow.org/tutorials/keras/save_and_load)
  * [Save, serialize, and export models](https://www.tensorflow.org/guide/keras/serialization_and_saving)

Tensorflow timeseries:

  * [Time series forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)

## Setup 

    [Install Python 3.8 on Ubuntu 23.04 via conda](https://askubuntu.com/questions/1493434/how-to-install-python3-8-on-ubuntu-23-04):

    # Install conda
    wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh  
    bash Anaconda3-2023.03-1-Linux-x86_64.sh -b 
    PATH="$HOME/anaconda3/bin:$PATH" 
    conda init bash 
    source ~/.bashrc  
    conda update -y conda 

    # Create python 3.8 environment
    conda create -n trading python=3.8
    conda activate trading

    # Install libraries
    pip install -r requirements.txt

## Create features

Create features in `data/quandl_cpd_nonelbw.csv`:

    python -m examples.create_features_yfinance


## Train

    rm -rf results
    python -m examples.run_dmn_experiment LSTM 2010 2022 2024 3




val_sharpe 1.759723108867906
22/22 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - loss: -2.9700 - sharpe: 1.7597
Epoch 53/300
6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step ep - loss: -3.0768

val_sharpe 1.4272980966468283
22/22 ━━━━━━━━━━━━━━━━━━━━ 1s 22ms/step - loss: -3.0710 - sharpe: 1.4273
Epoch 54/300
6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step ep - loss: -2.9423

val_sharpe 1.892510054990828
22/22 ━━━━━━━━━━━━━━━━━━━━ 1s 23ms/step - loss: -2.9449 - sharpe: 1.8925
Trial 50 Complete [00h 00m 30s]
sharpe: 2.3447209353849705

Best sharpe So Far: 3.0614459757640584
Total elapsed time: 00h 45m 41s
/Users/asprenger/miniforge3/lib/python3.10/site-packages/keras/src/saving/saving_lib.py:719: UserWarning: Skipping variable loading for optimizer 'adam', because it has 2 variables whereas the saved optimizer has 12 variables.
  saveable.load_own_variables(weights_store.get(inner_path))
6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 47ms/step
Best validation loss = 3.0521826882668996
Best params:
hidden_layer_size = 40
dropout_rate = 0.1
max_gradient_norm = 1.0
learning_rate = 0.01
batch_size = 128
Predicting on test set...
813/813 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step
performance (sliding window) = 0.304506655902489
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step
performance (fixed window) = 0.5589236478915969
WARN WARN WARN
Fix standard_window_size
WARN WARN WARN
WARN WARN WARN
Fix standard_window_size
WARN WARN WARN
WARN WARN WARN
Fix standard_window_size
WARN WARN WARN
WARN WARN WARN
Fix standard_window_size
WARN WARN WARN
WARN WARN WARN
Fix standard_window_size


backtest.py::run_single_window():
	model_features = ModelFeatures()
	dmn = LstmDeepMomentumNetworkModel()
	best_hp, best_model = dmn.hyperparameter_search(model_features.train, model_features.valid)


class DeepMomentumNetworkModel():
	def hyperparameter_search(self, train_data, valid_data):
        data, labels, active_flags, _, _ = ModelFeatures._unpack(train_data)
        val_data, val_labels, val_flags, _, val_time = ModelFeatures._unpack(valid_data)


        if self.evaluate_diversified_val_sharpe:
            callbacks = [
                SharpeValidationLoss(...),
                tf.keras.callbacks.TerminateOnNaN(),
            ]
            self.tuner.search(...)
        else:

            callbacks = [
                tf.keras.callbacks.EarlyStopping(...),
                tf.keras.callbacks.TerminateOnNaN(),
            ]
            self.tuner.search(...)


class LstmDeepMomentumNetworkModel(DeepMomentumNetworkModel):	
	def model_builder(self, hp):
		...
        model = keras.Model(inputs=input, outputs=output)
        adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
        sharpe_loss = SharpeLoss(self.output_size).call # !!!
        model.compile(
            loss=sharpe_loss,
            optimizer=adam,
        )
		return model


deep_momentum_network.py implements two losses: SharpeLoss and SharpeValidationLoss.
SharpeLoss is a Keras loss function that is used to calculate the train loss. SharpeValidationLoss 
is a Keras callback that is used to calculate the validation loss during training. This guides also 
the hyperparameter search. SharpeValidationLoss also tracks the best model and implements early stopping.




----------------
Change point detection

gpflow==2.9.2
tensorflow==2.18.0
tensorflow-probability==0.25.0
keras==3.6.0
keras-tuner==1.4.7

python -m examples.cpd_quandl SPY /tmp/foo.csv 2010-01-01 2023-01-01 21
