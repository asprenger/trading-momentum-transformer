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



Best validation loss = 2.64734747526861
Best params:
hidden_layer_size = 20
dropout_rate = 0.3
max_gradient_norm = 100.0
learning_rate = 0.01
batch_size = 128
Predicting on test set...
359/359 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
performance (sliding window) = 0.5856333140229941
6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step
performance (fixed window) = 0.5900800387524858



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