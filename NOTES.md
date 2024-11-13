# Notes

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

    python -m examples.run_dmn_experiment LSTM 2014 2020 2024 3



time_steps: 63
input_size: 8
