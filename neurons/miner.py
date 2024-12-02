# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 reed987

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import bittensor as bt

import crypto_ai
from crypto_ai.base.miner import BaseMinerNeuron
from enum import Enum

import os
import requests
import configparser
import joblib
import time
import torch
import typing

import keras
import sklearn

class ModelType(Enum):
    """Class containing static model type constants."""
    LR = "lr"
    LSTM = "lstm"
    GRU = "gru"
    CNN = "cnn"
    # TODO add your own models if needed

class Miner(BaseMinerNeuron):
    """
    Miner neuron class.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        
        # Load default symbols, model path and currency from config file
        self.crypto_symbols, self.model_path, self.currency, self.intervals = self.load_config()
        
        # Load model
        if self.model_path:
            self.model = self.load_model(self.model_path)  # Load the model based on its file type

            self.model_type = self.check_model_type(self.model)
        else:
            raise ValueError("Model path must be provided in the configuration file.")
        
    def load_config(self):
        # default values
        config = configparser.ConfigParser()
        config.read('miner.properties')
        
        # Load symbols, model path, and currency from DEFAULT section
        if config.has_section('DEFAULT'):
            symbols = config.get('DEFAULT', 'symbols', fallback='tao').split(',') # Default to 'tao'
            model_path = config.get('DEFAULT', 'model_path', fallback=None)
            currency = config.get('DEFAULT', 'currency', fallback='usd')  # Default to 'usd'
            intervals = config.get('DEFAULT', 'intervals', fallback='5m').split(',')  # Default to '5m'
            
            return [symbol.strip() for symbol in symbols], model_path, currency, intervals  # Strip whitespace from symbols
        
        return [], None, 'usd', '5m'  # Return empty list, None for model path, default currency and intervals

    def load_model(self, model_path):
        """Load a model from various formats based on its file extension."""
        _, ext = os.path.splitext(model_path)  # Get the file extension
        
        # Load PyTorch models (.pth or .pt)
        if ext in ['.pth', '.pt']:
            return torch.load(model_path)  # Load PyTorch models
            
        # Load Keras models (.h5)
        elif ext == '.h5':
            from tensorflow.keras.models import load_model
            return load_model(model_path)  # Load Keras models
            
        # Load joblib or pickle models (.pkl or .joblib)
        elif ext in ['.pkl', '.joblib']:
            return joblib.load(model_path)  # Load joblib or pickle models
            
        else:
            raise ValueError(f"Unsupported model format: {ext}")  # Raise error for unsupported formats

    def check_model_type(model):
        """Check the type of the loaded model."""
        
        # Check for Logistic Regression (sklearn .pkl or .joblib)
        if isinstance(model, sklearn.linear_model.LinearRegression):
            return ModelType.LR

        # Check if the model is a Keras model (.h5)
        elif isinstance(model, keras.Model):
            # Check for LSTM layers
            if any(isinstance(layer, keras.layers.LSTM) for layer in model.layers):
                return ModelType.LSTM
            # Check for GRU layers
            elif any(isinstance(layer, keras.layers.GRU) for layer in model.layers):
                return ModelType.GRU
            # Check for CNN layers
            elif any(isinstance(layer, keras.layers.Conv2D) for layer in model.layers):
                return ModelType.CNN
            else:
                raise ValueError("Unknown Keras model type.")
        
        # If the model is a PyTorch model, we can check its architecture (.pt or .pth)
        elif isinstance(model, torch.nn.Module):
            # Check for LSTM layers
            if any(isinstance(layer, torch.nn.LSTM) for layer in model.modules()):
                return ModelType.LSTM
            # Check for GRU layers
            elif any(isinstance(layer, torch.nn.GRU) for layer in model.modules()):
                return ModelType.GRU
            # Check for CNN layers
            elif any(isinstance(layer, torch.nn.Conv2d) for layer in model.modules()):
                return ModelType.CNN
            # Check for Linear Model
            elif isinstance(model, torch.nn.Linear):  # Assuming it's a simple linear model
                return ModelType.LR
            else:
                raise ValueError("Unknown PyTorch model type.")

        raise ValueError("Unknown or unsupported model type.")
    
    def get_historical_prices(self, symbols: list, currency: str):
        """
        # TODO Get/load historical prices
        """
        pass

    def generate_prediction(self, historical_prices):
        """
        Generate predictions using the loaded model.
        
        Args:
            historical_prices: Current prices of cryptocurrencies.

        Returns:
            predicted prices
        """
        if self.model_type == ModelType.LR:
            # TODO logic for Linear Regression
            pass
        elif self.model_type == ModelType.LSTM:
            # TODO logic for Long Short Term Memory
            pass
        elif self.model_type == ModelType.CNN:
            # TODO logic for Convolutional Neural Network
            pass
        elif self.model_type == ModelType.GRU:
            # TODO logic for Gated Recurrent Unit
            pass
        # TODO logic for other models
        pass

    async def forward(self, synapse: crypto_ai.protocol.Dummy) -> crypto_ai.protocol.Dummy:
        requested_symbols = synapse.request_data.get('symbols', self.crypto_symbols)
        requested_intervals = synapse.request_data.get('intervals', self.intervals)
        requested_currency = synapse.request_data.get('currency', self.currency)  # Use request currency or default

        # check if provided symbols are supported by the miner
        is_valid_symbol = all(symbol in self.crypto_symbols for symbol in requested_symbols)
        if not is_valid_symbol:
            msg = "No supported crypto symbols provided"
            synapse.dummy_output = msg
            bt.logging.warning(msg)
            return synapse

        # check if provided intervals are supported by the miner
        is_valid_interval = all(interval in self.intervals for interval in requested_intervals)
        if not is_valid_interval:
            msg = "No supported intervals provided"
            synapse.dummy_output = msg
            bt.logging.warning(msg)
            return synapse

        historical_prices = self.get_historical_prices(requested_symbols, requested_currency)

        if historical_prices:
            predicted_prices = self.generate_prediction(historical_prices)
            
            predictions_with_symbols = {
                symbol: {
                    "predicted_price": predicted_prices[symbol],
                    "currency": requested_currency  # Currency returned ie USD
                } for symbol in requested_symbols if symbol in predicted_prices
            }

            synapse.dummy_output = predictions_with_symbols
            
            return synapse
        else:
            msg = "Error fetching prices"
            synapse.dummy_output = msg
            bt.logging.warning(msg)
            return synapse
    
    async def blacklist(self, synapse: crypto_ai.protocol.Dummy) -> typing.Tuple[bool, str]:
        pass
        
    async def priority(self, synapse: crypto_ai.protocol.Dummy) -> float:
        pass


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)
