import asyncio
import random as rnd

import numpy as np
from db import fetch_training_data

SUPPORTED_SYMBOLS = ["A", "B"]

async def load_data(n=150):
    """Loads training data from the database, with n being
    the size of the validation and test datasets."""
    
    data = await fetch_training_data()
    data = zip(data[0].tolist(), data[1])
    data = list(data)
    data = rnd.sample(data, k=len(data))
    validation_data = data[:n]
    test_data = data[n:n*2]
    data = data[n*2:]
    
    vals, labels = zip(*validation_data)
    validation_data = (np.array(vals), labels)
    vals, labels = zip(*test_data)
    test_data = (np.array(vals), labels)
    vals, labels = zip(*data)
    data = (np.array(vals), labels)
    
    return (data, validation_data, test_data)
    
async def load_data_wrapper(n=150):
    """Loads training data from the database, with n being
    the size of the validation and test datasets."""
    tr_d, va_d, te_d = await load_data(n)
    tr_inputs = [np.reshape(x, (63, 1)) for x in tr_d[0]]
    tr_results = [vectorized_result(y) for y in tr_d[1]]
    tr_data = zip(tr_inputs, tr_results)
    va_inputs = [np.reshape(x, (63, 1)) for x in va_d[0]]
    va_data = zip(va_inputs, va_d[1])
    te_inputs = [np.reshape(x, (63, 1)) for x in te_d[0]]
    te_data = zip(te_inputs, te_d[1])
    return (list(tr_data), va_data, list(te_data))

def vectorized_result(symbol):
    e = np.zeros((len(SUPPORTED_SYMBOLS), 1))
    i = SUPPORTED_SYMBOLS.index(symbol)
    e[i] = 1.0
    return e

if __name__ == "__main__":
    asyncio.run(load_data_wrapper())
