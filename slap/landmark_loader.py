import asyncio
import pickle
import random as rnd

import numpy as np
from db import fetch_training_data


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
    
    print(data[0].shape)
    print(validation_data[0].shape)
    print(test_data[0].shape)
    return (data, validation_data, test_data)
    

if __name__ == "__main__":
    asyncio.run(load_data())
