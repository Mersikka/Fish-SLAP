import asyncio
import pickle

import numpy as np
from db import fetch_training_data


async def load_data():
    data = await fetch_training_data()
    print(data)

if __name__ == "__main__":
    asyncio.run(load_data())
