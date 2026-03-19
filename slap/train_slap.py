import asyncio

import landmark_loader as loader

from slap import Slap


async def train(sizes, n=150):
    """The first element of sizes should be 63, with the last
    element being equal to the amount of supported symbols.

    The argument n indicates the size of the validation and
    test datasets."""
    
    print("Loading training data...")
    training_data, validation_data, test_data = await loader.load_data_wrapper(n)
    print("Loaded!\n")

    print("Initializing Fish-SLAP...")
    if sizes[0] != 63:
        print("Invalid size of input layer.")
    if sizes[-1] != len(loader.SUPPORTED_SYMBOLS):
        print("Invalid size of output layer.")
    
    net = Slap(sizes)

    net.sgd(training_data, 30, 10, 3.0, test_data=test_data)
    print("Training finished!")

if __name__ == "__main__":
    sizes = [63, 20, 2]
    
    asyncio.run(train(sizes=sizes))
