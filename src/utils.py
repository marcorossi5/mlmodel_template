import random
import numpy as np


def set_seed() -> float:
    seed = random.randint(0, 100)
    np.random.seed(seed)
    random.seed()
    return seed
