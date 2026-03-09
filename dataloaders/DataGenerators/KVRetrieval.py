from collections import OrderedDict
from wonderwords import RandomWord
from dataloaders.BaseDataGenerator import BaseDataGenerator 
import random


class KVRetrieval(BaseDataGenerator):
    def __init__(self, num_pairs=20, seed=0, **kwargs):
        """
        Initialize the dataset.

        Args:
            num_pairs (int): Number of key-value pairs in the dictionary.
            seed (int, optional): Seed for reproducibility of randomness.
        """
        self.num_pairs = num_pairs
        self.seed = seed
        self.numbers = list(range(100))
        self.random = random.Random(seed)  # Create an isolated random instance

    def __iter__(self):
        while True:
            yield self.generate_example()
        
    def reset(self):
        raise NotImplementedError("Reset method is not implemented.")

    def generate_example(self):
        # Use the isolated random instance for randomness
        keys = RandomWord().random_words(self.num_pairs)
        values = self.random.sample(self.numbers, self.num_pairs)
        kv_pairs = OrderedDict(zip(keys, values))

        # Randomly select one key to be the answer
        selected_key = self.random.choice(keys)
        answer = kv_pairs[selected_key]

        # Create the example
        question = (
            f"Memorize the following dictionary:\n" +
            "\n".join([f"{k}:{v}" for k, v in kv_pairs.items()]) +
            f"\nThe value of the key '{selected_key}' is {answer}"
        )
        return question
