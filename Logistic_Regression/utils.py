# utils.py

import pickle

class Indexer:
    """
    Bijection between objects and integers starting at 0. Useful for mapping
    labels, features, etc. into coordinates of a vector space.
    """
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

def save_model(model, filepath):
    """Saves the model to the specified filepath."""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    """Loads a model from the specified filepath."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
