import pickle

def save_pickle(object_to_save, directory):
    """
    Save object_to_save as a pickle to directory

    Args:
        object_to_save (obj): any object or variable from Python to be saved
        directory (str): location to save the pickled object

    Returns:
        0 (int): value indicating successful run of the procedure
    """
    with open(directory, 'wb') as f:
        pickle.dump(object_to_save, f)

    return 0


def load_pickle(directory):
    """
    Load a pickle from directory

    Args:
        directory (str): location where the pickle is

    Returns:
        loaded_content (obj): the variable that stores the contents loaded
        from the pickle
    """
    with open(directory, 'rb') as f:
        loaded_content = pickle.load(f)

    return loaded_content
