import pickle

def save_pickle(object_to_save, directory):
    with open(directory, 'wb') as f:
        pickle.dump(object_to_save, f)

    return 0

def load_pickle(directory):
    with open(directory, 'rb') as f:
        loaded_content = pickle.load(f)

    return loaded_content