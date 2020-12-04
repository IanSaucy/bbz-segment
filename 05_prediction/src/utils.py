from tqdm import tqdm


def load(what, **kwargs):
    """
    A helper function to load a saved model from a file path using a custom model class
    Args:
        what (): The custom model class used to load model
        **kwargs (): Arguments passed to the custom model class

    Returns: The loaded model

    """
    loaded_models = dict()
    for modelClass, modelName in tqdm(what, desc="loading models"):
        loaded_models[modelName] = modelClass(modelName, **kwargs)
    return loaded_models


