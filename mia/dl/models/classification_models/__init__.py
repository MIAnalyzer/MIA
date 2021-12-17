# this is a clone from https://github.com/qubvel/classification_models
# with minor changes
# licence MIT, https://github.com/qubvel/classification_models/blob/master/LICENSE

import dl.models.keras_applications as ka

def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', ka._KERAS_BACKEND)
    layers = kwargs.get('layers', ka._KERAS_LAYERS)
    models = kwargs.get('models', ka._KERAS_MODELS)
    utils = kwargs.get('utils', ka._KERAS_UTILS)
    return backend, layers, models, utils