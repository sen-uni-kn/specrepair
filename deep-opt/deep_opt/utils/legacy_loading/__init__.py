import pickle
from pickle import load


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # rename module, which was renamed to fix a typo
        if module == 'deep_opt.models.differrentiably_approximatable_nn_modules':
            module = 'deep_opt.models.differentiably_approximatable_nn_modules'
        return super().find_class(module, name)
