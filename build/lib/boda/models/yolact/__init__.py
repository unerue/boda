from .configuration_yolact import YolactConfig, yolact_pretrained_models
from .architecture_yolact import YolactPredictNeck, YolactPredictHead, YolactModel
from .loss_yolact import YolactLoss


__all__ = [
    'YolactLoss', 'YolactConfig', 'YolactPredictNeck',
    'YolactPredictHead', 'YolactModel', 'YolactLoss',
    'yolact_pretrained_models'
]

# _import_structure = {
#     'configuration_yolact': ['YolactConfig'],
#     'architecture_yolact': ['YolactPredictNeck', 'YolactPredictHead', 'YolactModel'],
#     'loss_yolact': ['YolactLoss']
# }
# import importlib
# import os
# import sys


# class _BaseLazyModule(ModuleType):
#     """
#     Module class that surfaces all objects but only performs associated imports when the objects are requested.
#     """

#     # Very heavily inspired by optuna.integration._IntegrationModule
#     # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
#     def __init__(self, name, import_structure):
#         super().__init__(name)
#         self._modules = set(import_structure.keys())
#         self._class_to_module = {}
#         for key, values in import_structure.items():
#             for value in values:
#                 self._class_to_module[value] = key
#         # Needed for autocompletion in an IDE
#         self.__all__ = list(import_structure.keys()) + sum(import_structure.values(), [])

#     # Needed for autocompletion in an IDE
#     def __dir__(self):
#         return super().__dir__() + self.__all__

#     def __getattr__(self, name: str) -> Any:
#         if name in self._modules:
#             value = self._get_module(name)
#         elif name in self._class_to_module.keys():
#             module = self._get_module(self._class_to_module[name])
#             value = getattr(module, name)
#         else:
#             raise AttributeError(f"module {self.__name__} has no attribute {name}")

#         setattr(self, name, value)
#         return value

#     def _get_module(self, module_name: str) -> ModuleType:
#         raise NotImplementedError


# class _LazyModule(_BaseLazyModule):
#     """
#     Module class that surfaces all objects but only performs associated imports when the objects are requested.
#     """

#     __file__ = globals()["__file__"]
#     __path__ = [os.path.dirname(__file__)]

#     def _get_module(self, module_name: str):
#         return importlib.import_module("." + module_name, self.__name__)

    
