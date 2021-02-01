# from Cython.Build import cythonize
# from numpy.distutils.misc_util import Configuration


# def cythonize_extensions(top_path, config):
#     config.ext_modules = cythonize(
#         config.ext_modules,
#         compiler_directives={'language_level': '3'})


# def configuration(parent_package='', top_path=None):
#     config = Configuration('boda', parent_package, top_path)
#     config.add_subpackage('models')
#     config.add_subpackage('utils')
#     config.add_subpackage('lib')
#     cythonize_extensions(top_path, config)

#     return config


# if __name__ == '__main__':
#     from numpy.distutils.core import setup

#     setup(**configuration(top_path='').todict())
