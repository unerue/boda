# import os
# import numpy as np
# from numpy.distutils.misc_util import Configuration


# def configuration(parent_package='', top_path=None):
#     config = Configuration('utils', parent_package, top_path)

#     # libraries = []
#     # if os.name == 'posix':
#     #     libraries.append('m')

#     config.add_extension(
#         '_cython_utils',
#         sources=['_cython_utils.pyx'],
#         include_dirs=[np.get_include()],
#         language='c++',)
#         # libraries=libraries)

#     return config


# if __name__ == '__main__':
#     from numpy.distutils.core import setup

#     setup(**configuration().todict())
