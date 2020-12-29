from setuptools import find_packages
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
from distutils.command.clean import clean as Clean
from distutils.command.sdist import sdist
import os
import shutil


def configuration(parent_package='', top_path=None):
    config = Configuration(None, parent_package, top_path)

    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True)
    config.add_subpackage('seelab')

    return config


class CleanCommand(Clean):
    description = 'Remove build artifacts from the source tree'

    def run(self):
        Clean.run(self)
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, 'PKG-INFO'))
        if remove_c_files:
            print('Will remove generated .c files')
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('sklearn'):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       (".so", ".pyd", ".dll", ".pyc")):
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in ['.c', '.cpp']:
                    pyx_file = str.replace(filename, extension, '.pyx')
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))


cmdclass = {'clean': CleanCommand, 'sdist': sdist}


def setup_packages():
    metadata = dict(
        name='seelab',
        version='0.0.1',
        install_requires=['Cython', 'pytorch', 'numpy'],
        author='Kyungsu',
        author_email='unerue@me.com',
        maintainer='Kyungsu',
        maintainer_email='unerue@me.com',
        description='boda is a library for instance segmentation.',
        packages=find_packages(),
        # include_package_data=True,
        entry_points={'console_scripts': ['seelab=seelab.cli.command:main']},
        classifiers=[
            'Programming Language :: C',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8'],
        cmdclass=cmdclass,
        configuration=configuration,
        python_requires='>=3.6')

    setup(**metadata)


if __name__ == '__main__':
    setup_packages()