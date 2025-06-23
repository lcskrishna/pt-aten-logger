from distutils.core import setup
from setuptools import setup, Extension, distutils, Command, find_packages
import setuptools.command.install
import distutils.command.build
import distutils.command.clean
import os
import shutil

## extending the install command functionality.
class install(setuptools.command.install.install):
    def run(self):
        print ("INFO: Setup.py is installing")
    
        setuptools.command.install.install.run(self)

## extending the clean command functionality.
class clean(distutils.command.clean.clean):
    def run(self):
        print ("INFO: setup.py is being cleaned")
        cwd = os.getcwd()
        build_dir = os.path.join(cwd, "build")
        egg_dir = os.path.join(cwd, "pt_aten_logger.egg-info")
        if os.path.isdir(build_dir):
            shutil.rmtree(build_dir)
            shutil.rmtree(egg_dir)
            print("OK: Deleted the build directory.")

        distutils.command.clean.clean.run(self)

cmd_class = {
    "clean"   : clean,
    "install" : install,
    }

setup(
    name='pt-aten-logger',
    version='0.1',
    cmdclass=cmd_class,
    packages=['pt_aten_logger',],
    long_description=open('README.md').read(),
    )
