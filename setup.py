from distutils.core import setup

from setuptools import find_packages

from dmi.sst.mw_oe.mw_oe_sst_processor import MwOeSstProcessor

setup(name='dmi-oe-sst', version=MwOeSstProcessor._version,
      description='SST retrieval from microwave sensors using optimal estimation techniques developed at DMI (Danish Meteorological Institute)',
      author='Tom Block', author_email='tom.block@brockmann-consult.de', url='https://github.com/bcdev/dmi-oe-sst',
      packages=find_packages(), install_requires=['numpy', 'xarray', 'netcdf4'])
