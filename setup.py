from setuptools import setup

setup(
   name='smdt',
   version='1.0',
   description='Small molecule design toolkit',
   author='Rahul Avadhoot',
   author_email='rahulavd@uw.edu',
   packages=['smdt'],
   install_requires=['numpy','pandas','rdkit']
)