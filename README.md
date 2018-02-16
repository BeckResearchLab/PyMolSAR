# small-molecule-design-toolkit
Small-Molecule-Design-Toolkit aims to provide a generalizable open-source tool 
for calculating 760 molecular descriptors and test out several different supervised learning algorithms to build the most-appropriate Quantitative Structure-Activity Relationship (QSAR) 
classification or regression model that accurately predicts the chemical properties or activities of small molecules.

**Table of contents:**
* [Requirements](https://github.com/BeckResearchLab/small-molecule-design-toolkit#requirements)
* [Installation](https://github.com/BeckResearchLab/small-molecule-design-toolkit#installation)
* [Getting Started](https://github.com/BeckResearchLab/small-molecule-design-toolkit#getting-started)
    * [Input Formats](https://github.com/BeckResearchLab/small-molecule-design-toolkit#input-formats)
    * [Data Featurization](https://github.com/BeckResearchLab/small-molecule-design-toolkit#data-featurization)
    * [Models](https://github.com/BeckResearchLab/small-molecule-design-toolkit#models)
* [Examples](https://github.com/BeckResearchLab/small-molecule-design-toolkit/notebooks)


## Requirements
* [pandas](http://pandas.pydata.org/)
* [rdkit](http://www.rdkit.org/docs/Install.html)
* [sklearn](http://github.com/scikit-learn/scikit-learn.git)
* [numpy](http://store.continuum.io/cshop/anaconda/)
* [matplotlib](http://matplotlib.org/)
* [keras](http://keras.io/)

## Installation

**Using a conda environment**
```buildoutcfg
git clone https://github.com/BeckResearchLab/small-molecule-design-toolkit.git
cd small-molecule-design-toolkit
python setup.py install                                 
```

## Getting Started

Two good tutorials to get started are [Melting Point Prediction](http://github.com/BeckResearchLab/small-molecule-design-toolkit/blob/master/notebooks/MeltingPoint.ipynb) and [Lithium Blood-Brain Barrier Penetration Classification](http://github.com/BeckResearchLab/small-molecule-design-toolkit/blob/master/notebooks/Lithium%20Blood-Brain-Barrier%20Penetration.ipynb). 
Follow along with the tutorials to see how to predict properties on molecules using machine learning.

### Input Formats

- A column containing SMILES strings.
- A column containing an experimental measurement.

### Data Featurization

Most machine learning algorithms require that input data form vectors. 
However, input data for cheminformatics and drug discovery datasets routinely come in the format of lists of molecules and associated experimental readouts. To transform lists of molecules into vectors,
we need to calculate a set of molecular descriptors using `smdt.molecular_descriptors.getAllDescriptors()`

### Models

`smdt` can build and evaluate different classification and regression models built on top of `sklearn`.
A model report is generated to facilitate the user to choose the most appropriate Quantitative Structure-Activity Relationship (QSAR) or 
Quantitative Structure-Property Relationship (QSPR) model.

