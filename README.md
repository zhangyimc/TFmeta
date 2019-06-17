# TFmeta
A machine learning approach to uncover transcription factors governing metabolic reprogramming.

Yi Zhang (yi dot zhang at uky dot edu)

***
## Table of Contents
* Introduction
* Prerequisites
***

## Introduction
Metabolic reprogramming is a hallmark of cancer. In cancer cells, transcription factors (TFs) govern metabolic reprogramming through abnormally increasing or decreasing the transcription rate of metabolic enzymes, which provides cancer cells growth advantages and concurrently leads to the altered metabolic phenotypes observed in many cancers. Consequently, targeting TFs that govern metabolic reprogramming can be highly effective for novel cancer therapeutics. **TFmeta** is a machine learning-based method to uncover TFs that govern reprogramming of cancer metabolism. Leveraging TF binding profiles inferred from genome-wide ChIP-Seq experiments and RNA-Seq data, TFmeta predicts a set of key TFs that may be the major regulators of the gene expression changes of metabolic enzymes in the altered metabolic pathways observed in many cancers.

## Prerequisites
* [Python](https://www.python.org/)
* [NumPy](http://www.numpy.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [XGBoost](https://xgboost.readthedocs.io/en/latest/)
* [SciPy](https://www.scipy.org/)
* [GCC](https://gcc.gnu.org/)

