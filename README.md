# TFmeta
A machine learning approach to uncover transcription factors governing metabolic reprogramming.

Yi Zhang (yi dot zhang at uky dot edu)

***
## Table of Contents
* [Introduction](https://github.com/zhangyimc/TFmeta/blob/master/README.md#introduction)
* [Prerequisites](https://github.com/zhangyimc/TFmeta/blob/master/README.md#prerequisites)
* [Usage](https://github.com/zhangyimc/TFmeta/blob/master/README.md#usage)
* [Citation](https://github.com/zhangyimc/TFmeta/blob/master/README.md#citation)
***

## Introduction
Metabolic reprogramming is a hallmark of cancer. In cancer cells, transcription factors (TFs) govern metabolic reprogramming through abnormally increasing or decreasing the transcription rate of metabolic enzymes, which provides cancer cells growth advantages and concurrently leads to the altered metabolic phenotypes observed in many cancers. Consequently, targeting TFs that govern metabolic reprogramming can be highly effective for novel cancer therapeutics. In this work, we present TFmeta, a machine learning approach to uncover TFs that govern reprogramming of cancer metabolism. Our approach achieves state-of-the-art performance in reconstructing interactions between TFs and their target genes on public benchmark data sets. Leveraging TF binding profiles inferred from genome-wide ChIP-Seq experiments and 150 RNA-Seq samples from 75 paired cancerous (CA) and non-cancerous (NC) human lung tissues, our approach predicted 19 key TFs that may be the major regulators of the gene expression changes of metabolic enzymes of the central metabolic pathway glycolysis, which may underlie the dysregulation of glycolysis in non-small-cell lung cancer patients.

## Prerequisites
* [Python](https://www.python.org/)
* [NumPy](http://www.numpy.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [XGBoost](https://xgboost.readthedocs.io/en/latest/)
* [SciPy](https://www.scipy.org/)

## Usage

## Citation
Zhang, Y., Zhang, X., Lane, A.N., Fan, T.W. and Liu, J., 2018, August. TFmeta: A Machine Learning Approach to Uncover Transcription Factors Governing Metabolic Reprogramming. In *Proceedings of the 2018 ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics* (pp. 351-359). ACM.

DOI: [https://doi.org/10.1145/3233547.3233580](https://doi.org/10.1145/3233547.3233580)
