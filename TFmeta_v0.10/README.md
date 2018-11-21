# TFmeta
A machine learning approach to uncover transcription factors governing metabolic reprogramming.

Yi Zhang (yi dot zhang at uky dot edu)

## Manual

### 1. Data preprocessing



### 2. Interaction inference as a feature selection problem

```
 python interactionLearner.py \
 --profile ./test/regulationStatusMasterTable.txt \
 --network ./test/TFBindingProfiling_Glycolysis.txt \
 --out ./test/ranking_Glycolysis.txt \
 --score_file ./test/fit_scores_Glycolysis.txt
```

```
 ./importanceMerge ./test/ranking_Glycolysis.txt ./test/ranking_Glycolysis_merged.txt
```

### 3. TF ranking test

```
 python TFrankTest.py \
 --network ./test/TFBindingProfiling_Glycolysis.txt \
 --interaction ./test/ranking_Glycolysis_merged.txt \
 --ranks_table ./test/TFrankingTest_Glycolysis.txt
```

## Citation
Zhang, Y., Zhang, X., Lane, A.N., Fan, T.W. and Liu, J., 2018, August. TFmeta: A Machine Learning Approach to Uncover Transcription Factors Governing Metabolic Reprogramming. In *Proceedings of the 2018 ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics* (pp. 351-359). ACM.

DOI: [https://doi.org/10.1145/3233547.3233580](https://doi.org/10.1145/3233547.3233580)

