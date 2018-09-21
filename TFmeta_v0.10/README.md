# TFmeta
A machine learning approach to uncover transcription factors governing metabolic reprogramming.

Yi Zhang (yi dot zhang at uky dot edu)

## Manual

### 1. Interaction Inference as a Feature Selection Problem

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

### 2. TF ranking test

```
 python TFrankTest.py \
 --network ./test/TFBindingProfiling_Glycolysis.txt \
 --interaction ./test/ranking_Glycolysis_merged.txt \
 --ranks_table ./test/TFrankingTest_Glycolysis.txt
```
