import sys
import argparse
import numpy as np
import xgboost as xgb


# One-hot encoding for regulation status
def oneHotEncoding(array):
    res = []
    for i in range(array.shape[0]):
        x = []
        for j in range(array.shape[1]):
            if array[i][j] == '2':
                x.extend([1, 0, 0])
            elif array[i][j] == '0':
                x.extend([0, 0, 1])
            else:
                x.extend([0, 1, 0])
        res.append(x)
    np_res = np.array(res)
    return np_res


def main(args):
    parser = argparse.ArgumentParser(description = "Interaction inference as a feature selection problem")
    parser.add_argument("--profile", dest="profile", type=str, default="./profile.csv")   # Input: Regulation status table
    parser.add_argument("--network", dest="network", type=str, default="./network.csv")   # Input: TF binding profiling
    parser.add_argument("--out", dest="out", type=str, default="./ranking.txt")   # Output: Separate TF ranking
    parser.add_argument("--score_file", dest="score_file", type=str, default="./fit_scores.txt")   # Output: Fitting scores
    parser.add_argument("--objective", dest="objective", type=str, default="multi:softmax")
    parser.add_argument("--num_class", dest="num_class", type=int, default=3)
    parser.add_argument("--eta", dest="eta", type=float, default=0.01)
    parser.add_argument("--max_depth", dest="max_depth", type=int, default=3)
    parser.add_argument("--min_child_weight", dest="min_child_weight", type=int, default=5)
    parser.add_argument("--num_round", dest="num_round", type=int, default=300)
    parser.add_argument("--seed", dest="seed", type=int, default=42)
    parser.add_argument("--silent", dest="silent", type=int, default=1)
    parser.add_argument("--nthread", dest="nthread", type=int, default=16)

    opts = parser.parse_args(args[1:])
    inputPro = np.genfromtxt(opts.profile, delimiter = '\t', dtype = str)
    with open(opts.network) as ins:
    	lines = [line.split() for line in ins]
    inputNet = np.asarray(lines)
    
    params = {}
    params['objective'] = opts.objective
    params['num_class'] = opts.num_class
    params['eta'] = opts.eta
    params['max_depth'] = opts.max_depth
    params['min_child_weight'] = opts.min_child_weight
    params['seed'] = opts.seed
    params['silent'] = opts.silent
    params['nthread'] = opts.nthread

    results = []
    fit_scores = []
    for i in range(inputNet.shape[0]):
        x = inputPro[:, [j for j in range(inputPro.shape[1]) for k in range(1, len(inputNet[i])) if inputPro[0][j] == inputNet[i][k]]]
        y = inputPro[:, [j for j in range(inputPro.shape[1]) if inputPro[0][j] == inputNet[i][0]]]

        # Pick altered metabolic enzymes
        x_new = x[[j for j in range(1, y.shape[0]) if y[j][0] != '1'], :]
        y_new = y[[j for j in range(1, y.shape[0]) if y[j][0] != '1'], :]
        print y[0][0]
        if x_new.shape[0] < 39:
            continue

        # Generate feature names
        xlabel = []
        for m in x[0,]:
            xlabel.extend([m + '_a', m + '_b', m + '_c'])
        dtrain = xgb.DMatrix(oneHotEncoding(x[1:,]), label=y[1:,].astype(float), feature_names=xlabel)
        clas = xgb.train(params, dtrain, num_boost_round=opts.num_round, verbose_eval=False)
        feature_importances_ = clas.get_fscore()
        for key in feature_importances_:
            results.append([key, y[0][0], float(feature_importances_[key])])
        fit_scores.append([y[0][0], float(np.sum(clas.predict(dtrain).reshape(-1,1) == y[1:,].astype(float))) / y[1:,].astype(float).shape[0]])

    outfile = open(opts.out, 'w')
    for i in range(len(results)):
        statement = results[i][0] + '\t' + results[i][1] + '\t' + str(results[i][2]) + '\n'
        outfile.write(statement)
    outfile.close()
    outfile2 = open(opts.score_file, 'w')
    for i in range(len(fit_scores)):
        statement = fit_scores[i][0] + '\t' + str(fit_scores[i][1]) + '\n'
        outfile2.write(statement)
    outfile2.close()


# Main
if __name__ == '__main__':
    main(sys.argv)