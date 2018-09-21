# Usage:
#    mpiexec -np #processes python mpi_GBT.py


from mpi4py import MPI
import os
import sys
import argparse
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def read_data(datafile, tgfile):
    X = np.genfromtxt(datafile, delimiter = ',', dtype = str)
    Y = np.genfromtxt(tgfile, delimiter = ',', dtype = str)
    return X,Y


def main(args):
    parser = argparse.ArgumentParser(description = "Gradient boosting tree regression")
    parser.add_argument("--data", dest="data", type=str, default="./595_42/upstreamGene.csv")
    parser.add_argument("--out", dest="out", type=str, default="./ranking.txt")
    parser.add_argument("--score_file", dest="score_file", type=str, default="./fit_scores.txt")
    parser.add_argument("--tg", dest="tg", type=str, default="./595_42/downstreamGene.csv")
    parser.add_argument("--n_estimators", dest="n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=0.1)
    # 'mse' for the mean squared error, which is equal to variance reduction as feature selection criterion. 'mae' for the mean absolute error.
    parser.add_argument("--criterion", dest="criterion", type=str, default="friedman_mse")
    parser.add_argument("--max_features", dest="max_features", type=int, default=None)
    parser.add_argument("--max_depth", dest="max_depth", type=int, default=3)
    parser.add_argument("--subsample", dest="subsample", type=float, default=1.0)

    opts = parser.parse_args(args[1:])
    X, Y = read_data(opts.data, opts.tg)

    regr = GradientBoostingRegressor(n_estimators=opts.n_estimators, criterion=opts.criterion, max_features=opts.max_features, max_depth=opts.max_depth, subsample=opts.subsample)
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    results = []
    fit_scores = []
    for i in range(Y.shape[1]):
        if i % size == rank:
            y_new = Y[1:,i].astype(float)
            x_new = X[:, [j for j in range(X.shape[1]) if X[0][j] != Y[0][i]]]
            regr.fit(x_new[1:,].astype(float), y_new)
            for k in range(len(x_new[0])):
                results.append([x_new[0][k],Y[0][i],regr.feature_importances_[k]])
            fit_scores.append([Y[0][i], regr.score(x_new[1:,].astype(float), y_new)])
#    results = sorted(results,key=lambda l:l[2],reverse=True)
    comm.Barrier()
    if rank != 0:
        comm.send(results, dest=0, tag=1)
        comm.send(fit_scores, dest=0, tag=2)
    else:
        for i in range(size - 1):
            temp = comm.recv(source=MPI.ANY_SOURCE, tag=1)
            results += temp
            fit_scores += comm.recv(source=MPI.ANY_SOURCE, tag=2)
        results = sorted(results,key=lambda l:l[2], reverse=True)
    comm.Barrier()
    if (rank == 0):
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
    comm.Barrier()
    MPI.Finalize()

# Main
if __name__ == '__main__':
    main(sys.argv)
