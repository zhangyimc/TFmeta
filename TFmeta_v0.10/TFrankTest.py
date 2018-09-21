import sys
import os
import argparse
import numpy as np
from scipy.stats import rankdata # assign ranks to data, dealing with ties appropriately
from scipy.stats import wilcoxon # calculate the Wilcoxon signed-rank test


def main(args):
	parser = argparse.ArgumentParser(description = "Merge separate TF rankings for TF discovery")
	parser.add_argument("--network", dest="network", type=str, default="./network.csv")   # Input: TF binding profiling
	parser.add_argument("--interaction", dest="interaction", type=str, default="./interaction.txt")   # Input: Predicted interactions 
	parser.add_argument("--ranks_table", dest="ranks_table", type=str, default="./ranks_table.txt")   # Output

	opts = parser.parse_args(args[1:])
	with open(opts.network) as ins:
		lines = [line.split() for line in ins]
	inputNet = np.asarray(lines)

	with open(opts.interaction) as ins:
		lines = [line.split() for line in ins]
	inputInteract = np.asarray(lines)

	print inputNet.shape
	print inputInteract.shape

	# Generate the MElist and TFlist
	MElist = []
	TFlist = []
	for line in inputNet:
		MElist.append(line[0])
		for item in line[1:]:
			if item not in TFlist:
				TFlist.append(item)

	# Generate the importance score mastertable
	IStable = np.zeros((len(MElist), len(TFlist)), dtype=float)
	for line in inputInteract:
		IStable[MElist.index(line[1])][TFlist.index(line[0])] = float(line[2])

	# Calculate the ranks of TFs for each enzyme
	rankTable = []
	for line in IStable:
		rankTable.append(rankdata(line, method='average'))

	# Calculate the Wilcoxon signed-rank test
	rankTestList = []
	for i in range(len(TFlist)):
		for j in range(i + 1, len(TFlist)):
			TF_i = [row[i] for row in rankTable]
			TF_j = [row[j] for row in rankTable]
			statistic, pvalue = wilcoxon(TF_i, TF_j, zero_method='wilcox', correction=False)
			rankTestList.append([TFlist[i], sum(TF_i), TFlist[j], sum(TF_j), statistic, pvalue])

	outfile = open(opts.ranks_table, 'w')

	'''
	outfile.write('MasterTable\t')
	for i in range(len(TFlist)):
		statement = TFlist[i] + '\t'
		outfile.write(statement)
	outfile.write('\n')
	for i in range(len(MElist)):
		statement = MElist[i] + '\t'
		for j in range(len(TFlist)):
			statement += rankTable[i][j].astype('str') + '\t'
		outfile.write(statement)
		outfile.write('\n')


	outfile.write('\n')
	outfile.write('\n')
	outfile.write('\n')
	outfile.write('\n')
	'''

	for i in range(len(rankTestList)):
		statement = rankTestList[i][0] + '\t' + str(rankTestList[i][1]) + '\t' + rankTestList[i][2] + '\t' + str(rankTestList[i][3]) + '\t' + str(rankTestList[i][4]) + '\t' + str(rankTestList[i][5]) + '\n'
		outfile.write(statement)

	outfile.close()
	

# Main
if __name__ == '__main__':
	main(sys.argv)

