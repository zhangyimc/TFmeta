#include <fstream>
#include <unordered_map>
#include <cstring>

using namespace std;

int main(int argc, char** argv){
	string input = argv[1];
	string output = argv[2];

	ifstream fin(input.c_str());
	ofstream fout(output.c_str());
	string info_TF, info_target, info_score, info_other;
	string scoreKey;
	float score;
	unordered_map<string, float> rankList;

	while(fin >> info_TF){
		fin >> info_target >> info_score;
		getline(fin, info_other);

		info_TF = info_TF.substr(0, info_TF.length() - 2);
		scoreKey = info_TF + "\t" + info_target;
		score = atof(info_score.c_str());

		if(rankList.find(scoreKey) != rankList.end())
			rankList[scoreKey] += score;
		else
			rankList[scoreKey] = score;
	}

	for(auto it = rankList.begin(); it != rankList.end(); ++it)
		fout << it->first << "\t" << it->second << endl;

	fin.close();
	fout.close();
	return 0;
}