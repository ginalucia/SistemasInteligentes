#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
using namespace std;

typedef string LABEL; //Output
typedef string ATTRIBUTE;
typedef string VALUE; //different values of attributes

//File descriptors

ifstream metadata("car-metadata.data");
ifstream training("car.data");
ifstream test("car-prueba.data");

ofstream matrix_output("output-matrix.txt"); //frequency matrix 
ofstream test_output("output-test.txt");//test cases

//Labels, class
int n_labels;
map<LABEL, int> labels;
map<int,LABEL> rlabels;
double * label_frequency;

//Attributes
int n_attributes;
map<ATTRIBUTE, int> attributes;

//Values
vector<int> n_values;
vector<map<VALUE, int> > values;

//Matrix
vector<double**> probability_matrix; // one for each attribute

//Data 
int N;
VALUE * input_data; // buffers
LABEL cat; // class



int I,J,K;

void Init() {
	
	// Labels
	LABEL aux;
	metadata >> n_labels;
	label_frequency = new double[n_labels];
	N = n_labels*n_labels;
	
	for (I = 0; I < n_labels; ++I) {
		metadata >> aux;
		labels.insert(pair<LABEL, int>(aux, I));
		rlabels.insert(pair<int,LABEL>(I,aux));
		label_frequency[I] = n_labels;
	}
	
	//Attributes & Values
	ATTRIBUTE aux2;
	VALUE aux3;
	metadata >> n_attributes;
	n_values.resize(n_attributes);
	values.resize(n_attributes);
	for (I = 0; I < n_attributes; ++I) {
		metadata >> aux2;
		attributes.insert(pair<ATTRIBUTE, int>(aux2, I));
		
		metadata >> n_values[I];
		for (J = 0; J < n_values[I]; ++J) {
			metadata >> aux3;
			values[I].insert(pair<VALUE, int>(aux3, J));
		}
	}
	input_data = new VALUE[n_attributes];
	//Matrix
	probability_matrix.resize(n_attributes);
	for (I = 0; I < n_attributes; ++I) {
		probability_matrix[I] = new double* [n_labels];
		for (J = 0; J < n_labels; ++J) {
			probability_matrix[I][J] = new double[n_values[I]];
			for (K = 0; K < n_values[I]; ++K) probability_matrix[I][J][K] = 1; // Laplace Correction. Avoid zeros.
		}
	}
	
}

void ReadTrainingSet() {
	while (!training.eof()) {
		for (I = 0; I < n_attributes; ++I)
			training >> input_data[I];
		training >> cat;
		++N;
		++label_frequency[labels[cat]];

		for (I = 0; I < n_attributes; ++I) {
			++probability_matrix[I][labels[cat]][values[I][input_data[I]]];
		}
	}
}

void PrintState() {

	auto L_I = labels.begin();
	for (I = 0; I < n_labels; ++I, ++L_I) {
		matrix_output << L_I->first << "\t" << label_frequency[L_I->second] << endl;
	}
	matrix_output << endl;
	//Print a Matrix for each attribute
	
	for (auto A_I = attributes.begin(); A_I != attributes.end(); ++A_I) {
		matrix_output << "Atribute "<<A_I->first <<" (class x value)" <<endl;
		for (J = 0; J < n_labels;++J) {
			for (K = 0; K < n_values[A_I->second]; ++K) {
				matrix_output << probability_matrix[A_I->second][J][K] << "\t";
			}
			matrix_output << endl;
		}
		matrix_output << endl;
	}

}

void FrequencyCalculator() {
	for (I = 0; I < n_attributes; ++I) {
		for (auto L_I = labels.begin(); L_I != labels.end(); ++L_I) {
			for (K = 0; K < n_values[I]; ++K)
				probability_matrix[I][L_I->second][K] /= label_frequency[L_I->second];
		}
	}
	for (I = 0; I < n_labels; ++I)
		label_frequency[I] /= N;

}

int main() {
	///////////training phase///////////
	Init();
	ReadTrainingSet();
	//PrintState();
	FrequencyCalculator();
	PrintState();
	////////////////////////////////////////
	
	
	/////////////////////DATA TEST//////////////////////////////////////
	double * P = new double[n_labels] ;
	int max = 0;

	while (!test.eof()) {
		for (I = 0; I < n_attributes; ++I)	test>> input_data[I];
		
		for (I = 0; I < n_labels; ++I) {
			P[I] = label_frequency[I];
			for (J = 0; J < n_attributes; ++J) {
				P[I] *= probability_matrix[J][I][values[J][input_data[J]]];
			}
			//test_output << P[I] << "\t";
		}
		for (I = 1; I < n_labels; ++I)
			if (P[max] < P[I]) max = I;
		
		test_output << rlabels[max] << endl;
		
		max = 0;
	}
	return EXIT_SUCCESS;
}