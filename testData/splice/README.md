
Splice Training Set

See splice.names for more details. The original data set is available here: http://archive.ics.uci.edu/ml/datasets/Molecular+Biology+%28Splice-junction+Gene+Sequences%29

Translations

The original data set contains instances of sixty letters made up of A, G, T, C, D, N, S and R. The first four are definite but the remaining are ambiguous thus:

Character	Meaning
D			A or G or T
N			A or G or C or T
S			C or G
R			A or G

Therefore, I have converted each value into a fuzzy set thus:

Character	Encoding
A			1,0,0,0
G			0,1,0,0
T			0,0,1,0
C			0,0,0,1
D			0.33333,0.33333,0.33333,0
N			0.25,0.25,0.25,0.25
S			0,0.5,0,0.5
R			0.5,0.5,0,0,

The classification is a simple translation thus:

Classification	Encoding
EI				1,0,0
IE				0,1,0
N (Neither)		0,0,1

Approximately 10% of the training data has been retained as a test set.
