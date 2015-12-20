
Splice Training Set<br><br>

See splice.names for more details. The original data set is available here: http://archive.ics.uci.edu/ml/datasets/Molecular+Biology+%28Splice-junction+Gene+Sequences%29<br><br>

Translations<br><br>

The original data set contains instances of sixty letters made up of A, G, T, C, D, N, S and R. The first four are definite but the remaining are ambiguous thus:<br>

D -> A or G or T<br>
N -> A or G or C or T<br>
S -> C or G<br>
R -> A or G<br><br>

Therefore, I have converted each value into a fuzzy set thus:<br><br>

A -> 1,0,0,0<br>
G -> 0,1,0,0<br>
T -> 0,0,1,0<br>
C -> 0,0,0,1<br>
D -> 0.33333,0.33333,0.33333,0<br>
N -> 0.25,0.25,0.25,0.25<br>
S -> 0,0.5,0,0.5<br>
R -> 0.5,0.5,0,0<br><br>

The classification is a simple translation thus:<br><br>

EI -> 1,0,0<br>
IE -> 0,1,0<br>
N (Neither) -> 0,0,1<br><br>

Approximately 10% of the training data has been retained as a test set.<br>
