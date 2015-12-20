= Splice Training Set =
See splice.names for more details. The original data set is available here: [http://archive.ics.uci.edu/ml/datasets/Molecular+Biology+%28Splice-junction+Gene+Sequences%29 http://archive.ics.uci.edu/ml/datasets/Molecular+Biology+%28Splice-junction+Gene+Sequences%29]


Translations


The original data set contains instances of sixty letters made up of A, G, T, C, D, N, S and R. The first four are definite but the remaining are ambiguous thus:



{| style="border-spacing:0;"
| style="border-top:0.002cm solid #000000;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:none;padding:0.097cm;"| Character
| style="border:0.002cm solid #000000;padding:0.097cm;"| Meaning

|-
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:none;padding:0.097cm;"| D
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:0.002cm solid #000000;padding:0.097cm;"| A or G or T

|-
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:none;padding:0.097cm;"| N
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:0.002cm solid #000000;padding:0.097cm;"| A or G or C or T

|-
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:none;padding:0.097cm;"| S
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:0.002cm solid #000000;padding:0.097cm;"| C or G

|-
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:none;padding:0.097cm;"| R
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:0.002cm solid #000000;padding:0.097cm;"| A or G

|}
Therefore, I have converted each value into a fuzzy set thus:



{| style="border-spacing:0;"
| style="border-top:0.002cm solid #000000;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:none;padding:0.097cm;"| Character
| style="border:0.002cm solid #000000;padding:0.097cm;"| Encoding

|-
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:none;padding:0.097cm;"| A
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:0.002cm solid #000000;padding:0.097cm;"| 1,0,0,0

|-
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:none;padding:0.097cm;"| G
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:0.002cm solid #000000;padding:0.097cm;"| 0,1,0,0

|-
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:none;padding:0.097cm;"| T
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:0.002cm solid #000000;padding:0.097cm;"| 0,0,1,0

|-
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:none;padding:0.097cm;"| C
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:0.002cm solid #000000;padding:0.097cm;"| 0,0,0,1

|-
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:none;padding:0.097cm;"| D
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:0.002cm solid #000000;padding:0.097cm;"| 0.33333,0.33333,0.33333,0

|-
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:none;padding:0.097cm;"| N
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:0.002cm solid #000000;padding:0.097cm;"| 0.25,0.25,0.25,0.25

|-
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:none;padding:0.097cm;"| S
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:0.002cm solid #000000;padding:0.097cm;"| 0,0.5,0,0.5

|-
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:none;padding:0.097cm;"| R
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:0.002cm solid #000000;padding:0.097cm;"| 0.5,0.5,0,0,

|}
The classification is a simple translation thus:



{| style="border-spacing:0;"
| style="border-top:0.002cm solid #000000;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:none;padding:0.097cm;"| Classification
| style="border:0.002cm solid #000000;padding:0.097cm;"| Encoding

|-
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:none;padding:0.097cm;"| EI
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:0.002cm solid #000000;padding:0.097cm;"| 1,0,0

|-
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:none;padding:0.097cm;"| IE
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:0.002cm solid #000000;padding:0.097cm;"| 0,1,0

|-
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:none;padding:0.097cm;"| N (Neither)
| style="border-top:none;border-bottom:0.002cm solid #000000;border-left:0.002cm solid #000000;border-right:0.002cm solid #000000;padding:0.097cm;"| 0,0,1

|}
Approximately 10% of the training data has been retained as a test set.
