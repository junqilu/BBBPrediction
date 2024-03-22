# BBBPrediction
## Datasets
Data source: Meng, F., Xi, Y., Huang, J. & Ayers, P. W. A curated diverse molecular database of blood-brain barrier permeability with chemical descriptors. Sci Data 8, 289 (2021). https://doi.org/10.1038/s41597-021-01069-5
* GitHub page: https://github.com/theochem/B3DB
* All .tsv were converted to .csv

\
2 datasets

|      Dataset files      |  Number of chemicals  | Number of BBB+ chemicals | Number of BBB- chemicals |  Number of descriptors  |
|:-----------------------:|:---------------------:|:------------------------:|:------------------------:|:-----------------------:|
|   BBB_regression.csv    |         1058          |           930            |           128            |          1623           |
| BBB_classification.csv  |         7807          |           4956           |           2851           |          1625           |

* In BBB_regression.csv, `logBB <= -1.01` is considered as BBB- and `logBB >= -1` is considered as BBB+






