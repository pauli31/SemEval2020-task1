# SemEval2020 Task1
Repository for SemEval 2020 Task 1: Unsupervised Lexical Semantic Change Detection for UWB Team

If you use this software for academic research, [please cite the paper](#publication)

Requirements:
--------
- Python 3
- NumPy
- SciPy
- gensim

The data for the shared task can be obtained here:

https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/

and must be copied into `cca/data/test-data/*`

In order to reproduce results stated in the paper the  [Word Embeddings](https://drive.google.com/drive/folders/1LQ1_Lp-rDAlFy9PpNM4tTnEKFFoY_Ztb?usp=sharing) must be downloaded and unziped into `cca/data/embedding_export/*`

Usage:
--------
The **cca** method can be run from file 
`cca/main.py` the parameters of the method must be changed directly in source code in `cca/compare.py`

Publication:
--------

If you use this software for academic research, please cite the following paper

```
@inproceedings{uwb-semeval-2020,
  title={UWB at SemEval-2020 Task 1: Lexical Semantic Change Detection},
  author="Pra\v{z}\'{a}k,  Ond\v{r}ej and
          P\v{r}ib\'{a}\v{n}, Pavel and
          Taylor, Stephen and
          Sido, Jakub",
  booktitle = "Proceedings of the 14th International Workshop on Semantic Evaluation ({S}em{E}val-2020)",
  year = {2020},
  month = {Sep},
  address = "Barcelona, Spain",
  publisher = "Association for Computational Linguistics"
}
```
Contact:
--------
{ondfa, pribanp, taylor, sidoj}@kiv.zcu.cz

[http://nlp.kiv.zcu.cz](http://nlp.kiv.zcu.cz)