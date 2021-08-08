# SemEval2020 Task1 -- CompareWords Framework
Original repository for SemEval 2020 Task 1: Unsupervised Lexical Semantic Change Detection for UWB Team

If you use this software for academic research, [please cite the paper](#publication)

See the paper for detailed description of our approach.

The code is utilized for English, German, Swedish and Latin but it can be easily used for other languages.

the core function that perform the transformation with Canonical Correlation Analysis is in file `cca/compare.py` in function
`compare(...)`. Use this function for other data and languages. See documentation in the code.

The function requires already pre-trained embeddings. The embeddings can be trained using any method.
You can use the file `cca/embeddings/embeddings_generator.py` for training word embeddings with w2v or fasttext method. 

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

For orthogonal transformation please see folder **vecmap**

Publication:
--------

If you use this software for academic research, please cite the following papers

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

@inproceedings{DBLP:conf/evalita/PrazakP020,
  author    = {Ondrej Praz{\'{a}}k and
               Pavel Prib{\'{a}}n and
               Stephen Taylor},
  editor    = {Valerio Basile and
               Danilo Croce and
               Maria Di Maro and
               Lucia C. Passaro},
  title     = {{UWB} @ DIACR-Ita: Lexical Semantic Change Detection with {CCA} and
               Orthogonal Transformation},
  booktitle = {Proceedings of the Seventh Evaluation Campaign of Natural Language
               Processing and Speech Tools for Italian. Final Workshop {(EVALITA}
               2020), Online event, December 17th, 2020},
  series    = {{CEUR} Workshop Proceedings},
  volume    = {2765},
  publisher = {CEUR-WS.org},
  year      = {2020},
  url       = {http://ceur-ws.org/Vol-2765/paper110.pdf},
  timestamp = {Wed, 16 Dec 2020 16:53:24 +0100},
  biburl    = {https://dblp.org/rec/conf/evalita/PrazakP020.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}


```

License:
--------
This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International. Licence
details: https://creativecommons.org/licenses/by-nc/4.0/.

i.e. for non-comercial usage only.

Contact:
--------
{ondfa, pribanp, taylor, sidoj}@kiv.zcu.cz

[http://nlp.kiv.zcu.cz](http://nlp.kiv.zcu.cz)
