The embeddings have been mapped using the VecMap tool: https://github.com/artetxem/vecmap

The target words have been compared with cosine similarity using eval_similarity.py script from VecMap. 
The similarities were printed each on separate line creating the file data/vecmap/results-lang-type.txt where lang is the language code and type is i for identical words mapping and unsup for the unsupervised one.

Next we computed the thresholds for the binary task and prepared the results in the standard format: PrepareResults.java
