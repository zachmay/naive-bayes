Zachary May
CPH 738
Naive Bayes Text Classification

Files:
------

README           This file.

classifiers.py   Contains class definitions for standard and binary classifiers
                 as well as a testing harness for running tests on a trained
                 classifier and computing stastitics for those test runs.

nb.py            The main program file.

test.py          Unit tests. (These can be ignored.)
             
Usage:
------

The program can be invoked with:

    python nb.py

This runs the default scenario: training set 1, no stop word filtering, using
the non-binary version of the algorithm.

The command-line arguments the script takes differ only slightly from the
description in the assignment to work better with Python's standard argument
parsing library. Run the program with the --help switch for specifics.

Results:
--------

The following table is an overview of the statistics for both normal and binary
classifiers, trained on both training sets, with or without stop-word
filtering. I was somewhat surprised at how poorly the binary classifier 
performed when trained with the larger corpus.

 Training Set  | Features                      | Prec | Recl | F-Score
---------------|-------------------------------|------|------|---------
 training1.csv | unigrams                      | 0.74 | 0.95 | 0.83
 training1.csv | unigrams, excl. stop words    | 0.75 | 0.96 | 0.84
 training1.csv | unigrams, binary              | 0.76 | 0.94 | 0.84
 training1.csv | unigrams, binary, excl. words | 0.74 | 0.94 | 0.83
 training2.csv | unigrams                      | 0.90 | 0.91 | 0.90
 training2.csv | unigrams, excl. stop words    | 0.90 | 0.91 | 0.90
 training2.csv | unigrams, binary              | 0.86 | 0.86 | 0.86
 training2.csv | unigrams, binary, excl. words | 0.88 | 0.86 | 0.87
