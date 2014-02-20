import argparse
import csv
import re
import sys

from classifiers import Classifier, BinaryClassifier, Tester

def document_stream(path, skip_first=True):
    """
    Open a CSV file containing documents for training/classification,
    acting as an iterator over the normalized representations of
    those documents, a pair: (list of normalized tokens in document, document tag)

    Argument skip_first can be set to false if the first line of the CSV
    file does not contain the column headers.
    """
    with open(path, 'rU') as csv_file:
        doc_reader = csv.reader(csv_file, dialect=csv.excel) 
        first = True
        for row in doc_reader:
            if first:
                first = False
                continue
            tokens = re.split('[ \t\n\'".,;:()?!]+', row[0])
            tag = row[1]
            yield (map(lambda s: s.lower(), tokens), tag)

def load_stop_words(path):
    with open(path, 'r') as the_file:
        return [line.strip().lower() for line in the_file]

def main():
    arg_parser = argparse.ArgumentParser(description='Train and test a Naive Bayes classifier', prog='python nb.py')
    arg_parser.add_argument('-s', help='Exclude stop words', action='store_true')
    arg_parser.add_argument('-b', help='Use binary version of Naive Bayes algorithm', action='store_true')
    arg_parser.add_argument('-t', '--train', help='The training set to use (1 or 2)', default=1, choices=(1,2), type=int)

    args = arg_parser.parse_args()

    training_set = args.train
    exclude_stop_words = args.s
    binary_version = args.b

    print 'Training set used:   ' + str(training_set)
    print 'Stop words excluded: ' + ('Yes' if exclude_stop_words else 'No')
    print 'Binary version:      ' + ('Yes' if binary_version else 'No')

    if exclude_stop_words:
        stop_words = load_stop_words('corpora/english.stop') 
    else:
        stop_words = []

    training_documents = document_stream('corpora/training' + str(training_set) + '.csv')
    testing_documents  = document_stream('corpora/testing1.csv')

    if binary_version:
        classifier = BinaryClassifier(training_documents, stop_words)
    else:
        classifier = Classifier(training_documents, stop_words)

    tester = Tester(classifier, positive_class='cancer', negative_class='nocancer')
    tester.test(testing_documents)

    print '       TP:', tester.true_positives
    print '       FP:', tester.false_positives
    print '       TN:', tester.true_negatives
    print '       FN:', tester.false_negatives
    print 'Precision:', tester.precision
    print '   Recall:', tester.recall
    print '  F-Score:', tester.f_score

if __name__ == '__main__':
    main()
