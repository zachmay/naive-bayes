import argparse
import csv
import re
import sys

def document_stream(path):
    with open(path, 'rU') as csv_file:
        doc_reader = csv.reader(csv_file, dialect=csv.excel) 
        for row in doc_reader:
            tokens = re.split('[ \t\n\'".,;:()?!]+', row[0])
            tag = row[1]
            yield (tokens, tag)

class Classifier:
    """
    Non-Binary Naive Bayes Classifier
    """
    def __init__(self, stop_words, documents=None):
        self.stop_words = stop_words
        if documents is not None:
            self.train(documents)

    def train(self, documents):
        """
        Train the classifier
        """

        self.is_trained = True


    def classify(self, document):
        """
        Classify the given document (a stream of tokens)

        Throws Error if classifier is not yet trained.
        """
        if not self.is_trained:
            raise Error('Classifier not trained!')

    def update_count(self, doc_number, token):
        """
        Update the count in the vocabulary for the given token.

        This is the non-binary implementation.
        """
        if (doc_number, token) not in self.vocabulary:
            self.vocabulary[(doc_number, token)] = 1
        else:
            self.vocabulary[(doc_number, token)] += 1


class BinaryClassifier(Classifier):
    """
    Binary version classifier
    """
    def __init__(self, stop_words, documents=None):
        Classifier.__init__(self, stop_words, documents)

    def update_count(self, doc_number, token):
        """
        Update the count in the vocabulary for the given token.

        This is the binary implementation, so a token is only counted
        once per document.
        """
        if (doc_number, token) not in self.vocabulary:
            self.vocabulary[(doc_number, token)] = 1

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
        stop_words = ['a', 'an', 'the']
    else:
        stop_words = []

    training_documents = document_stream('training' + str(training_set) + '.csv')

    if binary_version:
        classifier = BinaryClassifier(stop_words, training_documents)
    else:
        classifier = Classifier(stop_words, training_documents)

    

if __name__ == '__main__':
    main()
