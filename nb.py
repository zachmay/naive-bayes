import argparse
import csv
import re
import sys

from collections import defaultdict

def document_stream(path):
    """
    Open a CSV file containing documents for training/classification,
    acting as an iterator over those documents (represented by a
    pair: (list of tokens in document, document tag)
    """
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
        self.stop_words = set(stop_words)
        if documents is not None:
            self.train(documents)

    def _initialize(self):
        self.class_token_counts = defaultdict(lambda: defaultdict(int)) # Count of occurrences of a given token in a given class.
        self.class_global_counts = defaultdict(int)                     # Count of all tokens occurring in a given class.
        self.class_counts = defaultdict(int)                            # Count of documents in a given class.
        self.vocabulary = set()                                         # The set of all tokens encountered.

        self.priors = defaultdict(float)                                         # Parameter table: prior probability for a given class
        self.conditional_probabilities = defaultdict(lambda: defaultdict(float)) # Parameter table: for tag c and token t, P(t|c)

    def train(self, documents):
        """
        Train the classifier
        """

        self._initialize()

        doc_number = 0

        for (doc, tag) in documents:
            self._count_document(tag)
            for token in doc:
                if token not in self.stop_words:
                    self._count_token(doc_number, tag, token)
            doc_number += 1

        self.document_count = doc_number
        self.classes = self.class_counts.keys()

        for c in self.classes:
            self.priors[c] = self.class_counts[c] / float(self.document_count)
            #self.conditional_probabilities[c] = {}
            for t in self.vocabulary:
                self.conditional_probabilities[c][t] = self._compute_conditional_probability(c, t)

        self.is_trained = True

    def _compute_conditional_probability(self, tag, token):
        numer = float(self.class_token_counts[tag][token] + 1) 
        denom = float(self.class_global_counts[tag] + len(self.vocabulary))
        return numer / denom

    def _count_document(self, tag):
        """
        Update the count of documents that are assigned the given class tag.
        """

        self.class_counts[tag] += 1

    def _count_token(self, doc_number, tag, token):
        """
        Update the count in the vocabulary for the given token.

        This is the non-binary implementation.
        """

        # Update count for this token for the given class tag.
        self.class_token_counts[tag][token] += 1

        # Update count of all tokens for this the given class tag.
        self.class_global_counts[tag] += 1

        self.vocabulary.add(token)

    def parameter_dump(self):
        """
        Print a (very verbose) dump of the counts and 
        computed parameters learned by the training
        documents.
        """
        if not self.is_trained:
            raise Error('Classifier not trained!')

        print "Document count:  ", self.document_count
        print "Vocabulary count:", len(self.vocabulary)

        print "Class counts:"
        for c in self.class_counts:
            print "  ", c, "=", self.class_counts[c]

        print "Class global counts:"
        for c in self.class_global_counts:
            print "  ", c, "=", self.class_global_counts[c]

        print "Class token counts:"
        for c in self.class_token_counts:
            for t in self.class_token_counts[c]:
                print "  ", c, t, "=", self.class_token_counts[c][t]

        print "Priors:"
        for c in self.classes:
            print "  ", c, "-", self.priors[c]

        print "Conditional Probabilities:"
        for c in self.classes:
            for t in self.vocabulary:
                print "  ", "P(" + str(t) + "|" + str(c) + ") =", self.conditional_probabilities[c][t]

    def classify(self, document):
        """
        Classify the given document (a stream of tokens)

        Throws Error if classifier is not yet trained.
        """
        if not self.is_trained:
            raise Error('Classifier not trained!')

class BinaryClassifier(Classifier):
    """
    Binary version classifier
    """
    def __init__(self, stop_words, documents=None):
        Classifier.__init__(self, stop_words, documents)

    def _initialize(self):
        self.document_tokens_seen = defaultdict(set)
        Classifier._initialize(self)

    def _count_token(self, doc_number, tag, token):
        """
        Update the count in the vocabulary for the given token.

        This is the binary implementation, so a token is only counted
        once per document.
        """
        self.vocabulary.add(token)

        if token not in self.document_tokens_seen[doc_number]:
            self.document_tokens_seen[doc_number].add(token)

            # Update count for this token for the given class tag.
            self.class_token_counts[tag][token] += 1

            # Count this token
            self.class_global_counts[tag] += 1

class Tester:
    """
    Harness for testing a (trained) classifier.
    """
    def __init__(self, classifier, documents=None):
        self.classifier = classifier
        if documents is not None:
            self.test(documents)

    def test(self, documents):
        pass





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

    

#if __name__ == '__main__':
#    main()
