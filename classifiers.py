from collections import defaultdict
from math import log

class Classifier:
    """
    Non-Binary Naive Bayes Classifier
    """
    def __init__(self, documents=None, stop_words=[]):
        self.stop_words = set(stop_words)
        self.is_trained = False
        if documents is not None:
            self.train(documents)

    def initialize(self):
        self.is_trained = False
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

        self.initialize()

        doc_number = 0

        for (doc, tag) in documents:
            self._count_document(tag)
            for token in doc:
                normalized_token = token.lower()
                if normalized_token not in self.stop_words:
                    self.count_token(doc_number, tag, normalized_token)
            doc_number += 1

        self.document_count = doc_number
        self.classes = self.class_counts.keys()

        for c in self.classes:
            self.priors[c] = self.class_counts[c] / float(self.document_count)
            for t in self.vocabulary:
                self.conditional_probabilities[c][t] = self.compute_conditional_probability(c, t)

        self.is_trained = True

    def compute_conditional_probability(self, tag, token):
        numer = float(self.class_token_counts[tag][token] + 1) 
        denom = float(self.class_global_counts[tag] + len(self.vocabulary))
        # print len(self.vocabulary), tag, token, self.class_token_counts[tag][token], self.class_global_counts[tag], numer, denom
        return numer / denom

    def _count_document(self, tag):
        """
        Update the count of documents that are assigned the given class tag.
        """

        self.class_counts[tag] += 1

    def count_token(self, doc_number, tag, token):
        """
        Update the count in the vocabulary for the given token.

        This is the non-binary implementation.
        """
        # Update count for this token for the given class tag.
        self.class_token_counts[tag][token] += 1

        # Update count of all tokens for this the given class tag.
        self.class_global_counts[tag] += 1

        self.vocabulary.add(token)

    def classify(self, document, return_all=False):
        """
        Classify the given document (a stream of tokens)

        Throws Exception if classifier is not yet trained.
        """
        if not self.is_trained:
            raise Exception('Classifier not trained!')

        normalized_document = self.normalize_document(document)

        scores = {}
        for c in self.classes:
            scores[c] = log(self.priors[c])
            for t in normalized_document:
                try:
                    if self.conditional_probabilities[c][t] > 0.0:
                        scores[c] += log(self.conditional_probabilities[c][t])
                except:
                    print "Caught math domain error: c = %s, t = %s, value = %f" % (c, t, self.conditional_probabilities[c][t])
                    sys.exit()

        if return_all:
            return scores
        else:
            return self._key_max(scores)

    def normalize_document(self, document):
        """
        The non-binary case doesn't manipulate the document in any way.
        """
        return map(lambda x: x.lower(), document)

    def _key_max(self, dictionary):
        best_score = None
        best_key = None
        for key in dictionary:
            score = dictionary[key]
            if best_score is None or score > best_score:
                best_score = score
                best_key = key
        return best_key

    def parameter_dump(self, verbose=False):
        """
        Print a (potentially verbose) dump of the counts and 
        computed parameters learned by the training
        documents.
        """
        if not self.is_trained:
            raise Exception('Classifier not trained!')

        print "Document count:  ", self.document_count
        print "Vocabulary count:", len(self.vocabulary)

        print "Class counts:"
        for c in self.class_counts:
            print "  ", c, "=", self.class_counts[c]

        print "Class global counts:"
        for c in self.class_global_counts:
            print "  ", c, "=", self.class_global_counts[c]

        if verbose:
            print "Class token counts:"
            for c in self.class_token_counts:
                for t in self.class_token_counts[c]:
                    print "  ", c, t, "=", self.class_token_counts[c][t]

        print "Priors:"
        for c in self.classes:
            print "  ", c, "-", self.priors[c]

        print "Conditional Probabilities:"
        for c in self.classes:
            class_sum = 0
            for t in self.vocabulary:
                class_sum += self.conditional_probabilities[c][t]
                if verbose:
                    print "  ", "P(" + str(t) + "|" + str(c) + ") =", self.conditional_probabilities[c][t]
            print "  Total for class " + str(c) + ":", class_sum



class BinaryClassifier(Classifier):
    """
    Binary version classifier
    """
    def initialize(self):
        self.document_tokens_seen = defaultdict(set)
        Classifier.initialize(self)

    def count_token(self, doc_number, tag, token):
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

    def normalize_document(self, document):
        """
        In the binary case, collapse duplicates
        """
        return list(set(map(lambda x: x.lower(), document)))

class Tester:
    """
    Harness for testing a (trained) classifier.
    """
    def __init__(self, classifier, positive_class, negative_class):
        if not classifier.is_trained:
            raise Exception("Classifier not trained.")
        self.classifier = classifier
        self.positive_class = positive_class
        self.negative_class = negative_class

    def test(self, documents):
        self.true_positives  = 0
        self.true_negatives  = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.count = 0
        for (doc, actual_class) in documents:
            self.count += 1
            predicted_class = self.classifier.classify(doc)
            if actual_class == self.positive_class and predicted_class == self.positive_class:
                self.true_positives += 1
            elif actual_class == self.positive_class and predicted_class == self.negative_class:
                self.false_negatives += 1
            elif actual_class == self.negative_class and predicted_class == self.positive_class:
                self.false_positives += 1
            elif actual_class == self.negative_class and predicted_class == self.negative_class:
                self.true_negatives += 1
            else:
                raise Exception("Encountered actual class '%s' and predicted class '%s', "
                                "but expected positive class '%s' or negative class '%s'" % 
                                    (actual_class, predicted_class, self.positive_class, self.negative_class))

        self.precision = self.true_positives / float(self.true_positives + self.false_positives)
        self.recall    = self.true_positives / float(self.true_positives + self.false_negatives)

        self.f_score = 2 * self.true_positives / float(2 * self.true_positives + self.false_positives + self.false_negatives)
