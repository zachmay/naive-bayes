import unittest
import math
from classifiers import Classifier, BinaryClassifier, Tester

class NaiveBayesTest(unittest.TestCase):
    def setUp(self):
        self.training_documents = [ ( ['Chinese', 'Chinese', 'Beijing'], 'c' )
                                  , ( ['chinese', 'Chinese', 'Shanghai'], 'c' )
                                  , ( ['Chinese', 'Macao'], 'c' )
                                  , ( ['Tokyo', 'Japan', 'Chinese'], 'j' ) ]

        self.testing_documents = [ ( ['Japan', 'Tokyo', 'Japan', 'Yamamoto'],             'j', -8.2048541, -5.8985266 )
                                 , ( ['Japan', 'Tokyo', 'Japan'],                         'j', -8.2048541, -5.8985266 )
                                 , ( ['Chinese', 'Chinese', 'Chinese', 'Tokyo', 'Japan'], 'c', -8.1076903, -8.9066813 )
                                 ]

        self.cls = Classifier(self.training_documents)

    def test_trained_vocabulary_is_normalized_and_unique(self):
        self.assertEquals(self.cls.vocabulary, set(['chinese', 'beijing', 'shanghai', 'macao', 'tokyo', 'japan']))

    def test_trained_probabilities(self):
        # Prior probabilities
        self.assertAlmostEquals(self.cls.priors['c'], 3.0 / 4.0)
        self.assertAlmostEquals(self.cls.priors['j'], 1.0 / 4.0)

        # Conditional probabilities:
        self.assertAlmostEquals(self.cls.conditional_probabilities['c']['chinese'], 6.0 / 14.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['c']['tokyo'],   1.0 / 14.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['c']['japan'],   1.0 / 14.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['c']['beijing'], 2.0 / 14.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['c']['macao'],   2.0 / 14.0)

        self.assertAlmostEquals(self.cls.conditional_probabilities['j']['chinese'], 2.0 / 9.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['j']['tokyo'],   2.0 / 9.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['j']['japan'],   2.0 / 9.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['j']['beijing'], 1.0 / 9.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['j']['macao'],   1.0 / 9.0)

    def test_classification_results(self):
        for (doc, expected, c_est, j_est) in self.testing_documents:
            actual = self.cls.classify(doc)
            self.assertEquals(expected, actual)

    def test_classification_estimates(self):
        for (doc, expected, c_est, j_est) in self.testing_documents:
            estimates = self.cls.classify(doc, return_all=True)
            self.assertAlmostEquals(estimates['c'], c_est)
            self.assertAlmostEquals(estimates['j'], j_est)

    def test_training_checks(self):
        untrained = Classifier()
        self.assertRaises(Exception, untrained.classify, [])
        self.assertRaises(Exception, untrained.parameter_dump)

    def test_parameter_dump_works(self):
        self.cls.parameter_dump(True)

class BinaryNaiveBayesTest(unittest.TestCase):
    def setUp(self):
        self.training_documents = [ ( ['Chinese', 'Chinese', 'Beijing'], 'c' )
                                  , ( ['chinese', 'Chinese', 'Shanghai'], 'c' )
                                  , ( ['Chinese', 'Macao'], 'c' )
                                  , ( ['Tokyo', 'Japan', 'Chinese'], 'j' ) ]

        self.testing_documents = [ ( ['Japan', 'Tokyo', 'Japan', 'Yamamoto'],             'j', -5.2574954, -4.3944492 )
                                 , ( ['Japan', 'Tokyo', 'Japan'],                         'j', -5.2574954, -4.3944492 )
                                 , ( ['Chinese', 'Chinese', 'Chinese', 'Tokyo', 'Japan'], 'j', -6.3561077, -5.8985266 )
                                 ]

        self.cls = BinaryClassifier(self.training_documents)

    def test_trained_vocabulary_is_normalized_and_unique(self):
        self.assertEquals(self.cls.vocabulary, set(['chinese', 'beijing', 'shanghai', 'macao', 'tokyo', 'japan']))

    def test_trained_probabilities(self):
        # Prior probabilities
        self.assertAlmostEquals(self.cls.priors['c'], 3.0 / 4.0)
        self.assertAlmostEquals(self.cls.priors['j'], 1.0 / 4.0)

        # Conditional probabilities:
        self.assertAlmostEquals(self.cls.conditional_probabilities['c']['chinese'], 4.0 / 12.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['c']['tokyo'],   1.0 / 12.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['c']['japan'],   1.0 / 12.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['c']['beijing'], 2.0 / 12.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['c']['macao'],   2.0 / 12.0)

        self.assertAlmostEquals(self.cls.conditional_probabilities['j']['chinese'], 2.0 / 9.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['j']['tokyo'],   2.0 / 9.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['j']['japan'],   2.0 / 9.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['j']['beijing'], 1.0 / 9.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['j']['macao'],   1.0 / 9.0)

    def test_classification_results(self):
        for (doc, expected, c_est, j_est) in self.testing_documents:
            actual = self.cls.classify(doc)
            self.assertEquals(expected, actual)

    def test_classification_estimates(self):
        for (doc, expected, c_est, j_est) in self.testing_documents:
            estimates = self.cls.classify(doc, return_all=True)
            self.assertAlmostEquals(estimates['c'], c_est)
            self.assertAlmostEquals(estimates['j'], j_est)
        
class NaiveBayesStopWordsTest(unittest.TestCase):
    def setUp(self):
        self.stop_words = ['chinese', 'tokyo']
        self.training_documents = [ ( ['Chinese', 'Chinese', 'Beijing'], 'c' )
                                  , ( ['chinese', 'Chinese', 'Shanghai'], 'c' )
                                  , ( ['Chinese', 'Macao'], 'c' )
                                  , ( ['Tokyo', 'Japan', 'Chinese'], 'j' ) ]

        self.testing_documents = [ ( ['Japan', 'Tokyo', 'Japan', 'Yamamoto'],             'j', -4.1795024, -3.2188758 )
                                 , ( ['Japan', 'Tokyo', 'Japan'],                         'j', -4.1795024, -3.2188758 )
                                 , ( ['Chinese', 'Chinese', 'Chinese', 'Tokyo', 'Japan'], 'c', -2.2335922, -2.3025851 )
                                 ]

        self.cls = Classifier(self.training_documents, self.stop_words)

    def test_trained_vocabulary_is_normalized_and_unique(self):
        self.assertEquals(self.cls.vocabulary, set(['beijing', 'shanghai', 'macao', 'japan']))

    def test_trained_probabilities(self):
        # Prior probabilities
        self.assertAlmostEquals(self.cls.priors['c'], 3.0 / 4.0)
        self.assertAlmostEquals(self.cls.priors['j'], 1.0 / 4.0)

        # Conditional probabilities:
        self.assertAlmostEquals(self.cls.conditional_probabilities['c']['chinese'],  0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['c']['tokyo'],    0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['c']['japan'],    1.0 / 7.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['c']['beijing'],  2.0 / 7.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['c']['shanghai'], 2.0 / 7.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['c']['macao'],    2.0 / 7.0)

        self.assertAlmostEquals(self.cls.conditional_probabilities['j']['chinese'],  0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['j']['tokyo'],    0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['j']['japan'],    2.0 / 5.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['j']['beijing'],  1.0 / 5.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['j']['shanghai'], 1.0 / 5.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['j']['macao'],    1.0 / 5.0)

    def test_classification_results(self):
        for (doc, expected, c_est, j_est) in self.testing_documents:
            actual = self.cls.classify(doc)
            self.assertEquals(expected, actual)

    def test_classification_estimates(self):
        for (doc, expected, c_est, j_est) in self.testing_documents:
            estimates = self.cls.classify(doc, return_all=True)
            self.assertAlmostEquals(estimates['c'], c_est)
            self.assertAlmostEquals(estimates['j'], j_est)


class BinaryNaiveBayesStopWordsTest(unittest.TestCase):
    def setUp(self):
        self.stop_words = ['chinese', 'tokyo']
        self.training_documents = [ ( ['Chinese', 'Chinese', 'Beijing', 'Beijing'], 'c' )
                                  , ( ['chinese', 'Chinese', 'Shanghai'], 'c' )
                                  , ( ['Chinese', 'Macao'], 'c' )
                                  , ( ['Tokyo', 'Japan', 'Chinese', 'Japan'], 'j' ) ]

        self.testing_documents = [ ( ['Japan', 'Tokyo', 'Japan', 'Yamamoto'],                      'c', -2.2335922, -2.3025851 )
                                 , ( ['Japan', 'Tokyo', 'Japan'],                                  'c', -2.2335922, -2.3025851 )
                                 , ( ['Chinese', 'Chinese', 'Chinese', 'Tokyo', 'Japan', 'Japan'], 'c', -2.2335922, -2.3025851 )
                                 ]

        self.cls = BinaryClassifier(self.training_documents, self.stop_words)

    def test_trained_vocabulary_is_normalized_and_unique(self):
        self.assertEquals(self.cls.vocabulary, set(['beijing', 'shanghai', 'macao', 'japan']))

    def test_trained_probabilities(self):
        # Prior probabilities
        self.assertAlmostEquals(self.cls.priors['c'], 3.0 / 4.0)
        self.assertAlmostEquals(self.cls.priors['j'], 1.0 / 4.0)

        # Conditional probabilities:
        self.assertAlmostEquals(self.cls.conditional_probabilities['c']['chinese'],  0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['c']['tokyo'],    0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['c']['japan'],    1.0 / 7.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['c']['beijing'],  2.0 / 7.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['c']['shanghai'], 2.0 / 7.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['c']['macao'],    2.0 / 7.0)

        self.assertAlmostEquals(self.cls.conditional_probabilities['j']['chinese'],  0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['j']['tokyo'],    0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['j']['japan'],    2.0 / 5.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['j']['beijing'],  1.0 / 5.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['j']['shanghai'], 1.0 / 5.0)
        self.assertAlmostEquals(self.cls.conditional_probabilities['j']['macao'],    1.0 / 5.0)

    def test_classification_results(self):
        for (doc, expected, c_est, j_est) in self.testing_documents:
            actual = self.cls.classify(doc)
            self.assertEquals(expected, actual)

    def test_classification_estimates(self):
        for (doc, expected, c_est, j_est) in self.testing_documents:
            estimates = self.cls.classify(doc, return_all=True)
            self.assertAlmostEquals(estimates['c'], c_est)
            self.assertAlmostEquals(estimates['j'], j_est)

class HarnessTest(unittest.TestCase):
    def setUp(self):
        self.training_documents = [ ( ['Chinese', 'Chinese', 'Beijing'], 'pos' )
                                  , ( ['chinese', 'Chinese', 'Shanghai'], 'pos' )
                                  , ( ['Chinese', 'Macao'], 'pos' )
                                  , ( ['Tokyo', 'Japan', 'Chinese'], 'neg' ) ]

        self.testing_documents = [ ( ['Japan', 'Tokyo', 'Japan'],                         'neg' ) # Predict neg
                                 , ( ['Chinese', 'Chinese', 'Chinese', 'Tokyo', 'Japan'], 'pos' ) # Predict pos
                                 , ( ['Japan', 'Tokyo', 'Japan'],                         'pos' ) # Predict neg
                                 , ( ['Chinese', 'Chinese', 'Chinese', 'Tokyo', 'Japan'], 'neg' ) # Predict pos
                                 ]

        self.true_positive = [ ( ['Chinese', 'Chinese', 'Chinese', 'Tokyo', 'Japan'], 'pos' )     # Predict pos
                             , ( ['Chinese', 'Chinese', 'Chinese', 'Tokyo', 'Japan'], 'pos' )     # Predict pos
                             ]
        
        self.true_negative = [ ( ['Japan', 'Tokyo', 'Japan'], 'neg' )                             # Predict neg
                             , ( ['Chinese', 'Chinese', 'Chinese', 'Tokyo', 'Japan'], 'pos' )     # Predict pos
                             ]
        
        self.false_positive = [ ( ['Chinese', 'Chinese', 'Chinese', 'Tokyo', 'Japan'], 'neg' )    # Predict pos
                              , ( ['Chinese', 'Chinese', 'Chinese', 'Tokyo', 'Japan'], 'pos' )    # Predict pos
                              ]

        self.false_negative = [ ( ['Japan', 'Tokyo', 'Japan'], 'pos' )                            # Predict neg
                              , ( ['Chinese', 'Chinese', 'Chinese', 'Tokyo', 'Japan'], 'pos' )    # Predict pos
                              ]
        self.cls = Classifier(self.training_documents)

    def test_confusion_matrix_classification(self):
        tester = Tester(self.cls, positive_class='pos', negative_class='neg')

        # Test each of the above documents in isolation.

        tester.test(self.true_positive)
        self.assertEquals( (tester.true_positives, tester.false_positives, tester.true_negatives, tester.false_negatives, tester.count),
                           (2, 0, 0, 0, 2) )
        self.assertAlmostEquals(tester.precision, 1.0)
        self.assertAlmostEquals(tester.recall, 1.0)
        self.assertAlmostEquals(tester.f_score, 1.0)
        
        tester.test(self.true_negative)
        self.assertEquals( (tester.true_positives, tester.false_positives, tester.true_negatives, tester.false_negatives, tester.count),
                           (1, 0, 1, 0, 2) )
        self.assertAlmostEquals(tester.precision, 1.0)
        self.assertAlmostEquals(tester.recall, 1.0)
        self.assertAlmostEquals(tester.f_score, 1.0)

        tester.test(self.false_negative)
        self.assertEquals( (tester.true_positives, tester.false_positives, tester.true_negatives, tester.false_negatives, tester.count),
                           (1, 0, 0, 1, 2) )
        self.assertAlmostEquals(tester.precision, 1.0)
        self.assertAlmostEquals(tester.recall, 0.5)
        self.assertAlmostEquals(tester.f_score, 1.0 / 1.5)

        tester.test(self.false_positive)
        self.assertEquals( (tester.true_positives, tester.false_positives, tester.true_negatives, tester.false_negatives, tester.count),
                           (1, 1, 0, 0, 2) )
        self.assertAlmostEquals(tester.precision, 0.5)
        self.assertAlmostEquals(tester.recall, 1.0)
        self.assertAlmostEquals(tester.f_score, 1.0 / 1.5)

        tester.test(self.testing_documents)
        self.assertEquals( (tester.true_positives, tester.false_positives, tester.true_negatives, tester.false_negatives, tester.count),
                           (1, 1, 1, 1, 4) )
        self.assertAlmostEquals(tester.precision, 0.5)
        self.assertAlmostEquals(tester.recall, 0.5)
        self.assertAlmostEquals(tester.f_score, 0.5)


if __name__ == '__main__':
    unittest.main(buffer=True)
