
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Mathias Mueller / mmueller@cl.uzh.ch

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.dummy import DummyClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from collections import defaultdict

import logging
import argparse
import random
import codecs
import sys
import csv


random.seed(42)


class Trainer(object):
    """
    Reads training data and trains a classifier.
    """

    def __init__(self, model="model.pkl", data=None, verbose=False):
        """
        """
        self._model = model
        self._data = data
        self._verbose = verbose

        # outcomes
        self.classes = []
        self.num_classes = 0
        self.train_X = None
        self.train_y = None
        self.vectorizer = None
        self.classifier = None
        self.pipeline = None

    def train(self):
        """
        Preprocesses data, fits a model, and finally saves the model to a file.
        """
        self._preprocess()
        self._build_pipeline()
        self._fit()

    def _preprocess(self):
        """
        Reads lines from the raw training data.
        """
        
        # Dictionnary, der für jeden key, auch wenn nicht instanziiert, einen Standardvalue hat (in diesem Fall eine Leere Liste)
        # -> sonst müsste man das mit ifschleife lösen
        
        d = defaultdict(list)

        # checken ob datei vorhanden, sonst error
        if self._data:
            data = codecs.open(self._data, "r", "UTF-8")
        else:
            logging.warning("--data not found, assuming input from STDIN")
            data = sys.stdin

        # csv Datei wird durch dict reader in dictionnary verwandelt    
        reader = csv.DictReader(data, delimiter=",", quotechar='"')

        # Dictionnary mit Name (label) als key, einzelne Texte als listenelemente
        for row in reader:
            X, y = row['Text'], row['Label']
            d[y].append(X)

        # Infos für user    
        logging.debug("Examples per class:")
        for k, v in d.items():
            logging.debug("%s %d" % (k, len(v)))
        logging.debug("Total training examples: %d\n" %
                      sum([len(v) for v in d.values()]))

        # Klassenattribute bestimmen: sortierte Philosophen und die Anzahl Klassen
        self.classes = d.keys()
        self.classes = sorted(self.classes)
        self.num_classes = len(self.classes)

        # Von jedem Philosophen das erste Sample als logging ausgeben
        # Liste von Tupeln (Text, Philosophe) erstellen
        l = []
        logging.debug("Samples from the data:")
        for k, values in d.items():
            logging.debug("%s\t%s" % (values[0], k))
            for value in values:
                l.append( (value, k) )

        # shuffle, just to be sure
        # Trennt Tuple auf in train_x = Texte und train_y = Philosophen, auch beides Tupel
        random.shuffle(l)
        self.train_X, self.train_y = zip(*l)

    def _build_pipeline(self):
        """
        Builds a sklearn Pipeline. The pipeline consists of a kind of
        vectorizer, followed by a kind of classifier.
        """
        stop = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
        #https://gist.github.com/sebleier/554280
        self.vectorizer = TfidfVectorizer(stop_words=stop) ###
        #self.classifier = GaussianNB(stop_words = GaussianNB.get_stop_words()) ###
        self.classifier = KNeighborsClassifier(100) #k mitgeben, z.b. 100

        self.pipeline = Pipeline([
            ("vectorizer", self.vectorizer),
            ("clf", self.classifier)
        ])

        logging.debug(self.vectorizer)
        logging.debug(self.classifier)
        logging.debug(self.pipeline)

    def _fit(self):
        """
        Fits a model for the preprocessed data.
        """
        self.pipeline.fit(self.train_X, self.train_y)

    def save(self):
        """
        Saves the whole pipeline to a pickled file.
        """
        from sklearn.externals import joblib
        joblib.dump(self.pipeline, self._model)
        logging.debug("Classifier saved to '%s'" % self._model)


class Predictor(object):
    """
    Predicts the label of text, given a trained model.
    """

    def __init__(self, model="model.pkl"):
        """
        """
        self._model = model
        self._load()

    def _load(self):
        """
        Loads a model that was previously trained and saved.
        """
        from sklearn.externals import joblib
        self.pipeline = joblib.load(self._model)
        logging.debug("Loading model pipeline from '%s'" % self._model)

    def predict(self, samples, label_only=False):
        """
        Predicts the class label of new text samples.
        """
        predictions = []

        reader = csv.DictReader(samples, delimiter=",", quotechar='"')

        for row in reader:
            string = row["Text"]
            if label_only:
                predictions.append(self.pipeline.predict([string])[0])
            else:
                predictions.append((string, self.pipeline.predict([string])[0]))

        return predictions

    def evaluate(self, samples):
        """
        Evaluates the classifier with gold labelled data.
        """
        test_y = []
        test_X = []

        reader = csv.DictReader(samples, delimiter=",", quotechar='"')

        for row in reader:
            test_y.append(row['Label'])
            test_X.append(row['Text'])

        logging.debug("Number of gold samples found: %d" % len(test_y))

        predictions = self.predict(test_X, label_only=True)
        logging.info(metrics.classification_report(test_y, predictions, target_names=None))


def parse_cmd():
    parser = argparse.ArgumentParser(
        description="train a classifier for text data and use it for predictions")

    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="if --train, then save model to this path. If --predict, use saved model at this path."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        required=False,
        help="write verbose output to STDERR (default: False)"
    )

    mode_options = parser.add_mutually_exclusive_group(required=True)
    mode_options.add_argument(
        "--train",
        action="store_true",
        required=False,
        help="train a new model and save to the path -m/--model"
    )
    mode_options.add_argument(
        "--predict",
        action="store_true",
        required=False,
        help="predict classes of new samples, write predicted classes to STDOUT"
    )
    mode_options.add_argument(
        "--evaluate",
        action="store_true",
        required=False,
        help="evaluate trained model, write report to STDOUT. If --evaluate, data in --samples is assumed to include the gold label"
    )

    train_options = parser.add_argument_group("training parameters")

    train_options.add_argument(
        "--data",
        type=str,
        required=False,
        help="path to file with text data as CSV, UTF-8. If --data is not given, input from STDIN is assumed"
    )

    predict_options = parser.add_argument_group("prediction parameters")

    predict_options.add_argument(
        "--samples",
        type=str,
        required=False,
        help="Path to file containing samples for which a class should be predicted. If --samples is not given, input from STDIN is assumed"
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_cmd()

    # set up logging
    if args.verbose:
        level = logging.DEBUG
    elif args.evaluate:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    logging.debug(args)

    if args.train:
        t = Trainer(model=args.model,
                    data=args.data,
                    verbose=args.verbose,
                    )
        t.train()
        t.save()
    else:
        p = Predictor(model=args.model)
        if args.samples:
            input_ = codecs.open(args.samples, "r", "UTF-8")
        else:
            logging.debug("--samples not found, assuming input from STDIN")
            input_ = sys.stdin

        if args.evaluate:
            p.evaluate(samples=input_)
        else:
            predictions = p.predict(samples=input_, label_only=True)
            print("\"Id\",\"Prediction\"")
            writer = csv.writer(sys.stdout, delimiter=",", quotechar='"')
            for index, prediction in enumerate(predictions):
                writer.writerow([index+1, prediction])


if __name__ == '__main__':
    main()
