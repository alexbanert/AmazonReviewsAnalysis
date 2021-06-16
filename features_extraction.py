import string

import nltk
from nltk.collocations import *
from nltk.corpus import stopwords


def compare_types_reviews(pos_reviews, neg_reviews):
    """
    Compare two lists of texts and find bigrams unique to each one
    :param pos_reviews: List of positive reviews
    :param neg_reviews: List of negative reviews
    :return: Pair of sets (s1, s2) where s1 is bigrams exclusive to positive reviews and s2 is exclusive to negative reviews
    """
    # Extract bigrams for both review types
    positive_bigrams = extract(pos_reviews)
    negative_bigrams = extract(neg_reviews)

    # Find bigrams which appear in both positive and negative reviews
    intersection = positive_bigrams.intersection(negative_bigrams)

    # Keep only pairs exclusive to each review type
    positive_bigrams_set = positive_bigrams.difference(intersection)
    negative_bigrams_set = negative_bigrams.difference(intersection)

    return positive_bigrams_set, negative_bigrams_set


def extract(texts):
    """
    Find most popular bigrams in given texts
    :param texts: List of texts for analysis
    :return: Set of pairs (w1, w2) of most common consecutive words
    """
    all_words = []
    table = str.maketrans('', '', string.punctuation)
    stop_words = set(stopwords.words('english'))

    # Remove punctuation from texts, convert to lowercase, remove stopwords and merge texts into one list
    for text in texts:
        text = text.translate(table)
        tokens = nltk.word_tokenize(text.lower())
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        words = [w for w in words if not w in stop_words]
        all_words += words

    # Generate set of top bigrams based on their frequency
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder2 = BigramCollocationFinder.from_words(all_words, window_size=6)
    finder2.apply_freq_filter(6)
    tuples = finder2.nbest(bigram_measures.likelihood_ratio, 40)
    tuples_set = set()
    for a, b in tuples:
        if a != b:
            tuples_set.add((a, b))

    return tuples_set
