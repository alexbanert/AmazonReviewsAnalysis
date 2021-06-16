import gzip
from collections import Counter
from datetime import datetime
from itertools import combinations

import matplotlib.pyplot as plt
from tabulate import tabulate

from features_extraction import compare_types_reviews


def parse(filename):
    """
    Reads data file and returns entries as dictionary
    :param filename: Name of the file containing data
    """
    f = gzip.open(filename, 'r')
    entry = {}
    for l in f:
        l = l.strip().decode()
        colonPos = l.find(':')
        if colonPos == -1:
            yield entry
            entry = {}
            continue
        eName = l[:colonPos]
        rest = l[colonPos + 2:]
        entry[eName] = rest
    yield entry


def reviews_year_bar_chart(year_ratings, title=''):
    """
    Generate bar chart and LaTeX table for reviews in given years
    :param year_ratings: Dictionary of {year: no_reviews}
    :param title: Title for chart if given else no title
    """
    years = []
    years_pos = []
    no_ratings = []
    rows = []
    i = 0
    for key, value in sorted(year_ratings.items()):
        years.append(key)
        no_ratings.append(value)
        years_pos.append(i)
        rows.append([key, value])
        i += 1

    plt.bar(years_pos, no_ratings, color='blue', zorder=3)

    plt.xlabel("Year")
    plt.ylabel("Number of ratings")

    plt.xticks(years_pos, years, rotation='vertical')
    plt.grid(axis='y', zorder=0)

    if title != '':
        plt.title(title)

    plt.show()
    print(tabulate(rows, headers=['Year', 'Number of ratings'], tablefmt="latex"))


def popular_products_bar_chart(product_ratings, n, name_mapping):
    """
    Generate bar chart and LaTeX table for the amount of reviews by product
    :param product_ratings: Dictionary of {product_id: no_reviews}
    :param n: Amount of top popular products to analyse
    :param name_mapping: Dictionary of {product_id: product_title}
    """

    n = min(n, len(product_ratings))
    products = []
    prod_pos = []
    no_ratings = []
    rows = []
    i = 0
    for key, value in sorted(product_ratings.items(), key=lambda item: item[1], reverse=True)[:n]:
        products.append(key)
        prod_pos.append(i)
        no_ratings.append(value)
        rows.append([value, key, name_mapping[key]])
        i += 1

    plt.bar(prod_pos, no_ratings, color='blue', zorder=3)
    plt.xlabel("Products")
    plt.ylabel("Number of ratings")
    plt.title('Number of ratings for %s most popular products' % str(n))

    plt.xticks(prod_pos, products, rotation='vertical')
    plt.grid(axis='y', zorder=0)
    plt.show()
    print(tabulate(rows, headers=['Number of ratings', 'ProductId', 'Product'], tablefmt="latex"))


def score_distribution_pie_chart(scores_dict):
    """
    Generate pie chart and LaTeX table for the amount of reviews by score
    :param scores_dict: Dictionary of {score: no_reviews}
    """
    scores = []
    scores_pos = []
    no_ratings = []
    rows = []
    i = 0
    for key, value in sorted(scores_dict.items()):
        scores.append(key)
        scores_pos.append(i)
        no_ratings.append(value)
        rows.append([key, value])
        i += 1

    plt.pie(no_ratings, labels=scores, autopct='%1.1f%%')
    plt.title('Score reviews distribution')

    plt.show()
    print(tabulate(rows, headers=['Score', 'Number of ratings'], tablefmt="latex"))


def score_type_distribution_pie_chart(no_positives, no_negatives):
    """
    Generate pie chart and LaTeX table for the amount of reviews by review type
    :param no_positives: Number of positive reviews
    :param no_negatives: Number of negative reviews
    """
    labels = ['Positive', 'Negative']
    plt.pie([no_positives, no_negatives], labels=labels, autopct='%1.1f%%')
    plt.title('Review distribution by type')
    plt.show()
    print(tabulate([labels, [no_positives, no_negatives]], headers=['Type', 'Number of ratings'], tablefmt="latex"))


def review_bigrams_table(positive_bigrams, negative_bigrams):
    """
    Generate LaTeX tables for positive and negative bigrams
    :param positive_bigrams: List of positive bigrams represented as pairs (w1, w2)
    :param negative_bigrams: List of negative bigrams represented as pairs (w1, w2)
    """
    pos_rows = []
    for a, b in positive_bigrams:
        pos_rows.append([a + ' ' + b])
    print(tabulate(pos_rows, headers=['Positive bigrams'], tablefmt="latex"))

    neg_rows = []
    for a, b in negative_bigrams:
        neg_rows.append([a + ' ' + b])
    print(tabulate(neg_rows, headers=['Negative bigrams'], tablefmt="latex"))


def suggested_products(entries, analysed_id, name_mapping):
    """
    Generate LaTeX table for products related to the one analysed
    :param entries: List of entries
    :param analysed_id: Product_id of the analysed product
    :param name_mapping: Dictionary of {product_id: product_title}
    """
    # Set of all product keys
    possible_products = set()
    # List of purchased products for each user
    users_purchases = {}

    for entry in entries:
        product_id = entry['product/productId']
        user_key = entry['review/userId']
        possible_products.add(product_id)

        # Process review if user id is valid
        if user_key != 'unknown':
            if user_key not in users_purchases.keys():
                users_purchases[user_key] = [entry['product/productId']]
            else:
                users_purchases[user_key].append(entry['product/productId'])

    # Ignore users who bought only 1 product
    for key in list(users_purchases.keys()):
        if len(users_purchases[key]) < 2:
            users_purchases.pop(key)

    # Prepare dictionary of dictionaries of related products for further mapping
    product_map = {}
    for product in possible_products:
        product_map[product] = {}

    # Iterate over users purchases and and increment counter for products which were bought together
    for user in users_purchases:
        products = users_purchases[user]

        for (p1, p2) in combinations(products, 2):
            if p2 not in product_map[p1]:
                product_map[p1][p2] = 1
            else:
                product_map[p1][p2] += 1

            if p1 not in product_map[p2]:
                product_map[p2][p1] = 1
            else:
                product_map[p2][p1] += 1

    # Remove products which have no related products
    for key in list(product_map.keys()):
        if len(product_map[key]) == 0:
            product_map.pop(key)

    # Generate table for top 10 most related products in regard to analysed one
    related_dict = product_map[analysed_id]
    related = sorted(related_dict.items(), key=lambda item: item[1], reverse=True)

    n = min(len(related), 10)

    rows = []
    for prod_id, no_purchases in related[:n]:
        rows.append([name_mapping[prod_id], no_purchases])

    print(tabulate(rows, headers=['Product title', 'Number of purchases'], tablefmt="latex"))


def run_analysis(entries):
    """
    Process entries and generate results of analysis
    :param entries: List of entries
    """
    # Fields for general summary
    no_reviews = len(entries)
    unique_reviewers = set()
    unique_products = set()
    score_counter = Counter()
    ratings_by_year = Counter()
    ratings_by_product = Counter()
    total_rating = 0.0
    total_helpful_positive = 0.0
    total_helpful_scored = 0.0

    # Fields for extracting features
    positive_reviews = []
    no_positive_reviews = 0
    negative_reviews = []
    no_negative_reviews = 0

    # Mapping product titles to their ids
    product_name_mapping = {}

    # Id of product which will be individually analysed
    analysed_product_id = 'B0009B0IX4'
    analysed_product_by_year = Counter()

    for entry in entries:
        product_key = entry['product/productId']
        product_name = entry['product/title']
        product_name_mapping[product_key] = product_name

        # Combine summary and text for feature extraction
        text = entry['review/summary'] + " " + entry['review/text']

        unique_reviewers.add(entry['review/userId'])
        unique_products.add(product_key)

        # Process score and classify review as positive or negative
        score = float(entry['review/score'])
        score_counter[score] += 1
        total_rating += score
        if score >= 3.0:
            no_positive_reviews += 1
            positive_reviews.append(text)
        else:
            no_negative_reviews += 1
            negative_reviews.append(text)
        ratings_by_product[product_key] += 1

        # Process helpfulness
        helpfulness = entry['review/helpfulness'].split('/')
        total_helpful_positive += int(helpfulness[0])
        total_helpful_scored += int(helpfulness[1])

        # Process time and year
        unix_time = entry['review/time']
        year = datetime.utcfromtimestamp(int(unix_time)).strftime('%Y')
        ratings_by_year[year] += 1
        if product_key == analysed_product_id:
            analysed_product_by_year[year] += 1

    # Print metrics
    print('Total number of reviews: ', no_reviews)
    print('Unique products: ', len(unique_products))
    print('Unique reviewers: ', len(unique_reviewers))
    print('Average rating: ', "{:.2f}".format(round(total_rating / no_reviews, 2)))
    print('Average helpfulness: ', "{:.2f}".format(round(total_helpful_positive / total_helpful_scored, 2)))

    # Generate graphs and LaTeX tables
    popular_products_bar_chart(ratings_by_product, 20, product_name_mapping)
    reviews_year_bar_chart(ratings_by_year, title='Number of reviews in the following years')
    reviews_year_bar_chart(analysed_product_by_year,
                           title='Number of reviews for %s in the following years' % analysed_product_id)
    score_distribution_pie_chart(score_counter)
    score_type_distribution_pie_chart(no_positive_reviews, no_negative_reviews)

    # Extract bigrams from reviews
    positive_bigrams, negative_bigrams = compare_types_reviews(positive_reviews, negative_reviews)
    review_bigrams_table(positive_bigrams, negative_bigrams)

    # Find suggested product in regard to analysed one
    suggested_products(entries, analysed_product_id, product_name_mapping)


if __name__ == '__main__':

    entries = []

    for e in parse("Cell_Phones_&_Accessories.txt.gz"):
        entries.append(e)

    # Remove last entry since last row is empty
    entries.pop()

    # Remove entries with empty product title
    entries = [entry for entry in entries if entry['product/title'] != '']

    run_analysis(entries)
