import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import tarfile
from six.moves import urllib


DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = 'datasets/housing'
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + '/housing.tgz'


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    '''
    Fetch data from online repository; save to file (create directory if non-existent)

    INPUT
    housing_url: full url link to dataset
    housing_path: directory path above root

    OUTPUT
    none
    '''
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    '''
    Load data from file and create categorized median income variable

    INPUT
    housing_path: directory of saved data

    OUTPUT
    dataframe with categorized median income variable added
    '''
    csv_path = os.path.join(housing_path, 'housing.csv')
    df = pd.read_csv(csv_path)
    return categorize_median_income(df)


def make_histograms(df):
    '''
    Create a plot of histograms for each numerical variable

    INPUT
    df: housing dataframe

    OUTPUT
    none; saved plot
    '''
    df.hist(bins=50, figsize=(10,8))
    plt.savefig('plots/histograms', dpi=200)
    plt.close()


def create_new_attributes(df):
    '''
    Create new attribute combinations and add to input dataframe

    INPUT
    df: housing dataframe

    OUTPUT
    housing dataframe with new attributes added
    '''
    df['rooms_per_household'] = df['total_rooms']/df['households']
    df['bedrooms_per_room'] = df['total_bedrooms']/df['total_rooms']
    df['population_per_household'] = df['population']/df['households']
    return df


def categorize_median_income(df):
    '''
    Create a variable that categorizes the median income, capping the highest category value at 5.0

    INPUT
    df: housing dataframe

    OUTPUT
    housing dataframe with categorized median income variable added
    '''
    df['income_cat'] = np.ceil(df['median_income'] / 1.5)
    df['income_cat'].where(df['income_cat'] < 5, 5.0, inplace=True)
    return df


def stratified_split(df):
    '''
    Perform sklearn's StratifiedShuffleSplit to create training and test sets, with the median income category variable serving as the feature to stratify the split

    INPUT
    df: housing dataframe

    OUTPUT
    strat_train: stratified training set
    strat_test: stratified test set
    '''
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in split.split(df, df['income_cat']):
        strat_train = df.loc[train_idx]
        strat_test = df.loc[test_idx]

    return strat_train, strat_test


if __name__ == '__main__':
    fetch_housing_data()
    df = load_housing_data()
    # make_histograms(df) # first run saved to file
    # df = create_new_attributes(df) # add new attribute combinations (also done later with custom class)
    train, test = stratified_split(df) # stratified split using the newly created income_cat column

    # drop income_cat column to revert back to original state
    for s in (train, test):
        s.drop(['income_cat'], axis=1, inplace=True)

    # save training and test sets to file
    train.to_csv(HOUSING_PATH + '/train.csv', index=False)
    test.to_csv(HOUSING_PATH + '/test.csv', index=False)
