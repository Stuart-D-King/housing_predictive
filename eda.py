import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix


HOUSING_PATH = 'datasets/housing'


def lat_long_visual(df):
    '''
    Scatter plot the latitude and longitude coordinates of each district

    INPUT
    df: housing dataframe

    OUTPUT
    none; saved plot
    '''
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)

    ax.scatter(x=df.longitude, y=df.latitude, alpha=0.1)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    # df.plot(kind='scatter', x='longitude', y='latitude') # another way to do the same as the above
    plt.savefig('plots/lat_long.png', dpi=200)
    plt.close()


def lat_long_visual_2(df):
    '''
    Scatter plot of latitude and longitude coordinates of each district; point sizes dictated by population; color associated by median house value

    INPUT
    df: housing dataframe

    OUTPUT
    none; saved plot
    '''
    df.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=df['population']/100, label='population', c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
    plt.legend()
    plt.savefig('plots/lat_long_2.png', dpi=200)
    plt.close()


def make_scatter_matrix(df):
    '''
    Scatter matrix of select numerical features

    INPUT
    df: housing dataframe

    OUTPUT
    none; saved plot
    '''
    attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
    sc = scatter_matrix(df[attributes], figsize=(12,8))
    plt.tight_layout()
    plt.savefig('plots/scatter_matrix.png', dpi=200)
    plt.close()


def median_house_value_scatter(df):
    '''
    Scatter plot of median home value variable

    INPUT
    df: housing dataframe

    OUTPUT
    none; saved plot
    '''
    df.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)
    plt.savefig('plots/value_vs_income.png', dpi=200)
    plt.close()


if __name__ == '__main__':
    train = pd.read_csv(HOUSING_PATH + '/train.csv')
    df_train = train.copy()

    lat_long_visual(df_train)
    lat_long_visual_2(df_train)

    corr_matrix = df_train.corr() # correlation matrix of every pair of attributes

    make_scatter_matrix(df_train)
    median_house_value_scatter(df_train)
