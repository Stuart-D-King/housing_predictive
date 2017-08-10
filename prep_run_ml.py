import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, LabelBinarizer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from helper_classes import CombinedAttributesAdder, DataFrameSelector
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.externals import joblib


HOUSING_PATH = 'datasets/housing'


def impute_missing_values(df):
    '''
    Impute missing values with the median

    INPUT
    df: dataframe for which to impute missing values

    OUTPUT
    transformed housing dataframe
    '''
    imputer = Imputer(strategy='median')
    df_num = df.drop('ocean_proximity', axis=1)
    imputer.fit(df_num)
    X = imputer.transform(df_num)
    housing_tr = pd.DataFrame(X, columns=df_num.columns)
    return housing_tr


def encode_text(df):
    '''
    First encodes categorical values, then transforms into a one-hot encoded representation

    INPUT
    df: dataframe for which to encode categorical variables

    OUTPUT
    encoder object
    '''
    encoder = LabelEncoder()
    df_cat = df['ocean_proximity']
    df_cat_encoded = encoder.fit_transform(df_cat)

    onehot_encoder = OneHotEncoder()
    df_cat_1hot = onehot_encoder.fit_transform(df_cat_encoded.reshape(-1,1))
    return encoder


def label_binarize(df):
    '''
    Encodes categorical values and transforms them into one-hot encodings all in one go

    INPUT
    df: dataframe for which to one-hot encode

    OUTPUT
    one-hot encoding array of categorical feature
    '''
    encoder = LabelBinarizer(sparse_output=True)
    df_cat = df['ocean_proximity']
    df_cat_1hot = encoder.fit_transform(df_cat)
    return df_cat_1hot


def create_pipeline(data):
    '''
    Create and run a pipeline of transformation for numerical and categorical data

    INPUT
    data: dataframe of numerical and categorical features

    OUTPUT
    full pipeline object that processes numerical and categorical variables
    '''
    housing_num = data.drop('ocean_proximity', axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ['ocean_proximity']

    num_pipeline = Pipeline([
            ('selector', DataFrameSelector(num_attribs)),
            ('imputer', Imputer(strategy='median')),
            ('attribs_add', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(cat_attribs)),
            ('label_binarizer', LabelBinarizer()),
    ])

    full_pipeline = FeatureUnion(transformer_list=[
            ('num_pipeline', num_pipeline),
            ('cat_pipeline', cat_pipeline),
    ])

    # housing_prepared = full_pipeline.fit_transform(data)
    # return housing_prepared
    return full_pipeline

def run_linear_reg(X, y):
    '''
    fit a linear model

    INPUT
    X: features
    y: target variable

    OUTPUT
    linear regression model
    '''
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    lin_reg_predictions = lin_reg.predict(X)
    lin_reg_mse = mean_squared_error(y, lin_reg_predictions)
    lin_rmse = np.sqrt(lin_reg_mse)
    # print('Linear Regression RMSE:', lin_rmse)
    return lin_reg


def run_decision_tree(X, y):
    '''
    fit a decision tree regressor

    INPUT
    X: features
    y: target variable

    OUTPUT
    decision tree regressor
    '''
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(X, y)
    tree_predictions = tree_reg.predict(X)
    tree_mse = mean_squared_error(y, tree_predictions)
    tree_rmse = np.sqrt(tree_mse)
    # print('Decision Tree RMSE:', tree_rmse)
    return tree_reg


def run_random_forest(X, y):
    '''
    fit a random forest regressor

    INPUT
    X: features
    y: target variable

    OUTPUT
    random forest regressor
    '''
    rf_reg = RandomForestRegressor()
    rf_reg.fit(X, y)
    rf_predictions = rf_reg.predict(X)
    rf_mse = mean_squared_error(y, rf_predictions)
    rf_rmse = np.sqrt(rf_mse)
    # print('Random Forest RMSE:', rf_rmse)
    return rf_reg


def perform_cross_val(models, X, y):
    '''
    Perform cross validation for a series of modesl, display RMSE scores for each, and save each model

    INPUT
    models: list of regression models to score
    X: features
    y: target variable

    OUTPUT
    None (cross validation scores printed)
    '''
    for model in models:
        scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=10)
        rmse_scores = np.sqrt(-scores)
        display_scores(model, rmse_scores)

        joblib.dump(model, 'models/' + model.__class__.__name__ + '.pkl') # pickle and save each model; can load model later by running 'model = joblib.load('model_name.pkl')'


def display_scores(model, scores):
    '''
    Display summary of RMSE scores for a tested model

    INPUT
    model: model tested
    scores: RMSE scores for the passed in model

    OUTPUT
    Printed list of scores, mean score, and standard deviation
    '''
    print(model.__class__.__name__)
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard Deviation:', scores.std())
    print('--------------')


def perform_rf_grid_search(X, y):
    '''
    Run a grid search for the random forest regressor

    INPUT
    X: features
    y: target variable

    OUTPUT
    fitted grid search object
    '''
    param_grid = [
        {'n_estimators': [3,10,30], 'max_features': [2,4,6,8]}, {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2,3,4]}
    ]

    rf_reg = RandomForestRegressor()
    grid_search = GridSearchCV(rf_reg, param_grid, cv=5, scoring='neg_mean_squared_error')

    grid_search.fit(X, y)
    # cv_results = grid_search.cv_results_
    # for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
    #     print(np.sqrt(-mean_score), params)

    return grid_search


def show_feature_importance(data, encoder, grid):
    '''
    Display feature importances for the best estimator (regressor model)

    INPUT
    data: housing dataframe
    encoder: encoder object for categorical variable
    grid: random forest grid search object

    OUTPUT
    Print out of each attribute and its associated feature importance score
    '''
    housing_num = data.drop('ocean_proximity', axis=1)
    num_attribs = list(housing_num)
    cat_attribs = list(encoder.classes_)
    extra_attribs = ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_room']
    attributes = num_attribs + extra_attribs + cat_attribs

    feature_importances = grid.best_estimator_.feature_importances_

    for importance, attrib in sorted(zip(feature_importances, attributes), reverse=True):
        print('{}: {}'.format(attrib, importance))


def evaluate_final_model(grid, X, y):
    '''
    Score the final tuned model on the holdout test set

    INPUT
    grid: random forest grid search object
    X: test set features
    y: test set target variable

    OUTPUT
    RMSE score for final model
    '''
    final_model = grid.best_estimator_
    final_predictions = final_model.predict(X)
    final_mse = mean_squared_error(y, final_predictions)
    final_rmse = np.sqrt(final_mse)
    return final_rmse


if __name__ == '__main__':
    # load training data
    train = pd.read_csv(HOUSING_PATH + '/train.csv')
    # y = train.pop('median_house_value').values
    # X = train.values
    housing = train.drop('median_house_value', axis=1)
    housing_labels = train['median_house_value'].copy()
    encoder = encode_text(housing)
    pipeline = create_pipeline(housing)
    housing_prepared = pipeline.fit_transform(housing)


    # fit models, perform grid search, show feature importances
    lin_reg = run_linear_reg(housing_prepared, housing_labels)
    tree_reg = run_decision_tree(housing_prepared, housing_labels)
    rf_reg = run_random_forest(housing_prepared, housing_labels)

    models = [lin_reg, tree_reg, rf_reg]

    perform_cross_val(models, housing_prepared, housing_labels)
    grid_search = perform_rf_grid_search(housing_prepared, housing_labels)
    show_feature_importance(housing, encoder, grid_search)


    # load test data and evaluate final model
    test = pd.read_csv(HOUSING_PATH + '/test.csv')
    X_test = test.drop('median_house_value', axis=1)
    y_test = test['median_house_value'].copy()
    X_test_prepared = pipeline.transform(X_test)

    final_score = evaluate_final_model(grid_search, X_test_prepared, y_test)
