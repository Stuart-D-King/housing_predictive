### Modeling Housing Prices in California

This repository is a collection of Python scripts I developed while following along with the instruction from Chapter 2 of __Hands-On Machine Learning with Scikit-Learn and Tensorflow__ by Aurélien Géron. This excellent book steps the reader through a simulated end-to-end machine learning project using California census data to build a prediction model of housing prices in California. The provided dataset consists of 20,640 observations and 10 attributes.

#### Data collection and EDA
After downloading and saving the data to file, initial exploratory data analysis was performed to get a quick feel for the data by plotting histograms of all numerical features. Next, the data was split into training and test sets using Scikit-Learn's __StratifiedShuffleSplit__ class. Stratified sampling was performed using a newly created median income category variable. Additional visualizations were then created using the training data, including scatter plots of each district's latitude and longitude coordinates, with marker size determined by population and marker color by median home value. Furthermore, scatter and correlation matrices of numerical values illuminated some of the more promising attributes to predict the median house value, the strongest correlation being observed between median house value and median income.

#### Data preparation and transformation
To prepare the data for machine learning, a series of functions were created to impute missing values, encode categorical variables, and binarize (one-hot) numerical encodings. In addition, custom classes was created to add new attribute combinations to the dataset, as well as select user-defined features from the dataset. To streamline the series of transformations, a Scikit-Learn pipeline object was created to easily impute, encode, and standardize the training and test datasets.

#### Model selection and training
Three models were fit using the preprocessed training data: Linear Regression model, Decision Tree Regressor, and a Random Forest Regressor. Upon fitting each model using the training data, cross validation was performed to score and evaluate each model's performance. Each model's average root mean squared error (RMSE) was calculated using 10-fold cross validation. The __Random Forest Regressor__ output the lowest RMSE score, and was thus selected as the model to be fine-tuned.

#### Model tuning
To select the best model hyperparameters, I used Scikit-Learn's __GridSearchCV__ to explore the various combinations of hyperparameters that would improve model performance.

#### Feature importance
Once the best estimator (model with set of hyperparameters that provided the lowest RMSE score) was determined, I evaluated the feature importance scores for each attribute of the data. Based on the results of this analysis, median income, inland (ocean proximity category), and population per household were the top three most useful predictors of housing prices.

#### Evaluation on test set
Finally, the Random Forest Regressor model was evaluated on the test set.
