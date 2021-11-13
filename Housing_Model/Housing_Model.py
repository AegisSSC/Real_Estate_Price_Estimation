# To Do list:   
# Build a Visualizer 
# Build a Scraper to read in data
# Properly Segment and Generalize the Code
# Add SQL/Database Support
from plotter.plotting import *
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit



#Split the data set into a training set and a testing set. The Test Set will be roughly 20% of the 
# from sklearn.model_selection import train_test_split
def SplitToTraining(data_set, size):
    return train_test_split(data_set, test_size=size, random_state=42)


#Divide the information into a bar graph to represent "Bins" of data to better represent the data
# import numpy as np
def DivideToBins(housing):
    housing['income_cat'] = pd.cut(housing['median_income'], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
    housing['income_cat'].hist()
    plt.show()


#Now that you've seen that the data set is stratified, it is important to stratify the information. 
# In this case, we stratify based on the income category  
# from sklearn.model_selection import StratifiedShuffleSplit
def main():

    figure_size = (10,8)
    test_size = 0.2



    housing = pd.read_csv("housing-data/housing.csv")
    housing.head()
    plot_histogram(housing,50,figure_size)
    SplitToTraining(housing,test_size)
    DivideToBins(housing)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    print(strat_test_set['income_cat'].value_counts() / len(strat_test_set))

    #Now you can remove the income category added for stratification to get the data back to its original form
    for set_ in (strat_train_set, strat_test_set):
        set_.drop('income_cat', axis=1, inplace=True)
    housing = strat_train_set.copy()

        

    #HeatMap of housing data. 
    #Plot the data set as a scatter plot to visualize the data. 
    heatmap(housing)

    #use the corr() to find the correlation coefficent between each of the attributes. 
    corr_matrix = housing.corr()
    print(corr_matrix.median_house_value.sort_values(ascending=False))


    #revise the correlation by looking at it in terms of 3 new terms. 
    # Term 1: Rooms per household, the average number of rooms in a house at a given data point
    # Term 2: Bedrooms per room, the ratio of bedrooms to total number of rooms in a house at a given data point. 
    # Term 3: Population per household, the average number of people who live in a house at a given data point
    housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
    housing["population_per_household"] = housing["population"]/housing["households"]

    #redefine the correlation matrix in terms of these new terms
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    # Prepare the data to be used in model training
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    median = housing["total_bedrooms"].median()
    housing["total_bedrooms"].fillna(median, inplace=True)

    housing_num = housing.drop("ocean_proximity", axis=1)

    # from sklearn.base import BaseEstimator, TransformerMixin

    # column index
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
            self.add_bedrooms_per_room = add_bedrooms_per_room
        def fit(self, X, y=None):
            return self  # nothing else to do
        def transform(self, X):
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            population_per_household = X[:, population_ix] / X[:, households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household,
                            bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]

    # from sklearn.preprocessing import OneHotEncoder
    # from sklearn.pipeline import Pipeline
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.impute import SimpleImputer
    num_pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    housing_num_tr = num_pipeline.fit_transform(housing_num)

    # from sklearn.compose import ColumnTransformer
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
    housing_prepared = full_pipeline.fit_transform(housing)

    # from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    data = housing.iloc[:5]
    labels = housing_labels.iloc[:5]
    data_preparation = full_pipeline.transform(data)
    print("Predictions: ", lin_reg.predict(data_preparation))