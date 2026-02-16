import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.preprocessing import OrdinalEncoder  # Uncomment if you prefer ordinal
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL_FILE_NEW='model_new.pkl'
PIPELINE_FILE_NEW='pipeline_new.pkl'

def build_pipeline(num_attribs,cat_attribs):
    # Numerical pipeline
    num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline([
    # ("ordinal", OrdinalEncoder())  # Use this if you prefer ordinal encoding
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Full pipeline
    full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),    #num and cat are just name for the transformation
    ("cat", cat_pipeline, cat_attribs),
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE_NEW):
    #then train the model
    # 1. Load the data
    housing = pd.read_csv("housing.csv")

   # 2. Create a stratified test set based on income category
    housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set= housing.loc[train_index].drop("income_cat", axis=1) 
        strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)
    
    #
    strat_test_set['median_house_value'].copy().to_csv('labels.csv',index=False)
    strat_test_set.drop('median_house_value',axis=1).to_csv('test.csv',index=False)
    
    # 3. Separate predictors and labels
    housing_labels = strat_train_set["median_house_value"].copy()
    housing_features = strat_train_set.drop("median_house_value", axis=1)

    # 4. Separate numerical and categorical columns
    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    pipeline=build_pipeline(num_attribs,cat_attribs)
   
    housing_prepared=pipeline.fit_transform(housing_features)
    
    model=RandomForestRegressor(random_state=42)
    model.fit(housing_prepared,housing_labels)

    joblib.dump(model,MODEL_FILE_NEW)
    joblib.dump(pipeline,PIPELINE_FILE_NEW)
    print("Model is trained . Congrats!")


else:
    #lets do inference-we don't check for accuray or error ,just do prediciton
    model=joblib.load(MODEL_FILE_NEW)
    pipeline=joblib.load(PIPELINE_FILE_NEW)

    input_data=pd.read_csv("test.csv")
    transformed_input=pipeline.transform(input_data)
    predictions=model.predict(transformed_input)
    input_data["median_house_value"]=predictions

    input_data.to_csv('output_test.csv',index=False)
    print('Inference is complete , result saved to output.csv Enjoy!')