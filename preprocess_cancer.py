import ucimlrepo
from sklearn.impute import SimpleImputer

import numpy as np

def PreprocessFeatures(data):

	data['Mitoses'].replace(10, 9, inplace=True)
	return data


def PrepareCancerData():
	
	breast_cancer_wisconsin_original = ucimlrepo.fetch_ucirepo(id=15)
	
	X = breast_cancer_wisconsin_original.data.features 
	y = breast_cancer_wisconsin_original.data.targets
	assert y['Class'].notna().all(), 'y must have no NA'
	y['Class'] = y['Class'].replace({2: 0, 4: 1})
	X = PreprocessFeatures(X)
	
	return X , y

def impute_missing_with_most_frequent(df, column_name, imputer=None, fit=False):
    # Replace 'nan', 'NaN' with actual np.nan values
    df[column_name] = df[column_name].replace(['nan', 'NaN'], np.nan)
    
    if fit:
        # Fit the imputer on the training data only
        imputer = SimpleImputer(strategy='most_frequent')
        df[[column_name]] = imputer.fit_transform(df[[column_name]])
    else:
        # For the test data, only apply transform
        df[[column_name]] = imputer.transform(df[[column_name]])

    # Convert to int after imputation (if needed)
    df[column_name] = df[column_name].astype(int)
    
    return df, imputer

if __name__ == '__main__':
	
	X , y = PrepareCancerData()
	for column in X.columns:
		print(f"Unique values in '{column}': {X[column].unique()}")
