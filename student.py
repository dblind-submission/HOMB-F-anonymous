




import ucimlrepo
import pandas as pd

# def PreprocessFeatures(data):

#     #Binary Features : 13 features
#     school_map = {
# 		'GP' : 1,
# 		'MS' : 2}
#     data.loc[:, 'school'] = data['school'].map(school_map)

#     sex_map = {
# 		'F' : 0,
# 		'M' : 1}
#     data.loc[:, 'sex'] = data['sex'].map(sex_map)

#     address_map = {
# 		'U' : 0,
# 		'R' : 1}
#     data.loc[:, 'address'] = data['address'].map(address_map)

#     famsize_map = {
# 		'LE3' : 0,
# 		'GT3' : 1}
#     data.loc[:, 'famsize'] = data['famsize'].map(famsize_map)

#     Pstatus_map = {
# 		'A' : 0,
# 		'T' : 1}
#     data.loc[:, 'Pstatus'] = data['Pstatus'].map(Pstatus_map)

#     schoolsup_map = {
# 		'no' : 0,
# 		'yes' : 1}
#     data.loc[:, 'schoolsup'] = data['schoolsup'].map(schoolsup_map)

#     famsup_map = {
# 		'no' : 0,
# 		'yes' : 1}
#     data.loc[:, 'famsup'] = data['famsup'].map(famsup_map)

#     paid_map = {
# 		'no' : 0,
# 		'yes' : 1}
#     data.loc[:, 'paid'] = data['paid'].map(paid_map)

#     activities_map = {
# 		'no' : 0,
# 		'yes' : 1}
#     data.loc[:, 'activities'] = data['activities'].map(activities_map)

#     nursery_map = {
# 		'no' : 0,
# 		'yes' : 1}
#     data.loc[:, 'nursery'] = data['nursery'].map(nursery_map)

#     higher_map = {
# 		'no' : 0,
# 		'yes' : 1}
#     data.loc[:, 'higher'] = data['higher'].map(higher_map)

#     internet_map = {
# 		'no' : 0,
# 		'yes' : 1}
#     data.loc[:, 'internet'] = data['internet'].map(internet_map)

#     romantic_map = {
# 		'no' : 0,
# 		'yes' : 1}
#     data.loc[:, 'romantic'] = data['romantic'].map(romantic_map)




#     #Numerical  Features: integers
#     # Age 
#     # failures :[0, 3, 1, 2]
#     #  absences : integers

#     # Nominal : Mjob (#unique: 5), Fjob(#unique: 5), reson(#unique: 4), guardian(3), 
#     #max unique values: 5
#     #Ordinal  Features

    
#     Medu_map = {
# 		0 : 1,
# 		1 : 2,
# 		2 : 3,
# 		3 : 4,
#         4 : 5}
#     data.loc[:, 'Medu'] = data['Medu'].map(Medu_map)

#     Fedu_map = {
# 		0 : 1,
# 		1 : 2,
# 		2:  3,
# 		3 : 4,
# 		4 : 5}
#     data.loc[:, 'Fedu'] = data['Fedu'].map(Fedu_map)

#     # traveltime, studytime : 1,2,3,4
#     # famrel, freetime, goout, Dalc, Walc, health : 1,2,3,4,5

#     ordinal_data = data[['Medu', 'Fedu', 'traveltime', 'studytime', 'famrel', 'freetime', 'goout', 'Dalc' ,'Walc', 'health']]
#     # Binary_data = data[['school', 'sex', 'address','famsize', 'Pstatus','schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']]
#     # Nominal_data = data[['Mjob', 'Fjob', 'reason', 'guardian',   'school', 'sex', 'address','famsize', 'Pstatus','schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']]
#     # Numerical_data = data[['age', 'failures', 'absences']]
#     # return ordinal_data, Nominal_data, Numerical_data
#     return ordinal_data

def PreprocessOrdinals(data):
    
    ordinal_columns = ['Medu', 'Fedu', 'traveltime', 'studytime', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']
    ordinal_data = data[ordinal_columns].copy()

    Medu_map = {
		0 : 1,
		1 : 2,
		2 : 3,
		3 : 4,
    4 : 5}
    ordinal_data.loc[:, 'Medu'] = ordinal_data['Medu'].map(Medu_map)

    Fedu_map = {
		0 : 1,
		1 : 2,
		2:  3,
		3 : 4,
		4 : 5}
    ordinal_data.loc[:, 'Fedu'] = ordinal_data['Fedu'].map(Fedu_map)

    return ordinal_data

def ConvertToBinary(data):
    # Ensure the target column is named 'Class'
    possible_target_names = ['class', 'target', 'label', 'outcome', 'category']  # Add any other possible names
    for name in possible_target_names:
        if name in data.columns:
            data.rename(columns={name: 'Class'}, inplace=True)
    
    assert set(data['Class']) == {'acc', 'vgood', 'unacc', 'good'}
    
    data = data.copy()
    
    # Mapping target values to binary
    target_to_number = {
        'unacc': 0,
        'acc'  : 1,
        'good' : 1,
        'vgood': 1
    }
    
    data['Class'] = data['Class'].map(target_to_number)
    
    return data


def ConvertToBinary(data):

    data = data[['G3']].copy() 
    data.rename(columns={'G3': 'Class'}, inplace=True)
    
    target_to_number ={
        0     :  0,
        1     :  0,
        5     :  0,
        6     :  0,
        7     :  0,
        8     :  0,
        9     :  0,
        10    :  0,
        11    :  0,
        12    :  1,
        13    :  1,
        14    :  1,
        15    :  1,
        16    :  1,
        17    :  1,
        18    :  1,
        19    :  1}
    
    data['Class'] = data['Class'].map(target_to_number)

    return data

def PrepareStudentData():
	
	car_evaluation = ucimlrepo.fetch_ucirepo(id = 320)
	
	X = car_evaluation.data.features
	y = car_evaluation.data.targets
	
	# ordinal_data, Nominal_data, Numerical_data  = PreprocessFeatures(X)
	just_ordinal_data  = PreprocessOrdinals(X)
     
	y = ConvertToBinary(y)
	
	return X , just_ordinal_data

# def preprocess_mvl():
    
#     car_evaluation = ucimlrepo.fetch_ucirepo(id = 320)
	
#     X = car_evaluation.data.features
#     y = car_evaluation.data.targets
	
#     ordinal_data, Nominal_data, Numerical_data  = PreprocessFeatures(X)
#     y = ConvertToBinary(y)
	
#     return 

if __name__ == '__main__':
	
    student_performance = ucimlrepo.fetch_ucirepo(id=320) 

    X = student_performance.data.features 
    y = student_performance.data.targets

    missing_values = X.isnull().sum()
    
    print("Missing values in each column:")
    print(missing_values[missing_values > 0]) 
    total_missing = missing_values.sum()
    print(f'Total missing values in the dataset: {total_missing}')

    print('Size of x:', X.shape)
    print('Size of y:', y.shape)

    print("Columns in X:")
    print(X.columns.tolist())

    columns_with_services = df.columns[df.isin(['services']).any()]

# Print the columns that contain 'services'
    print("Columns containing 'services':", list(columns_with_services))

    print('*************************************')







