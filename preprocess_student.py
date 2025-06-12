from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd


def check_missing_values(df, step):
    print(f"Missing values after {step}:")
    print(df.isna().sum())


def ProcessOrdinalFeat(data):

    #Numerical  Features: integers
    # Age 
    # failures :[0, 3, 1, 2]
    #  absences : integers

    # Nominal : Mjob, Fjob, reson, guardian(3), 
    #max unique values: 5
    #Ordinal  Features
    Medu_map = {
		0 : 1,
		1 : 2,
		2 : 3,
		3 : 4,
    4 : 5}
    data.loc[:, 'Medu'] = data['Medu'].map(Medu_map)

    Fedu_map = {
		0 : 1,
		1 : 2,
		2 :  3,
		3 : 4,
		4 : 5}
    data.loc[:, 'Fedu'] = data['Fedu'].map(Fedu_map)

    # traveltime, studytime : 1,2,3,4
    # famrel, freetime, goout, Dalc, Walc, health : 1,2,3,4,5

    data['Medu'] = data['Medu'].astype(int)
    data['Fedu'] = data['Fedu'].astype(int)
    ordinal_data = data[['Medu', 'Fedu', 'traveltime', 'studytime', 'famrel', 'freetime', 'goout', 'Dalc' ,'Walc', 'health']]

    # ordinal_data = data[['traveltime','studytime']]

    return ordinal_data

def Process_NominalFeat(data):
    #Binary Features
    school_map = {
		'GP' : 1,
		'MS' : 2}
    data.loc[:, 'school'] = data['school'].map(school_map)

    sex_map = {
		'F' : 0,
		'M' : 1}
    data.loc[:, 'sex'] = data['sex'].map(sex_map)

    address_map = {
		'U' : 0,
		'R' : 1}
    data.loc[:, 'address'] = data['address'].map(address_map)

    famsize_map = {
		'LE3' : 0,
		'GT3' : 1}
    data.loc[:, 'famsize'] = data['famsize'].map(famsize_map)

    Pstatus_map = {
		'A' : 0,
		'T' : 1}
    data.loc[:, 'Pstatus'] = data['Pstatus'].map(Pstatus_map)

    schoolsup_map = {
		'no' : 0,
		'yes' : 1}
    data.loc[:, 'schoolsup'] = data['schoolsup'].map(schoolsup_map)

    famsup_map = {
		'no' : 0,
		'yes' : 1}
    data.loc[:, 'famsup'] = data['famsup'].map(famsup_map)

    paid_map = {
		'no' : 0,
		'yes' : 1}
    data.loc[:, 'paid'] = data['paid'].map(paid_map)

    activities_map = {
		'no' : 0,
		'yes' : 1}
    data.loc[:, 'activities'] = data['activities'].map(activities_map)

    nursery_map = {
		'no' : 0,
		'yes' : 1}
    data.loc[:, 'nursery'] = data['nursery'].map(nursery_map)

    higher_map = {
		'no' : 0,
		'yes' : 1}
    data.loc[:, 'higher'] = data['higher'].map(higher_map)

    internet_map = {
		'no' : 0,
		'yes' : 1}
    data.loc[:, 'internet'] = data['internet'].map(internet_map)

    romantic_map = {
		'no' : 0,
		'yes' : 1}
    data.loc[:, 'romantic'] = data['romantic'].map(romantic_map)

    nominal_col = ['Mjob', 'Fjob', 'reason', 'guardian', 'school', 'sex', 'address','famsize', 'Pstatus','schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    nominal_data = data[nominal_col]

    return nominal_data



def Prepare_target(y):
    target_map = {
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
    y.loc[:, 'G3'] = y['G3'].map(target_map)

    y_target = y['G3']

    return y_target

def PrepareStudentData():

    student_performance = fetch_ucirepo(id=320) 

    X = student_performance.data.features 
    y = student_performance.data.targets

    check_missing_values(X, "after laoding data (orginal)")


    #X = process_features(X)

    y = Prepare_target(y)
    #y.rename('class', inplace=True)
    if isinstance(y, pd.Series):
        y = y.to_frame(name='Class')
    elif not isinstance(y, pd.DataFrame):
        y = pd.DataFrame(y, columns=['Class'])
    else:
        y.columns = ['Class'] 


    return X,y

def handleDataSet(X):
    
    # student_performance = fetch_ucirepo(id=320) 

    # X = student_performance.data.features 
    # y = student_performance.data.targets

    numeric_col = ['age', 'failures', 'absences']
    ordinal_col = ['Medu', 'Fedu', 'traveltime', 'studytime', 'famrel', 'freetime', 'goout', 'Dalc' ,'Walc', 'health']
    nominal_col = ['Mjob', 'Fjob', 'reason', 'guardian', 'school', 'sex', 'address','famsize', 'Pstatus','schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

    ordinal_data = ProcessOrdinalFeat(X[ordinal_col])
    nominal_data = X[nominal_col]
    numeric_data = X[numeric_col]

    return ordinal_data, nominal_data, numeric_data



def preprocess_student_data(X_train, X_test):
    
    # ## just active for baseline
    # for col in ['traveltime', 'studytime', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']:
    #   X_train[col] = X_train[col].astype('category')

    ordinal_data_train, nominal_data_train, numeric_data_train = handleDataSet(X_train)

    label_encoders = {}
    scaler = StandardScaler()
    
    ## ***** preprocess train *****
    #lable encoder for nominal
    for column in nominal_data_train.columns:
        le = LabelEncoder()
        try:
            nominal_data_train[column] = le.fit_transform(nominal_data_train[column])
        except Exception as e:
            print(f"Error encoding column '{column}': {e}")
        label_encoders[column] = le

    # Scale numeric data for training
    numeric_data_scaled_train = pd.DataFrame(scaler.fit_transform(numeric_data_train), columns=numeric_data_train.columns)

    # Reset indice --> avoid alignment issues
    ordinal_data_train.reset_index(drop=True, inplace=True)
    nominal_data_train.reset_index(drop=True, inplace=True)
    numeric_data_scaled_train.reset_index(drop=True, inplace=True)

    X_train_processed = pd.concat([ordinal_data_train, nominal_data_train, numeric_data_scaled_train], axis=1)


    ## ***** preprocess train (using the same encoders/scalers) ***** 
    ordinal_data_test, nominal_data_test, numeric_data_test = handleDataSet(X_test)

    for column in nominal_data_test.columns:
        nominal_data_test[column] = label_encoders[column].transform(nominal_data_test[column])

    numeric_data_scaled_test = pd.DataFrame(scaler.transform(numeric_data_test), columns=numeric_data_test.columns)

    ordinal_data_test = ordinal_data_test.reset_index(drop=True)
    nominal_data_test = nominal_data_test.reset_index(drop=True)
    numeric_data_scaled_test = numeric_data_scaled_test.reset_index(drop=True)

    X_test_processed = pd.concat([ordinal_data_test, nominal_data_test, numeric_data_scaled_test], axis=1)

    

    return X_train_processed, X_test_processed


if __name__ == '__main__':
	
    student_performance = fetch_ucirepo(id=320) 

    X = student_performance.data.features 
    y = student_performance.data.targets
    y = Prepare_target(y)
    #y.rename('class', inplace=True)
    if isinstance(y, pd.Series):
        y = y.to_frame(name='class')
    elif not isinstance(y, pd.DataFrame):
        y = pd.DataFrame(y, columns=['class'])
    else:
        y.columns = ['class']

    print(X.shape)

