import ucimlrepo


def PreprocessFeatures(data):
	
	buying_to_number = {
		'vhigh': 4,
		'high' : 3,
		'med'  : 2,
		'low'  : 1}
	
	data.loc[:, 'buying'] = data['buying'].map(buying_to_number)
	
	
	maint_to_number = {
		'vhigh': 4,
		'high' : 3,
		'med'  : 2,
		'low'  : 1}
	
	data.loc[:, 'maint'] = data['maint'].map(maint_to_number)
	
	
	doors_to_number = {
		'2'    : 1,
		'3'    : 2,
		'4'    : 3,
		'5more': 4}
	
	data.loc[:, 'doors'] = data['doors'].map(doors_to_number)
	
	
	persons_to_number = {
		'2'   : 1,
		'4'   : 2,
		'more': 3}
	
	data.loc[:, 'persons'] = data['persons'].map(persons_to_number)
	
	
	lug_boot_to_number = {
		'small': 1,
		'med'  : 2,
		'big'  : 3}
	
	data.loc[:, 'lug_boot'] = data['lug_boot'].map(lug_boot_to_number)
	
	
	safety_to_number = {
		'low' : 1,
		'med' : 2,
		'high': 3}
	
	data.loc[:, 'safety'] = data['safety'].map(safety_to_number)
	
	
	return data


def PreprocessTarget(data):
	
	target_to_number = {
		'unacc': 1,
		'acc'  : 2,
		'good' : 3,
		'vgood': 4 }
	
	data.loc[:, 'class'] = data['class'].map(target_to_number)
	
	return data


def ConvertToBinary(data):
    # Ensure the target column is named 'Class'
    possible_target_names = ['class', 'target', 'label', 'outcome', 'category']  # Add any other possible names
    for name in possible_target_names:
        if name in data.columns:
            data.rename(columns={name: 'Class'}, inplace=True)
    
    # Now the 'Class' column is guaranteed to exist
    assert set(data['Class']) == {'acc', 'vgood', 'unacc', 'good'}
    
    data = data.copy()
    
    # Mapping target values to binary
    target_to_number = {
        'unacc': 0,
        'acc'  : 1,
        'good' : 1,
        'vgood': 1
    }
    
    # Apply the mapping to the 'Class' column
    data['Class'] = data['Class'].map(target_to_number)
    
    return data

def PrepareCarData():
	
	car_evaluation = ucimlrepo.fetch_ucirepo(id = 19)
	
	X = car_evaluation.data.features
	y = car_evaluation.data.targets
	
	X = PreprocessFeatures(X)
	y = ConvertToBinary(y)
	
	return X , y
