from common import *

def map_and_replace_columns(df, columns):
    mappings = {}  # Dictionary to store the mapping for each column
    
    for column in columns:
        # Get unique values from the column, sorted
        unique_values = sorted(df[column].unique())
        
        # Create a mapping from sorted unique values to values starting from 1
        mapping = {original_value: new_value for new_value, original_value in enumerate(unique_values, start=1)}
        
        # Replace the values in the column based on the mapping
        df[column] = df[column].map(mapping)
        
        # Store the mapping for reference
        mappings[column] = mapping
    
    return df
def PrepareMimicData():
      
	X = pd.read_csv('../train.csv')
	y = X[['outcome_hospitalization']].rename(columns={'outcome_hospitalization': 'Class'})
  
	ordinal_columns = ['triage_acuity', 'triage_pain']
	
	
	X = X[ordinal_columns]
	X[ordinal_columns] = map_and_replace_columns(X, ordinal_columns)


	return X , y
