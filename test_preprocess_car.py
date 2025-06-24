import pytest

import preprocess_car


def test_expected_distance():
	
	X , y = preprocess_car.PrepareCarData()
	
	assert X.shape == (1728, 6)
	assert y.shape == (1728, 1)
	
	total = y.shape[0]
	case_count = y.iloc[:, 0].sum()
	
	prevalence = case_count / total
	
	assert case_count == 518
	assert round(prevalence, 3) == 0.30
