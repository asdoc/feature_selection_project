from unittest import TestCase
from ..build import forward_selected
from inspect import getargspec
from sklearn.linear_model import LinearRegression
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')
model = LinearRegression()


class TestForward_selected(TestCase):
    def test_forward_selected(self):
        # Input parameters tests
        args = getargspec(forward_selected)
        self.assertEqual(len(args[0]), 2, "Expected arguments %d, Given %d" % (2, len(args[0])))
        self.assertEqual(args[3], None, "Expected default values do not match given default values")

        # Return data types
        fwd_selection, best_score = forward_selected(data, model)
        self.assertIsInstance(fwd_selection, list,
                              "Expected data type for return value is `List`, you are returning %s" % (
                                  type(fwd_selection)))
        self.assertIsInstance(best_score, list,
                              "Expected data type for return value is `List`, you are returning %s" % (
                                  type(best_score)))

        # Return values tests
        expected_var = ['OverallQual', 'GrLivArea', 'BsmtFinSF1', 'GarageCars', 'KitchenAbvGr', '1stFlrSF',
                        'YearRemodAdd',
                        'LotArea', 'MasVnrArea', 'WoodDeckSF']

        top_10 = fwd_selection[0:10]

        expected_acc = [0.61972765016619102, 0.7110122362921284, 0.74208020244393813, 0.76370229136595302,
                        0.77146549956264021, 0.77743942439428682, 0.78190516290253775,
                        0.78559309845190683, 0.78926329832950681, 0.79084962683577575]
        top_acc = best_score[0:10]
        self.assertItemsEqual(top_10, expected_var, "Expected values does not match returned value")
        self.assertAlmostEqual(top_acc, expected_acc, 3, "Expected values does not match returned value")