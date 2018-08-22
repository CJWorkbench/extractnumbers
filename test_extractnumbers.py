import unittest
import pandas as pd
import numpy as np
from extractnumbers import render

m_map = {
            'type_extract': 'Any numerical value|Only integer|Only float value'.lower().split('|'),
            'type_format': 'U.S.|E.U.'.lower().split('|'),
            'type_replace': 'Null|0'.lower().split('|')
        }

class TestExtractNumbers(unittest.TestCase):

    def setUp(self):

        # Test data includes:
        #  - rows of string and categorical types
        #  - expected outputs
        #  - if column categorical type, retain categorical
        self.simple = pd.DataFrame([
            ['0,0',         '93,[note-2.0]',   '[note 4]88',   'nonumber'],
            ['1,000,000',   '99.0,98',      '999-72.2',     'yes99.0number']],
            columns=['stringcol1','stringcol2', 'catcol', 'nonum'])

        # Cast a category for index handling/preservation (string output only)
        self.simple['catcol'] = self.simple['catcol'].astype('category')

        self.result_simple_int = pd.DataFrame([
            [0,         93,   4,   0],
            [1000000,   98,   999, 0]],
            columns=['stringcol1','stringcol2', 'catcol', 'nonum'])


        self.result_simple_float = pd.DataFrame([
            [None,   -2.0,   None,   None],
            [None,   99.0,   -72.2,  99.0]],
            columns=['stringcol1','stringcol2', 'catcol', 'nonum'])

        self.result_simple_any = pd.DataFrame([
            [0,       93.0, 4.0,  None],
            [1000000.0, 99.0, 999.0,99.0]],
            columns=['stringcol1', 'stringcol2', 'catcol', 'nonum'])

        self.result_simple_number = pd.DataFrame([
            [None,      None, None, None],
            [1000000.0, None, None, None]],
            columns=['stringcol1', 'stringcol2', 'catcol', 'nonum'])

    def switch_formatting(self, table):
        s_placeholder = '|'
        d_placeholder = '*'
        table = table.applymap(lambda x: x.replace(',', s_placeholder))
        table = table.applymap(lambda x: x.replace('.', d_placeholder))
        table = table.applymap(lambda x: x.replace(s_placeholder, '.'))
        table = table.applymap(lambda x: x.replace(d_placeholder, ','))
        return table

    def test_NOP(self):
        # should NOP when first applied
        params = {'colnames': ''}
        out = render(self.simple, params)
        self.assertTrue(out.equals(self.simple))

    def test_extract_int(self):
        colnames = 'stringcol1,stringcol2,catcol,nonum'
        params = {'colnames': colnames,
                  'extract': True,
                  'type_extract': m_map['type_extract'].index('only integer'),
                  'type_format': m_map['type_format'].index('u.s.'),
                  'type_replace': m_map['type_replace'].index('0')}
        out = render(self.simple.copy(), params)
        pd.testing.assert_frame_equal(out, self.result_simple_int)

        # EU formatting
        params['type_format'] = m_map['type_format'].index('e.u.')
        out = render(self.switch_formatting(self.simple), params)
        pd.testing.assert_frame_equal(out, self.result_simple_int)

    def test_extract_float(self):
        colnames = 'stringcol1,stringcol2,catcol,nonum'
        params = {'colnames': colnames,
                  'extract': True,
                  'type_extract': m_map['type_extract'].index('only float value'),
                  'type_format': m_map['type_format'].index('u.s.'),
                  'type_replace': m_map['type_replace'].index('null')}
        out = render(self.simple.copy(), params)
        pd.testing.assert_frame_equal(out, self.result_simple_float)

        # EU
        params['type_format'] = m_map['type_format'].index('e.u.')
        out = render(self.switch_formatting(self.simple), params)
        pd.testing.assert_frame_equal(out, self.result_simple_float)

    def test_extract_any(self):
        colnames = 'stringcol1,stringcol2,catcol,nonum'
        params = {'colnames': colnames,
                  'extract': True,
                  'type_extract': m_map['type_extract'].index('any numerical value'),
                  'type_format': m_map['type_format'].index('u.s.'),
                  'type_replace': m_map['type_replace'].index('null')}
        out = render(self.simple.copy(), params)
        pd.testing.assert_frame_equal(out, self.result_simple_any)

        params['type_format'] = m_map['type_format'].index('e.u.')
        out = render(self.switch_formatting(self.simple), params)
        pd.testing.assert_frame_equal(out, self.result_simple_any)

    def test_to_number(self):
        colnames = 'stringcol1,stringcol2,catcol,nonum'
        params = {'colnames': colnames,
                  'extract': False,
                  'type_extract': m_map['type_extract'].index('any numerical value'),
                  'type_format': m_map['type_format'].index('u.s.'),
                  'type_replace': m_map['type_replace'].index('null')}
        out = render(self.simple.copy(), params)
        pd.testing.assert_frame_equal(out, self.result_simple_number)

        params['type_format'] = m_map['type_format'].index('e.u.')
        out = render(self.switch_formatting(self.simple), params)
        pd.testing.assert_frame_equal(out, self.result_simple_number)

if __name__ == '__main__':
    unittest.main()


