import unittest
import pandas as pd
import numpy as np
from extractnumbers import render


class TestExtractNumbers(unittest.TestCase):

    def setUp(self):
        # Test data includes:
        #  - rows of string and categorical types
        #  - expected outputs
        #  - if column categorical type, retain categorical
        self.table = pd.DataFrame([
            ['0.565[note1]',        '93[note2]',        '[note 4]88',       'nonumber'],
            ['10%',                 '99%   some text',  '999-72.2',         'yes99.0number']],
            columns=['stringcol1','stringcol2', 'catcol', 'nonum'])

        # Cast a category for index handling/preservation (string output only)
        self.table['catcol'] = self.table['catcol'].astype('category')

        self.result_int = pd.DataFrame([
            [1,     93,     4,      -1],
            [10,    99,     999,    -1]],
            columns=['stringcol1', 'stringcol2', 'catcol', 'nonum'])

        self.result_int['stringcol1'] = self.result_int['stringcol1'].astype(np.int64)
        self.result_int['stringcol2'] = self.result_int['stringcol2'].astype(np.int64)
        self.result_int['nonum'] = self.result_int['nonum'].astype(np.int64)
        self.result_int['catcol'] = self.result_int['catcol'].astype(np.int64)

        self.result_float = pd.DataFrame([
            [0.565,     np.nan,     np.nan,      None],
            [np.nan,      np.nan,     72.2,    99.0]],
            columns=['stringcol1', 'stringcol2', 'catcol', 'nonum'])

        self.result_float['stringcol1'] = self.result_float['stringcol1'].astype(float)
        self.result_float['stringcol2'] = self.result_float['stringcol2'].astype(float)
        self.result_float['nonum'] = self.result_float['nonum'].astype(float)
        self.result_float['catcol'] = self.result_float['catcol'].astype(float)

        self.result_any = pd.DataFrame([
            [0.565, 93, 4,  None],
            [10.0, 99, 999, 99.0]],
            columns=['stringcol1', 'stringcol2', 'catcol', 'nonum'])

        self.result_any['stringcol1'] = self.result_any['stringcol1'].astype(float)
        self.result_any['stringcol2'] = self.result_any['stringcol2'].astype(float)
        self.result_any['nonum'] = self.result_any['nonum'].astype(float)
        self.result_any['catcol'] = self.result_any['catcol'].astype(float)

        self.result_text_before_float = pd.DataFrame([
            ['0.565',   '',     '',         ''],
            ['',        '',     '999-72.2',     'yes99.0']],
            columns=['stringcol1', 'stringcol2', 'catcol', 'nonum'])

        self.result_text_before_float['catcol'] = self.result_text_before_float['catcol'].astype('category')

        self.result_text_before_int = pd.DataFrame([
            ['0.565[note1',     '93',     '[note 4',        ''],
            ['10',                '99',     '999',            '']],
            columns=['stringcol1', 'stringcol2', 'catcol', 'nonum'])

        self.result_text_before_int['catcol'] = self.result_text_before_int['catcol'].astype('category')

        self.result_text_before_any = pd.DataFrame([
            ['0.565',        '93',      '[note 4',      ''],
            ['10',           '99',      '999',          'yes99.0']],
            columns=['stringcol1','stringcol2', 'catcol', 'nonum'])

        # Cast a category for index handling/preservation (string output only)
        self.result_text_before_any['catcol'] = self.result_text_before_any['catcol'].astype('category')

        self.result_text_after_float = pd.DataFrame([
            ['0.565[note1]',        '',        '',       ''],
            ['',                    '',         '72.2',  '99.0number']],
            columns=['stringcol1', 'stringcol2', 'catcol', 'nonum'])

        self.result_text_after_float['catcol'] = self.result_text_after_float['catcol'].astype('category')

        self.result_text_after_int = pd.DataFrame([
            ['1]',        '93[note2]',        '4]88',       ''],
            ['10%',       '99%   some text',  '999-72.2',   '']],
            columns=['stringcol1', 'stringcol2', 'catcol', 'nonum'])

        self.result_text_after_int['catcol'] = self.result_text_after_int['catcol'].astype('category')

        self.result_text_after_any = pd.DataFrame([
            ['0.565[note1]',        '93[note2]',        '4]88',       ''],
            ['10%',                 '99%   some text',  '999-72.2',   '99.0number']],
            columns=['stringcol1', 'stringcol2', 'catcol', 'nonum'])

        self.result_text_after_any['catcol'] = self.result_text_after_any['catcol'].astype('category')


        self.result_text_before_and_after_float = pd.DataFrame([
            ['0.565[note1]', '', '', ''],
            ['', '', '999-72.2', 'yes99.0number']],
            columns=['stringcol1', 'stringcol2', 'catcol', 'nonum'])

        self.result_text_before_and_after_float['catcol'] = self.result_text_before_and_after_float['catcol'].astype('category')


        self.result_text_before_and_after_int = pd.DataFrame([
            ['0.565[note1]',        '93[note2]',        '[note 4]88',       ''],
            ['10%',                 '99%   some text',  '999-72.2',         '']],
            columns=['stringcol1', 'stringcol2', 'catcol', 'nonum'])

        self.result_text_before_and_after_int['catcol'] = self.result_text_before_and_after_int['catcol'].astype('category')

        self.result_text_before_and_after_any = pd.DataFrame([
            ['0.565[note1]',        '93[note2]',        '[note 4]88',       ''],
            ['10%',                 '99%   some text',  '999-72.2',         'yes99.0number']],
            columns=['stringcol1', 'stringcol2', 'catcol', 'nonum'])

        self.result_text_before_and_after_any['catcol'] = self.result_text_before_and_after_any['catcol'].astype('category')

    def test_NOP(self):
        # should NOP when first applied
        params = {'colnames': '', 'type': 0, 'text_before': False, 'text_after': False}
        out = render(self.table, params)
        self.assertTrue(out.equals(self.table))

    def test_extract_int(self):
        colnames = 'stringcol1,stringcol2,catcol,nonum'
        params = {'colnames': colnames, 'type': 0, 'text_before': False, 'text_after': False}
        out = render(self.table, params)
        pd.testing.assert_frame_equal(out, self.result_int)

    def test_extract_float(self):
        colnames = 'stringcol1,stringcol2,catcol,nonum'
        params = {'colnames': colnames, 'type': 1, 'text_before': False, 'text_after': False}
        out = render(self.table, params)
        pd.testing.assert_frame_equal(out, self.result_float)

    def test_extract_any(self):
        colnames = 'stringcol1,stringcol2,catcol,nonum'
        params = {'colnames': colnames, 'type': 2, 'text_before': False, 'text_after': False}
        out = render(self.table, params)
        pd.testing.assert_frame_equal(out, self.result_any)

    def test_extract_text_before(self):
        colnames = 'stringcol1,stringcol2,catcol,nonum'
        params = {'colnames': colnames, 'type': 1, 'text_before': True, 'text_after': False}
        out = render(self.table.copy(), params)
        pd.testing.assert_frame_equal(out, self.result_text_before_float,check_categorical=False)

        colnames = 'stringcol1,stringcol2,catcol,nonum'
        params = {'colnames': colnames, 'type': 0, 'text_before': True, 'text_after': False}
        out = render(self.table.copy(), params)
        pd.testing.assert_frame_equal(out, self.result_text_before_int, check_categorical=False)

        colnames = 'stringcol1,stringcol2,catcol,nonum'
        params = {'colnames': colnames, 'type': 2, 'text_before': True, 'text_after': False}
        out = render(self.table.copy(), params)
        pd.testing.assert_frame_equal(out, self.result_text_before_any, check_categorical=False)

    def test_extract_text_after(self):
        colnames = 'stringcol1,stringcol2,catcol,nonum'
        params = {'colnames': colnames, 'type': 1, 'text_before': False, 'text_after': True}
        out = render(self.table.copy(), params)
        pd.testing.assert_frame_equal(out, self.result_text_after_float, check_categorical=False)

        colnames = 'stringcol1,stringcol2,catcol,nonum'
        params = {'colnames': colnames, 'type': 0, 'text_before': False, 'text_after': True}
        out = render(self.table.copy(), params)
        pd.testing.assert_frame_equal(out, self.result_text_after_int, check_categorical=False)

        colnames = 'stringcol1,stringcol2,catcol,nonum'
        params = {'colnames': colnames, 'type': 2, 'text_before': False, 'text_after': True}
        out = render(self.table.copy(), params)
        pd.testing.assert_frame_equal(out, self.result_text_after_any, check_categorical=False)

    def test_extract_before_and_after(self):
        colnames = 'stringcol1,stringcol2,catcol,nonum'
        params = {'colnames': colnames, 'type': 1, 'text_before': True, 'text_after': True}
        out = render(self.table.copy(), params)
        pd.testing.assert_frame_equal(out, self.result_text_before_and_after_float, check_categorical=False)

        colnames = 'stringcol1,stringcol2,catcol,nonum'
        params = {'colnames': colnames, 'type': 0, 'text_before': True, 'text_after': True}
        out = render(self.table.copy(), params)
        pd.testing.assert_frame_equal(out, self.result_text_before_and_after_int, check_categorical=False)

        colnames = 'stringcol1,stringcol2,catcol,nonum'
        params = {'colnames': colnames, 'type': 2, 'text_before': True, 'text_after': True}
        out = render(self.table.copy(), params)
        pd.testing.assert_frame_equal(out, self.result_text_before_and_after_any, check_categorical=False)


if __name__ == '__main__':
    unittest.main()


