import regex

# Ignore negative numbers for now
# TODO first column goes twice

m_map = {
    'type_extract':   'Any numerical value|Only integer|Only float value'.lower().split('|'),
    'type_format':    'U.S.|E.U.'.lower().split('|'),
    'type_replace':    'Null|0'.lower().split('|')
}

p_map = {
    'u.s.': {
        'separator': ',',
        'decimal': '.'
    },
    'e.u.': {
        'separator': '.',
        'decimal': ','
    }
}

# Extracts all non-negative numbers for now
def render(table, params):
    # if no column has been selected, return table
    if not params['colnames']:
        return table

    extract = params['extract']
    type_extract = m_map['type_extract'][params['type_extract']] if extract else None
    type_format = m_map['type_format'][params['type_format']]
    type_replace = m_map['type_replace'][params['type_replace']]
    columns = [c.strip() for c in params['colnames'].split(',')]

    separator = p_map[type_format]['separator']
    decimal = p_map[type_format]['decimal']

    text_columns = table[columns].select_dtypes(include=['object', 'category']).columns
    dtypes = table[text_columns].get_dtype_counts().index

    type_replace = 0 if type_replace == '0' else None

    if not extract:
        dispatch = lambda x: convert_as_is(x, separator, decimal, type_replace) if x else type_replace
    elif type_extract == 'only integer':
        dispatch = lambda x: extract_int(x, separator, decimal, type_replace) if x else type_replace
    elif type_extract == 'only float value':
        dispatch = lambda x: extract_float(x, separator, decimal, type_replace) if x else type_replace
    else:
        dispatch = lambda x: extract_any(x, separator, decimal, type_replace) if x else type_replace

    for dtype in dtypes:
        dtype_columns = table[text_columns].dtypes[table[columns].dtypes == dtype].index
        if dtype == 'category':
            for column in dtype_columns:
                prepare_cat(table[column])
                # TODO cast as string for now, improve performance by operating on categorical index
                table[column] = table[column].astype(str).apply(dispatch)
        else:
            table[dtype_columns] = table[dtype_columns].applymap(dispatch)
    return table

# Replace null values with ''
def prepare_cat(series):
    if any(series.isna()):
        if '' not in series.cat.categories:
            series.cat.add_categories('', inplace=True)
        series.fillna('', inplace=True)

# Extracts substrings with numbers, separator, and decimal
def extract_chunks(string, separator, decimal):
    candidates = regex.sub(f'[^\p{"N"}\-{separator+decimal}]', ' ', string).split()
    if not candidates:
        return []
    # Separate negative numbers by inserting and splitting at '-'
    else:
        results = []
        for candidate in candidates:
            if '-' not in candidate:
                results.append(candidate)
            else:
                # Indices where '-' exists
                ls = list(candidate)
                for idx in [i for i, x in enumerate(ls) if x == '-']:
                    ls.insert(idx, ' ')
                for signed_num in ''.join(ls).split():
                    results.append(signed_num)
        # If decimal comes before separator, split and flatten
        candidates = results
        results = []
        for candidate in candidates:
            if separator in candidate and decimal in candidate:
                s_idx = candidate.index(separator)
                d_idx = candidate.index(decimal)
                if s_idx > d_idx:
                    results.append(candidate[:s_idx])
                    results.append(candidate[s_idx+1:])
                    continue
            results.append(candidate)
        return results

def detect_number_with_separators(string, separator, decimal):
    result = regex.search(f'^\d{{1,3}}({separator}\d{{3}})*(\\{decimal}\d+)?$', string)
    if result:
        return result.group()
    else:
        return None

# Only for number conversion (not extract). Returns float, otherwise null if not a number.
def convert_as_is(string, separator, decimal, type_replace):
    if separator not in string:
        try:
            # Need to replace decimal (can be both . and ,) with '.'
            return float(string.replace(decimal, '.'))
        except ValueError:
            return type_replace
    else:
        result = detect_number_with_separators(string, separator, decimal)
        if result:
            return float(result.replace(separator, '').replace(decimal, '.'))
        else:
            return type_replace

# First finds candidates that contain separator only (int)
# Then returns first candidate that matches int, otherwise type_replace
def extract_int(string, separator, decimal, type_replace):
    candidates = extract_chunks(string, separator, decimal)
    for candidate in candidates:
        # Throw out candidates with decimal
        if decimal in candidate:
            continue
        # Return candidate if no separator
        elif separator not in candidate:
            return int(candidate)
        else:
            # Returns null if candidate format incorrect, decimal negligible since already thrown out
            result = detect_number_with_separators(candidate, separator, decimal)
            # If the whole string is in the correct format (1,000,000) return it.
            # Otherwise, return first int ex: 1 in '01,99'
            if result:
                return int(result.replace(separator, ''))
            else:
                return int(candidate.split(separator)[0])
    return type_replace

def extract_float(string, separator, decimal, type_replace):
    candidates = extract_chunks(string, separator, decimal)
    for candidate in candidates:
        # Throw out candidates without decimal
        if decimal not in candidate:
            continue
        # Return candidate if no separator
        elif separator not in candidate:
            return float(candidate.replace(decimal, '.'))
        else:
            # Returns null if candidate format incorrect, decimal negligible since already thrown out
            result = detect_number_with_separators(candidate, separator, decimal)
            # If the whole string is in the correct format (1,000,000) return it.
            # Otherwise, return first int ex: 1 in '01,99'
            if result:
                return float(result.replace(separator, '').replace(decimal, '.'))
            else:
                return float(candidate.split(separator)[-1].replace(decimal, '.'))
    return type_replace

def extract_any(string, separator, decimal, type_replace):
    candidates = extract_chunks(string, separator, decimal)
    for candidate in candidates:
        # Return candidate if no separator
        if separator not in candidate:
            return float(candidate.replace(decimal, '.'))
        else:
            # Returns null if candidate format incorrect, decimal negligible since already thrown out
            result = detect_number_with_separators(candidate, separator, decimal)
            # If the whole string is in the correct format (1,000,000) return it.
            # Otherwise, return first int ex: 1 in '01,99'
            if result:
                return float(result.replace(separator, '').replace(decimal, '.'))
            else:
                return float(candidate.split(separator)[0].replace(decimal, '.'))
    return type_replace
