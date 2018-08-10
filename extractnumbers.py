# Extracts all non-negative numbers for now
def render(table, params):
    # if no column has been selected, return table
    if not params['colnames']:
        return table

    type_mapping = [None, 'int', 'float']
    text_before = params['text_before']
    text_after = params['text_after']
    type = params['type']
    columns = params['colnames'].split(',')
    columns = [c.strip() for c in columns]

    text_columns = table[columns].select_dtypes(include=['object', 'category']).columns

    if text_before or text_after:
        for col in text_columns:
            table[col] = table[col].apply(lambda x: extract_substring(x, type_mapping[type], text_before, text_after))
    else:
        for col in text_columns:
            table[col] = extract_number_only(table[col], type_mapping[type])

    return table

# Extracts the first number of type found, no int returns -1, else null
def extract_number_only(column, type):
    if type == 'int':
        return column.apply(lambda x: (find_numbers(x, type)) or -1).astype(np.int64)
    else:
        return column.apply(lambda x: find_numbers(x, type)).astype(np.float64)

# Extracts the substring based on before/after inputs
def extract_substring(string, type, text_before, text_after):
    number = find_numbers(string, type)
    if number:
        start = string.find(str(number))
        end = start + len(str(number))
        if text_before: start = 0
        if text_after: end = len(string) + 1
        return string[start:end]
    else:
        return ''

# Finds first number of type, null if none
def find_numbers(string, type=None):
    if not string:
        return None
    str_numbers = [x for x in re.sub('[^\d|\.|]', " ", string).split(' ') if x]
    if type == 'int':
        ints = [np.int64(x) for x in str_numbers if '.' not in x]
        if ints: return ints[0]
    elif type == 'float':
        floats = [np.float64(x) for x in str_numbers if '.' in x]
        if floats: return floats[0]
    elif str_numbers: return str_numbers[0]
    return None
