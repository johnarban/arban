from calendar import c
import json
from pandas import DataFrame, concat, Series, isna
from . import sql_helpers as sql
import warnings

DEBUG = False

def __debug_print__(*args):
    """
    Print debug messages
    """
    if DEBUG:
        print(*args)
def __make_dict__(x):
    if isna(x):
        return {}
    else:
        return dict(x)
    

def __expand_subdict__(df_col):
    """
    Expand a column of dictionaries into a Series
    Drop any all nan columns
    """
    return df_col.apply(Series).dropna(axis=1, how='all')
    

def __values_are_dict_like__(col):
    """
    Check if a is a dictionary
    """
    if col.dtype == 'object':
        
        if hasattr(col,'str') and col.str.contains(r'\{.*\}').any():
            return True
        elif col.apply(lambda x: isinstance(x, dict)).any():
            return True
        else:
            return False
    else:
        return False

def __values_are_list_like__(col):
    """
    Check if any element in a column is a list
    """
    return col.apply(lambda x: isinstance(x, list)).any()

def __convertable_to_DataFrame__(x):
    """
    Check if a value can be expanded to a dataframe
    """
    return __values_are_dict_like__(x) or __values_are_list_like__(x)
    
def flatten(df, parent = '', delimiter = '.', fname = None,  tab = 0):
    """
    Recursively flat a pandas DataFrame with nested dictionaries
    """
    markdown_string = lambda col: '  ' * tab+ '- {}'.format(col.replace(parent,'').lstrip(delimiter)) + '\n'
    
    if isinstance(df, Series) or __values_are_list_like__(df):
        new_df = __expand_subdict__(df).add_prefix(parent + delimiter)
        return flatten(new_df, parent = parent, delimiter = delimiter,  fname = fname, tab = tab + 1)
    else:
        for col in df.columns:
            if fname is not None: fname.write(markdown_string(col))
            
            if __convertable_to_DataFrame__(df[col]):
                df = df.join(flatten(df[col], parent = col, delimiter = delimiter,  fname = fname, tab = tab + 1))
                df.drop(col, axis=1, inplace=True)

    return df  
    

    
def infer_schema(df, schema = {}):
    """
    Recursively find the structure of  a 
    pandas DataFrame with nested dictionaries
    Github: @johnarban
    """
    # if already a series or has a list of subcolum
    if isinstance(df, Series) or __values_are_list_like__(df):
        return infer_schema(__expand_subdict__(df), schema)
    else:
        for col in df.columns:
            schema[col] = {}
            if __convertable_to_DataFrame__(df[col]):
                infer_schema(df[col], schema[col])
    
    return schema 

def __get_child__(df, column):
    df.columns = df.columns.astype(str)
    return df[column]

def get_child_dataframe(df, parent, delimiter = '.'):
    """
    Recursively a subcolumn and return the child dataframe
    and prefix column name with the parent column name
    """
    __debug_print__('prefix', parent)
    child = df[parent]
    if not __convertable_to_DataFrame__(child):
        __debug_print__('has name', child._name)
        return DataFrame(child)
    else:
        return __expand_subdict__(df[parent]).add_prefix(parent + delimiter)


def get_column(df, column_specifier, delimiter = '.', depth = 0):
    """
    Recursively grab subcolumn based on 
    specifier with dot notation
    """
    __debug_print__('depth', depth, column_specifier)
    columns = column_specifier.split(delimiter)
    col = columns[0]
    if len(columns) == 1:
        if col in df.columns:
            return get_child_dataframe(df, col, delimiter)
        else:
            return df
    elif col in df.columns:
        if __convertable_to_DataFrame__(df[col]):  
            new_col_spec = delimiter.join(columns[1:])
            child = __expand_subdict__(df[col])
            df = get_column(child , new_col_spec, delimiter, depth + 1)
        else:
            return df[col]
    
    return df.add_prefix(col + delimiter)
     
def get_colspec_from_wildcard(df, column_specifier, delimiter = '.'):
    """
    Extract subcolumn based on 
    specifier with dot notation
    """
    arr = []
    
    def replace_wildcard(df, arr, column_specifier, delimiter = '.'):
        # debug_print('in function',column_specifier)
        if '*' not in column_specifier:
            arr.append(column_specifier)
        
        else:
            columns = column_specifier.split('*')
            # get the first column
            parent = columns[0].rstrip(delimiter)
            
            # get all the children of the first column
            if column_specifier.endswith('*'):
                children = flatten(get_column(df, parent, delimiter)).columns.str.replace('{}.'.format(parent),'')
            else:
                children = get_column(df, parent, delimiter).columns.str.replace('{}.'.format(parent),'')
            __debug_print__(parent, children)
            
            # debug_print(column_specifier, parent, children)
            
            # get the names of the columns that match the wildcard
            new_column_specs = [column_specifier.replace('*', str(child), 1) for child in children]
            
            # loop through the new column specs
            for new_column_spec in new_column_specs:
                replace_wildcard(df, arr, new_column_spec, delimiter)


            return None
        
    replace_wildcard(df, arr, column_specifier, delimiter)

    if len(arr) == 1:
        return arr[0]
    return arr


def __get_nested_dictionary_value__(df, colspec):
    """
    Extract subcolumn based on 
    specifier with dot notation
    """
    columns = colspec.split('.')
    if len(columns) == 1:
        return __expand_subdict__(df[columns[0]])
    else:
        for col in df.columns:
            if col == columns[0]:
                return __get_nested_dictionary_value__(df[col], '.'.join(columns[1:]))
    return df

def get_star(df, colspec, delimiter = '.', flat = False):
    """
    Extract subcolumn based on 
    specifier with dot notation
    """
    if '*' not in colspec:
        return get_column(df, colspec, delimiter)
    else:
        colspecs = get_colspec_from_wildcard(df, colspec, delimiter)
        if isinstance(colspecs, list):
            out = concat([get_column(df, colspec, delimiter) for colspec in colspecs], axis=1)
        else:
            out = get_column(df, colspecs, delimiter)
    
    if flat:
        return flatten(out)
    return out
# dataframe drop all nan columns
