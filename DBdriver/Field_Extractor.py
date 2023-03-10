





import pandas as pd
import os 
import sys

dir_name = os.path.dirname(__file__)
#sys.path.append(dir_name)

fieldsDesc = pd.read_csv(dir_name+'/'+'Field_Desc.csv')
print('Warning: The above will not work when setup.py generator is complete. \n Must include the csv file in the redirect.')

'''
NOTE: the fieldsDesc dataframe can be constructed directly from the database.
However, the queries are Sqlite-specific. Consider doing this after moving to MySql.
e.g. dbf.collect_from_DB('SELECT * FROM sqlite_schema WHERE type = \'table\'')
'''

def create_schema(df,table):

    collection = df[df['Table'] == table]
    fields = collection['Field'].tolist()
    types = collection['Type'].tolist()
    isKey = collection['Key'].tolist()
    Desc = collection['Description'].tolist()
    Obs = collection['Observations'].tolist()

    baseDict = dict.fromkeys(fields)
    out = {}
    for i in range(len(fields)):
        baseDict[fields[i]] = types[i]

    sql = 'create table {} ('.format(table)
    fieldData = {}
    for i in range(len(fields)):
        fieldData[fields[i]]= {}
        fieldData[fields[i]]['description'] = Desc[i]
        fieldData[fields[i]]['observation'] = Obs[i]
        if isKey[i] :
            #print(isKey[i])
            sql += '{} {} primary key, '.format(fields[i],types[i])
            out['key'] = fields[i]
        else: 
            sql += '{} {}, '.format(fields[i],types[i])
    sql = sql[0:-2]
    out['fields'] = fieldData
    out['sql_create_table'] = '{})'.format(sql)
    out['upload_template'] = baseDict

    return out

'''
def create_schema(table):
    #SELECT name FROM pragma_table_info('Uploads')
    sqlc = 'SELECT name FROM pragma_table_info(\'%s\')' % table
    print(sqlc)
    schema = dbf.collect_from_DB(sqlc)
'''
def create_all_schemas(df):
    tables = df['Table'].drop_duplicates().tolist()
    out = {}
    for table in tables:
        out[table] = {} 
        out[table] = create_schema(df,table)
           
    return out
if __name__=='__main__':
    
    tables = fieldsDesc['Table'].drop_duplicates().tolist()
    var2 = create_all_schemas(fieldsDesc)
    var2.keys()
