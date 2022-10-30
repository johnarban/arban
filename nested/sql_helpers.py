import pandas as pd
import mysql.connector


def connect_to_db(database_dict = None):
    # connect to the database
    if database_dict is None:
        database_dict = {
            'host': 'localhost',
            'user': 'user',
            'password': 'password',
            'database': 'database'
            }
    mydb = mysql.connector.connect(
        host=database_dict["host"],
        user=database_dict["user"],
        passwd=database_dict["passwd"],
        database=database_dict["database"]
    )
    return mydb
    
# function to run an sql query and return the results as a pandas dataframe
def run_sql_query(sql_query, database_dict = None, return_df = True):
    # connect to the database
    mydb = connect_to_db(database_dict = database_dict)
    # run the query
    mycursor = mydb.cursor()
    mycursor.execute(sql_query)
    # get the results
    results = mycursor.fetchall()
    # get the column names
    col_names = [desc[0] for desc in mycursor.description]
    # close the connection
    mydb.close()
    # return the results as a pandas dataframe
    if return_df:
        return pd.DataFrame(results, columns=col_names, dtype=object)
    else:
        return results


def sql_query(num =1):
    if num == 1:
        sql_query = """
        SELECT student_id, story_state
        FROM cosmicds_db.StoryStates
        WHERE 
            student_id BETWEEN 0 AND 9999
        AND
            story_state LIKE '%Kernohan%'
        """
    else:
        sql_query = """
        SELECT student_id, story_state
        FROM cosmicds_db.StoryStates
        """
    return sql_query