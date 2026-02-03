import snowflake.connector
import pandas as pd
from data_connectors.connector_factory import ConnectorFactory
from typing import Optional
from app_secrets import *

DEFAULT_CONNECTOR = "file_connector"

def execute_query(query, connector_type=DEFAULT_CONNECTOR, limit: Optional[int] = None, **connector_params):
    """Execute a query using the specified connector type."""
    connector = None
    try:
        print(f"sql_execution - Executing query with connector type: {connector_type}")
        print(f"sql_execution - Query parameters: {query}")
        
        # Create the appropriate connector using ConnectorFactory
        connector = ConnectorFactory.create_connector(connector_type, **connector_params)
        print(f"sql_execution - Created connector: {type(connector).__name__}")

        # Connect and execute query based on the selected connector
        print(f"sql_execution - Attempting to connect...")
        if not connector.connect():
            print(f"sql_execution - Failed to connect to {connector_type}")
            return pd.DataFrame({"error": [f"Failed to connect to {connector_type}"]})
        print(f"sql_execution - Successfully connected")

        print(f"sql_execution - Executing query...")
        resultdf = connector.execute_query(query, limit=limit)
        print(f"sql_execution - Query executed successfully")
        
        return resultdf

    except Exception as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame({"error": [f"Error: {e}"]})
    finally:
        if connector:
            try:
                connector.close()
                print(f"sql_execution - Connection closed")
            except Exception as e:
                print(f"Error closing connection: {e}")

# For backward compatibility
def execute_sf_query(sql, limit: Optional[int] = None):
    """Legacy function to maintain compatibility with existing code"""
    return execute_query(sql, connector_type='snowflake', limit=limit)


if __name__ == "__main__":
    # Snowflake query
    query = '''
            select n.n_name , count(*) as order_count from analytics.raw.orders o 
            inner join analytics.raw.customer c on o.o_custkey = c.c_custkey
            inner join analytics.raw.nation n on c.c_nationkey = n.n_nationkey
            group by n.n_name order by order_count desc limit 3
    '''
    execute_sf_query(query)
