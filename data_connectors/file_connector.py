import os
import pandas as pd
from typing import Dict, List, Union, Optional
from data_connectors.base_connector import DataConnector

class FileConnector(DataConnector):
    def __init__(self, file_path: str) -> None:
        self.file_path: str = file_path
        self.data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None
        self.file_type: str = os.path.splitext(file_path)[1].lower()
    
    def _convert_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attempt to convert object columns to datetime."""
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Try to convert to datetime, ignore errors to leave non-dates as is
                    df[col] = pd.to_datetime(df[col], errors='ignore')
                except Exception:
                    pass
        return df

    def connect(self) -> bool:
        """Connect to the file and load its contents into memory.
        
        Returns:
            bool: True if connection successful, False otherwise
        
        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file does not exist
            pd.errors.EmptyDataError: If file is empty
        """
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")
            
            if self.file_type == '.csv':
                self.data = pd.read_csv(self.file_path)
                self.data = self._convert_date_columns(self.data)
            elif self.file_type in ['.xlsx', '.xls']:
                self.data = pd.read_excel(self.file_path, sheet_name=None)
                # Convert dates for each sheet
                for sheet, df in self.data.items():
                    self.data[sheet] = self._convert_date_columns(df)
            else:
                raise ValueError(f"Unsupported file type: {self.file_type}")
            
            return True
        except Exception as e:
            print(f"Error connecting to file: {str(e)}")
            return False
    
    def execute_query(self, query: Dict[str, Union[str, List[str], bool, Dict]], limit: Optional[int] = None) -> pd.DataFrame:
        """Execute a query on the loaded data.
        
        Args:
            query: A dictionary containing query parameters:
                operation (str): Type of operation ('filter', 'groupby', 'sort')
                columns (List[str]): Columns to select
                filter (str): Filter condition in pandas query syntax or Python expression
                groupby (str): Column to group by
                agg (Dict): Aggregation functions to apply
                sort (str): Column to sort by
                ascending (bool): Sort order (default: True)
        
        Returns:
            pd.DataFrame: Result of the query operation
            
        Raises:
            ValueError: If filter expression is invalid
            KeyError: If specified columns don't exist
        
        Note:
            For string operations (startswith, contains, etc.), the column name should be wrapped in backticks
            AttributeError: If operation is not supported
        """
        # Clean column names by removing backticks
        def clean_column_name(col: str) -> str:
            # Remove backticks but preserve the column name for string operations
            if isinstance(col, str):
                if '.str.' in col:  # If it's a string operation, keep the backticks
                    return col
                return col.strip('`')
            return col
        try:
            if self.data is None:
                raise RuntimeError("No data loaded. Call connect() first.")
            
            # Get the DataFrame to operate on
            df = self.data if isinstance(self.data, pd.DataFrame) else next(iter(self.data.values()))
            
            # Validate and clean column names
            if 'columns' in query and query['columns']:
                clean_columns = [col.strip('`') for col in query['columns']]
                invalid_columns = [col for col in clean_columns if col not in df.columns]
                if invalid_columns:
                    raise ValueError(f"Invalid column names: {invalid_columns}\nAvailable columns: {', '.join(df.columns)}")
                df = df[clean_columns]
            
            # Apply filter if present
            if 'filter' in query and query['filter']:
                filter_expr = query['filter']
                if isinstance(filter_expr, dict):
                    filter_items = []
                    for col, expr in filter_expr.items():
                        clean_col = col.strip('`')
                        if clean_col not in df.columns:
                            raise ValueError(f"Invalid column name in filter: {clean_col}\nAvailable columns: {', '.join(df.columns)}")
                        expr = expr.replace('Column', f"{clean_col}")
                        filter_items.append(expr)
                    filter_expr = ' and '.join(filter_items)
                else:
                    # Handle string operations by evaluating the expression directly
                    if '.str.' in filter_expr:
                        # Remove backticks from the column name part for internal processing
                        clean_filter_expr = filter_expr.replace('`', '')
                        
                        # Extract column name and the string operation part
                        parts = clean_filter_expr.split('.str.', 1) # Split only on the first occurrence
                        col_name = parts[0].strip()
                        str_operation_with_args = parts[1].strip() # e.g., startswith("C") or contains("text")
                        
                        # Ensure column exists
                        if col_name not in df.columns:
                            raise ValueError(f"Invalid column in filter expression: {col_name}\nAvailable columns: {', '.join(df.columns)}")
                        
                        # Explicitly handle common string operations
                        if str_operation_with_args.startswith('startswith('):
                            # Extract the argument, e.g., "C" from startswith("C")
                            try:
                                arg = str_operation_with_args.split('(')[1].strip(')')
                            except IndexError:
                                raise ValueError(f"Invalid startswith operation format: {str_operation_with_args}\nExpected format: startswith(\"value\")")
                            
                            # Validate argument format (must be quoted string)
                            if not (arg.startswith('"') and arg.endswith('"')):
                                raise ValueError(f"Invalid argument format in startswith: {arg}\nArgument must be enclosed in double quotes")
                            
                            search_value = arg.strip('"')
                            df = df[df[col_name].astype(str).str.startswith(search_value)]
                        elif str_operation_with_args.startswith('contains('):
                            # Extract the argument, e.g., "text" from contains("text")
                            try:
                                arg = str_operation_with_args.split('(')[1].strip(')')
                            except IndexError:
                                raise ValueError(f"Invalid contains operation format: {str_operation_with_args}\nExpected format: contains(\"value\")")
                            
                            # Validate argument format (must be quoted string)
                            if not (arg.startswith('"') and arg.endswith('"')):
                                raise ValueError(f"Invalid argument format in contains: {arg}\nArgument must be enclosed in double quotes")
                            
                            search_value = arg.strip('"')
                            df = df[df[col_name].astype(str).str.contains(search_value)]
                        else:
                            raise ValueError(f"Unsupported string operation: {str_operation_with_args}")
                    else:
                        # This block is for non-.str. operations, like "Column == 'value'"
                        try:
                            df = df.query(filter_expr)
                        except Exception as e:
                            # If df.query fails, try direct eval with formatted column names.
                            try:
                                import re
                                def format_for_eval(expr, dataframe_columns):
                                    # Find all backticked column names and replace with df["column name"]
                                    for col in dataframe_columns:
                                        expr = re.sub(r'`' + re.escape(col) + r'`', f'df["{col}"]', expr)
                                    return expr
                                
                                formatted_expr = format_for_eval(filter_expr, df.columns)
                                mask = eval(formatted_expr, {'df': df, 'pd': pd})
                                df = df[mask]
                            except Exception as inner_e:
                                raise ValueError(f"Invalid filter expression: {filter_expr}. Error: {str(inner_e)}")
            
            # Apply groupby if present
            if 'groupby' in query and query['groupby']:
                groupby_col = self.clean_column_name(query['groupby'])
                if groupby_col not in df.columns:
                    raise KeyError(f"Groupby column '{groupby_col}' not found in data.")
                
                if 'agg' in query and query['agg']:
                    agg_funcs = query['agg']
                    df = df.groupby(groupby_col).agg(agg_funcs)
                else:
                    df = df.groupby(groupby_col).size().reset_index(name='count')
            
            # Apply sort if present
            if 'sort' in query and query['sort']:
                sort_col = self.clean_column_name(query['sort'])
                if sort_col not in df.columns:
                    raise KeyError(f"Sort column '{sort_col}' not found in data.")
                
                ascending = query.get('ascending', True)
                df = df.sort_values(by=sort_col, ascending=ascending)

            # Apply limit if present
            if limit is not None and isinstance(limit, int) and limit > 0:
                df = df.head(limit)
            
            return df
        
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            raise
    
    def get_schema(self) -> List[Dict[str, str]]:
        """Get the schema information for the loaded data.
        
        Returns:
            List[Dict[str, str]]: List of dictionaries containing column information:
                table: Sheet name for Excel, file name for CSV
                column: Column name
                type: Data type of the column
                
        Raises:
            RuntimeError: If data is not loaded
            AttributeError: If data structure is invalid
        """
        if self.data is None:
            if not self.connect():
                raise RuntimeError("Failed to connect to file")
        
        schema = []
        try:
            if isinstance(self.data, pd.DataFrame):
                # For CSV files
                file_name = os.path.basename(self.file_path)
                for col in self.data.columns:
                    schema.append({
                        'table': file_name,
                        'column': col,
                        'type': str(self.data[col].dtype)
                    })
            else:
                # For Excel files with multiple sheets
                for sheet_name, df in self.data.items():
                    for col in df.columns:
                        schema.append({
                            'table': sheet_name,
                            'column': col,
                            'type': str(df[col].dtype)
                        })
            return schema
        except Exception as e:
            print(f"Error getting schema: {str(e)}")
            raise
    
    def close(self) -> None:
        """Close the file connection and clean up resources.
        
        This method releases the DataFrame from memory to free up resources.
        """
        try:
            if self.data is not None:
                del self.data
                self.data = None
        except Exception as e:
            print(f"Error closing connection: {str(e)}")
