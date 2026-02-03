# Add imports for the new connectors
import os
import streamlit as st
from sql_execution import execute_query
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.loading import load_prompt
from pathlib import Path
from PIL import Image
from openrouter_chat import ChatOpenRouter, get_available_models

# Project root directory
current_dir = Path(__file__)
root_dir = current_dir.parent

# Ensure uploads directory exists and is empty at startup
uploads_dir = root_dir / "uploads"
uploads_dir.mkdir(exist_ok=True)
for item in uploads_dir.iterdir():
    try:
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            import shutil
            shutil.rmtree(item)
    except PermissionError:
        print(f"Skipped deleting {item} because it is in use.")
    except Exception as e:
        print(f"Error deleting {item}: {e}")

# Define default prompt templates
default_sql_template = PromptTemplate.from_template("""
You are a SQL query assistant. Convert the following natural language query into a SQL query.
Use standard SQL syntax and ensure the query is properly formatted.

User query: {input}

SQL query:
""")

default_pandas_template = PromptTemplate.from_template("""
Example query formats:

1. Simple column selection (no filter):
{{
    "operation": "select",
    "columns": ["Job Title", "Company Name"]
}}

2. Filtered query with string operation:
{{
    "operation": "filter",
    "columns": ["Job Title"],
    "filter": {{"Job Title": "Column.str.startswith(\"C\")"}}
}}

3. Filtered query with exact match:
{{
    "operation": "filter",
    "columns": ["Job Title", "Location"],
    "filter": {{"Location": "Column == \"New York\""}}
}}


You are a data analysis assistant. Convert the following natural language query into a Python dictionary representing a pandas query.
The dictionary should contain operations like 'filter', 'groupby', 'sort', 'columns', etc.

IMPORTANT RULES FOR COLUMN NAMES AND FILTERS:
1. Column Names:
   - ALWAYS enclose ALL column names in backticks (`), even if they don't contain spaces
   - Example: `Name`, `Age`, `Job Title`
   - Use EXACT column names as provided in the data

2. Filter Dictionary Format:
   For string operations, use this format:
   {{
   "operation": "filter",
   "columns": ["Column_Name"],
   "filter": {{
       "Column_Name": "Column.str.startswith(\"prefix\")"
   }}
   }}

   Multiple filters example:
   {{
   "operation": "filter",
   "columns": ["Department", "Job_Title"],
   "filter": {{
       "Department": "Column == \"HR\"",
       "Job_Title": "Column.str.contains(\"Manager\")"
   }}
   }}
   
   The word "Column" will be automatically replaced with the correct column reference.

3. String Operations:
   - For startswith: Column.str.startswith("prefix")
   - For contains: Column.str.contains("text")
   - For exact match: Column == "exact text"
   - ALWAYS use double quotes for string values

4. Numeric Operations:
   - Greater than: Column > number
   - Less than: Column < number
   - Equal to: Column == number

5. Date/Time Operations:
   - Month extraction: Column.dt.month == 1
   - Year extraction: Column.dt.year == 2023
   - Day extraction: Column.dt.day == 15
   - Date comparison: Column > "2023-01-01"

6. List Operations:
   - In list: Column.isin(["Value1", "Value2"])
   - ALWAYS use consistent quote style in lists

User query: {input}
Return ONLY the raw JSON string. Do NOT use markdown code blocks. Do NOT include any explanations.
Query (Python dictionary with 'operation', 'columns', 'filter', etc.):
""")

# Initialize prompt_template with default SQL template
prompt_template = default_sql_template

def write_to_training_file(file_path, prompt, sql):
     try:
          with open(file_path, 'a') as file:
               file.write("\n prompt : {}".format(prompt))
               file.write("\n sql : {}".format(sql))
               file.write("\n lable : 1 \n\n")
               return "success"
     except Exception as e:
          print(f"problem in opening file: {e}")
          return f"problem in opening file: {e}"

# Frontend
st.set_page_config(
    page_title="Query Assistant",
    page_icon="🌄"
)

# Fetch available models
@st.cache_data
def load_models():
    return get_available_models()

available_models = load_models()

# Model selection in sidebar
st.sidebar.title("Configuration")
selected_model = st.sidebar.selectbox(
    "Select Model",
    available_models,
    index=available_models.index("qwen/qwen3-30b-a3b:free") if "qwen/qwen3-30b-a3b:free" in available_models else 0
)

st.sidebar.success("Select a page above")

# Add data source selection
data_source = st.sidebar.selectbox(
    "Select Data Source",
    ["Snowflake", "MongoDB", "MySQL", "CSV/Excel", "Google Sheets"]
)

# Data source configuration based on selection
if data_source == "Snowflake":
    connector_type = "snowflake"
    
    # Snowflake connection parameters
    with st.sidebar.expander("Snowflake Connection", expanded=True):
        user = st.text_input("User", type="password")
        password = st.text_input("Password", type="password")
        account = st.text_input("Account")
        warehouse = st.text_input("Warehouse")
        database = st.text_input("Database")
        schema = st.text_input("Schema")
        role = st.text_input("Role")

    connector_params = {
        "user": user,
        "password": password,
        "account": account,
        "warehouse": warehouse,
        "database": database,
        "schema": schema,
        "role": role,
    }
    
    # Load the TPC-H prompt template
    prompt_template = load_prompt(f"{root_dir}/prompts/tpch_prompt.yaml")
    


elif data_source == "MongoDB":
    connector_type = "mongodb"
    
    # MongoDB connection parameters
    with st.sidebar.expander("MongoDB Connection", expanded=True):
        connection_string = st.text_input("Connection String", "mongodb://localhost:27017")
        database = st.text_input("Database", "sample_db")
    
    connector_params = {
        "connection_string": connection_string,
        "database": database
    }
    
    # Load MongoDB prompt template if exists, otherwise use a default
    prompt_file = f"{root_dir}/prompts/mongodb_prompt.yaml"
    if os.path.exists(prompt_file):
        prompt_template = load_prompt(prompt_file)
    else:
        # Use a simple default prompt for MongoDB
        prompt_template_text = """You are a MongoDB query assistant. Convert the following natural language query into a MongoDB query format.

User query: {input}

MongoDB query (return as a Python dictionary):"""
        prompt_template = PromptTemplate.from_template(prompt_template_text)

elif data_source == "MySQL":
    connector_type = "mysql"
    
    # MySQL connection parameters
    with st.sidebar.expander("MySQL Connection", expanded=True):
        host = st.text_input("Host", "localhost")
        port = st.number_input("Port", value=3306)
        user = st.text_input("User", "root")
        password = st.text_input("Password", "", type="password")
        database = st.text_input("Database", "sample_db")
    
    connector_params = {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": database
    }
    
    # Load MySQL prompt template if exists, otherwise use a default
    prompt_file = f"{root_dir}/prompts/mysql_prompt.yaml"
    if os.path.exists(prompt_file):
        prompt_template = load_prompt(prompt_file)
    else:
        # Use a simple default prompt for MySQL
        prompt_template_text = """You are a MySQL query assistant. Convert the following natural language query into a MySQL query.

User query: {input}

MySQL query:"""
        prompt_template = PromptTemplate.from_template(prompt_template_text)

elif data_source == "CSV/Excel":
    connector_type = "csv"  # Default to CSV, will be updated based on file extension
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        # Save the uploaded file
        file_path = f"{root_dir}/uploads/{uploaded_file.name}"
        os.makedirs(f"{root_dir}/uploads", exist_ok=True)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state['uploaded_file_path'] = file_path # Store file path in session state

        connector_params = {
            "file_path": file_path
        }
        
        # Determine connector type based on file extension
        if uploaded_file.name.endswith((".xlsx", ".xls")):
            connector_type = "excel"
        else:
            connector_type = "csv"

        st.session_state['connector_type'] = connector_type
        st.session_state['connector_params'] = connector_params
        
        # Show file preview
        with st.sidebar.expander("File Preview"):
            import pandas as pd
            if connector_type == "csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Auto-convert date columns for preview consistency
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col], errors='ignore')
                    except:
                        pass
            
            st.dataframe(df.head())
        
        # Use pandas prompt template for file operations with column information and data preview
        columns_info = ", ".join([f"`{col}`" for col in df.columns])
        
        # Get first few rows as string for context
        try:
            head_info = df.head(3).to_markdown(index=False)
        except ImportError:
            head_info = df.head(3).to_string(index=False)
            
        prompt_template = PromptTemplate.from_template(
            default_pandas_template.template + 
            "\n\nAvailable columns: " + columns_info + 
            "\n\nDataset Preview (first 3 rows):\n" + head_info +
            "\n\nNote: Use exact column names as shown above, including backticks. Infer data types from the preview."
        )
    else:
        st.warning("Please upload a CSV or Excel file")
        connector_params = {}

elif data_source == "Google Sheets":
    connector_type = "gsheets"
    
    # Google Sheets parameters
    with st.sidebar.expander("Google Sheets Connection", expanded=True):
        credentials_file = st.text_input("Credentials JSON File Path", f"{root_dir}/credentials.json")
        spreadsheet_id = st.text_input("Spreadsheet ID", "")
    
    connector_params = {
        "credentials_file": credentials_file,
        "spreadsheet_id": spreadsheet_id
    }
    st.session_state['connector_type'] = connector_type
    st.session_state['connector_params'] = connector_params
    
    # Use pandas prompt template for Google Sheets
    try:
        # Try to get column information from Google Sheets
        import pandas as pd
        from googleapiclient.discovery import build
        from google.oauth2 import service_account
        
        # Load credentials and create service
        credentials = service_account.Credentials.from_service_account_file(
            credentials_file,
            scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
        )
        service = build('sheets', 'v4', credentials=credentials)
        
        # Get sheet data (headers + first 3 rows)
        sheet = service.spreadsheets()
        result = sheet.values().get(
            spreadsheetId=spreadsheet_id,
            range='A1:Z4'  # Get first row for headers + 3 rows of data
        ).execute()
        
        # Extract headers and data
        if 'values' in result and result['values']:
            rows = result['values']
            headers = rows[0]
            data_rows = rows[1:] if len(rows) > 1 else []
            
            columns_info = ", ".join([f"`{col}`" for col in headers])
            
            # Format data preview
            preview_str = ""
            if data_rows:
                # Create a simple markdown-like table or just a list of rows
                # Using pandas for consistent formatting if available, or manual string formatting
                try:
                    df_preview = pd.DataFrame(data_rows, columns=headers)
                    preview_str = df_preview.to_markdown(index=False)
                except (ImportError, ValueError):
                    # Fallback if markdown not available or column mismatch
                    preview_str = "\n".join([str(row) for row in data_rows])
            
            prompt_template = PromptTemplate.from_template(
                default_pandas_template.template + 
                "\n\nAvailable columns: " + columns_info + 
                "\n\nDataset Preview (first 3 rows):\n" + preview_str +
                "\n\nNote: Use exact column names as shown above, including backticks. Infer data types from the preview."
            )
        else:
            prompt_template = default_pandas_template
    except Exception as e:
        print(f"Could not fetch Google Sheets columns: {str(e)}")
        prompt_template = default_pandas_template
        st.session_state['connector_type'] = None
        st.session_state['connector_params'] = {}

# Main interface
st.title(f"Your {data_source} Assistant")
prompt = st.text_input("Enter your query")

# Display sections


# Define tab titles and create tabs
tab_titles = ["Results", "Query"]
tabs = st.tabs(tab_titles)
results_display = tabs[0]
query_display = tabs[1]

if st.button("Generate Query and Execute"):
    if not prompt:
        st.warning("Please enter a query.")
    elif 'connector_type' not in st.session_state or 'connector_params' not in st.session_state:
        st.warning("Please configure the data source connection parameters")
    else:
        connector_type = st.session_state['connector_type']
        connector_params = st.session_state['connector_params']
        query_text = None
        try:
            try:
                # Initialize LLM with consistent configuration
                llm = ChatOpenRouter(
                    model_name=selected_model,
                    temperature=0.3,
                    max_tokens=500
                )
                
                # Create chain and generate query
                if not isinstance(prompt_template, PromptTemplate):
                    prompt_template = PromptTemplate.from_template(prompt_template)
                
                chain = prompt_template | llm
                response = chain.invoke({"input": prompt})
                query_text = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                error_msg = str(e)
                print(f"Error generating query: {error_msg}")
                if "api key" in error_msg.lower():
                    error_msg = "API authentication error. Please check your OpenAI API key configuration."
                elif "timeout" in error_msg.lower():
                    error_msg = "Request timed out. Please try again."
                elif "rate limit" in error_msg.lower():
                    error_msg = "API rate limit exceeded. Please wait a moment and try again."
                results_display.error(error_msg)
            
            # Execute query if we have valid connector parameters and a generated query
            if connector_params and query_text:
                try:
                    # Handle different query types based on data source
                    if data_source in ["CSV/Excel", "Google Sheets"]:
                        # Parse dictionary-style queries for pandas operations
                        try:
                            import json
                            import re
                            
                            # Clean up the response to extract JSON
                            clean_query_text = query_text.strip()
                            
                            # Robust JSON extraction: Find the first '{' and last '}'
                            start_idx = clean_query_text.find('{')
                            end_idx = clean_query_text.rfind('}')
                            
                            if start_idx != -1 and end_idx != -1:
                                clean_query_text = clean_query_text[start_idx:end_idx+1]
                            else:
                                # If no brackets found, try to fix common issues or fail gracefully
                                pass

                            # Try to handle common JSON syntax errors from LLMs
                            try:
                                query_dict = json.loads(clean_query_text)
                            except json.JSONDecodeError:
                                # Try to fix common issues
                                # 1. Replace single quotes with double quotes
                                fixed_text = clean_query_text.replace("'", '"')
                                # 2. Fix trailing commas
                                fixed_text = re.sub(r',\s*\}', '}', fixed_text)
                                fixed_text = re.sub(r',\s*\]', ']', fixed_text)
                                try:
                                    query_dict = json.loads(fixed_text)
                                except:
                                    # If repair fails, raise original error with context
                                    raise ValueError(f"Failed to parse JSON: {clean_query_text[:100]}...")

                            # Extract limit from user_question if present
                            limit_match = re.search(r'\b(?:top|any)\s+(\d+)\b', prompt, re.IGNORECASE)
                            if limit_match:
                                query_dict['limit'] = int(limit_match.group(1))
                            print(f"Processing pandas query: {query_dict}")
                            
                            # Ensure the query dictionary has required fields
                            if not isinstance(query_dict, dict):
                                raise ValueError("Query must be a valid dictionary")
                            
                            # Log query details for debugging
                            print(f"Query type: Pandas operation")
                            print(f"Data source: {data_source}")
                            print(f"Connector type: {connector_type}")
                            print(f"Query dictionary: {query_dict}")
                            
                            # Convert filter expressions to actual Python expressions
                            if 'filter' in query_dict:
                                if not query_dict['filter']:  # Handle empty filter
                                    query_dict['filter'] = None
                                else:
                                    # Handle both string and dictionary filter formats
                                    if isinstance(query_dict['filter'], dict):
                                        # Convert dictionary filter to string expression
                                        filter_items = []
                                        for col, expr in query_dict['filter'].items():
                                            # Clean and validate column name
                                            clean_col = col.strip('`')
                                            if clean_col not in df.columns:
                                                raise ValueError(f"Invalid column in filter: {clean_col}\nAvailable columns: {', '.join(df.columns)}")
                                            # Replace 'Column' placeholder with column name (wrapped in backticks for safety)
                                            expr = expr.replace('Column', f"`{clean_col}`")
                                            filter_items.append(expr)
                                        filter_expr = ' and '.join(filter_items) if filter_items else None
                                    elif isinstance(query_dict['filter'], str):
                                        # Clean up the filter expression and validate column references
                                        filter_expr = query_dict['filter']
                                        # Extract column references and validate them
                                        for col in df.columns:
                                            if f'`{col}`' in filter_expr:
                                                filter_expr = filter_expr.replace(f'`{col}`', col)
                                    else:
                                        filter_expr = None
                                    
                                    if filter_expr:
                                        print(f"Original filter expression: {filter_expr}")
                                        
                                        # Handle string operations like startswith
                                        if '.str.' in filter_expr:
                                            # Extract column name and operation
                                            parts = filter_expr.split('.str.')
                                            col_name = parts[0].strip()
                                            operation = parts[1].strip()
                                            print(f"Extracted column name: {col_name}")
                                            print(f"Extracted operation: {operation}")
                                            
                                            # Validate column name
                                            clean_col = col_name.strip('`')
                                            if clean_col not in df.columns:
                                                raise ValueError(f"Invalid column in string operation: {clean_col}\nAvailable columns: {', '.join(df.columns)}")
                                            
                                            # For startswith, we need to use str.startswith
                                            if 'startswith' in operation:
                                                # Extract the argument from startswith("X")
                                                try:
                                                    arg = operation.split('(')[1].strip(')')
                                                except IndexError:
                                                    raise ValueError(f"Invalid startswith operation format: {operation}\nExpected format: startswith(\"value\")")
                                                # Validate argument format
                                                if not (arg.startswith('"') and arg.endswith('"')):
                                                    raise ValueError(f"Invalid argument format in startswith: {arg}\nArgument must be enclosed in double quotes")
                                                
                                                # For startswith operations, format the expression for pandas query
                                                # Remove backticks and format as a valid pandas string operation
                                                clean_col = clean_col.strip('`')
                                                filter_expr = f"{clean_col}.str.startswith({arg})"
                                                # Update the query dictionary with the clean filter expression
                                                query_dict['filter'] = filter_expr
                                                print(f"Final filter expression: {filter_expr}")
                                            else:
                                                # For other string operations
                                                filter_expr = f"{clean_col}.str.{operation}"
                                                print(f"Formatted string operation expression: {filter_expr}")
                                        else:
                                            # Handle regular column references
                                            for col in query_dict.get('columns', []):
                                                clean_col = col.strip('`')
                                                if clean_col not in df.columns:
                                                    raise ValueError(f"Invalid column reference: {clean_col}\nAvailable columns: {', '.join(df.columns)}")
                                                filter_expr = filter_expr.replace(f"`{col}`", clean_col)
                                                print(f"Replaced column reference: {col} -> {clean_col}")
                                        
                                        query_dict['filter'] = filter_expr
                                        print(f"Final filter expression: {filter_expr}")
                            
                            # Validate query dictionary structure
                            required_fields = ['operation', 'columns']
                            missing_fields = [field for field in required_fields if field not in query_dict]
                            if missing_fields:
                                raise ValueError(f"Missing required fields in query dictionary: {', '.join(missing_fields)}")

                            # Validate column names before execution
                            for col in query_dict['columns']:
                                clean_col = col.strip('`')
                                if clean_col not in df.columns:
                                    raise ValueError(f"Invalid column name: {clean_col}\nAvailable columns: {', '.join(df.columns)}")

                            # Execute the pandas query with enhanced error handling
                            try:
                                # Validate query dictionary structure
                                if not isinstance(query_dict, dict):
                                    raise ValueError("Invalid query format: Expected a dictionary")
                                
                                limit = query_dict.get('limit')
                                results = execute_query(query_dict, connector_type=connector_type, **connector_params, limit=limit)
                                print(f"Query execution successful, displaying results")
                                
                                # Display results in the Results tab
                                with tabs[0]:
                                    if results.empty:
                                        st.info("Query executed successfully but returned no results.")
                                    else:
                                        st.dataframe(results)
                                        st.text(f"Total rows: {len(results)}")
                                
                                # Display query details
                                with st.expander("Query Details"):
                                    st.text("Executed Query:")
                                    st.json(query_dict)
                                    st.text("Data Source:")
                                    st.code(data_source)
                                    st.text("Filter Expression:")
                                    st.code(query_dict.get('filter', 'No filter applied'))
                                    st.text("Query Status:")
                                    st.success("Query executed successfully")
                            except Exception as e:
                                error_msg = str(e)
                                print(f"Error executing pandas query: {error_msg}")
                                
                                if "name" in error_msg.lower() and "not defined" in error_msg.lower():
                                    error_msg = f"Column reference error: Please check column names and formatting.\nAvailable columns: {', '.join(df.columns)}"
                                elif "invalid syntax" in error_msg.lower():
                                    error_msg = f"Invalid filter expression syntax: {query_dict.get('filter', '')}"
                                elif "cannot convert" in error_msg.lower():
                                    error_msg = "Data type mismatch in filter condition. Please check your filter values."
                                
                                results_display.error(error_msg)
                                
                                # Show debug information
                                with st.expander("Debug Information"):
                                    st.text("Error Type:")
                                    st.code(type(e).__name__)
                                    st.text("Full Error Message:")
                                    st.code(str(e))
                                    st.text("Query Dictionary:")
                                    st.json(query_dict)

                                
                            # Display the actual pandas expression for debugging
                            with st.expander("Debug Information"):
                                st.text("Pandas Query Expression:")
                                st.code(filter_expr if 'filter' in query_dict else "No filter applied", language="python")
                        except json.JSONDecodeError as e:
                            error_msg = f"Failed to parse query dictionary: {str(e)}\nPlease ensure the query is a valid JSON object."
                            print(error_msg)
                            results_display.error(error_msg)
                        except ValueError as e:
                            error_msg = str(e)
                            print(f"Validation error: {error_msg}")
                            results_display.error(error_msg)
                        except Exception as e:
                            error_msg = f"Error executing pandas query: {str(e)}\nPlease check your query format and try again."
                            print(error_msg)
                            results_display.error(error_msg)
                            # Log detailed error information
                            print(f"Detailed error: {type(e).__name__}: {str(e)}")
                            if hasattr(e, '__traceback__'):
                                import traceback
                                print(f"Traceback: {traceback.format_exc()}")
                    else:
                        # Execute SQL query with enhanced error handling
                        try:
                            # Basic SQL injection prevention
                            dangerous_keywords = ["DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT", "ALTER", "CREATE", "RENAME"]
                            if any(keyword in query_text.upper() for keyword in dangerous_keywords):
                                raise ValueError("Query contains potentially dangerous operations. Only SELECT queries are allowed.")
                            
                            print(f"Executing SQL query for {data_source}")
                            if connector_type in ["file_connector", "gsheets_connector"]:
                                 results = execute_query(query_dict, connector_type=connector_type, **connector_params)
                            else:
                                 results = execute_query(query_text, connector_type=connector_type, **connector_params)
                            
                            # Display results in the Results tab
                            with tabs[0]:
                                if results.empty:
                                    st.info("Query executed successfully but returned no results.")
                                else:
                                    st.dataframe(results)
                            
                            # Display the query in the Query tab
                            with tabs[1]:
                                st.code(query_text, language="sql")
                                
                            # Show query execution details
                            with st.expander("Query Details"):
                                st.text("Executed Query:")
                                st.code(query_text, language="sql")
                                st.text("Data Source:")
                                st.code(data_source)
                                st.text("Query Status:")
                                st.success("Query executed successfully")
                        except ValueError as e:
                            error_msg = str(e)
                            print(f"Validation error: {error_msg}")
                            results_display.error(error_msg)
                        except Exception as e:
                            error_msg = f"Error executing SQL query: {str(e)}\nPlease check your query syntax and try again."
                            print(error_msg)
                            results_display.error(error_msg)
                            # Log detailed error information
                            print(f"Detailed error: {type(e).__name__}: {str(e)}")
                            if hasattr(e, '__traceback__'):
                                import traceback
                                print(f"Traceback: {traceback.format_exc()}")
                                
                except Exception as e:
                    error_msg = f"Error executing query: {str(e)}"
                    print(f"Execution error: {error_msg}")
                    
                    # Provide more specific error messages based on error type
                    if "connection" in str(e).lower():
                        error_msg = f"Failed to connect to {data_source}. Please check your connection parameters and try again."
                    elif "authentication" in str(e).lower() or "credential" in str(e).lower():
                        error_msg = f"Authentication failed for {data_source}. Please verify your credentials."
                    elif "timeout" in str(e).lower():
                        error_msg = f"Connection timed out. Please check your network connection and try again."
                    elif "permission" in str(e).lower() or "access" in str(e).lower():
                        error_msg = f"Permission denied. Please check your access rights for {data_source}."
                    
                    results_display.error(error_msg)
                    
                    # Show technical details in an expander

# Process the prompt if provided

                    with st.expander("Technical Details"):
                        st.text("Error Type:")
                        st.code(type(e).__name__)
                        st.text("Error Message:")
                        st.code(str(e))
                        if hasattr(e, '__traceback__'):
                            st.text("Stack Trace:")
                            import traceback
                            st.code(traceback.format_exc())
        except Exception as e:
            error_msg = f"Failed to generate query: {str(e)}"
            print(f"Generation error: {error_msg}")
            
            # Provide more specific error messages for query generation issues
            if "api" in str(e).lower():
                error_msg = "Error connecting to the language model. Please try again later."
            elif "token" in str(e).lower():
                error_msg = "The query is too complex. Please try a simpler query."
            elif "format" in str(e).lower():
                error_msg = "Failed to format the query correctly. Please rephrase your request."
            
            results_display.error(error_msg)
            
            # Show technical details in an expander
            with st.expander("Technical Details"):
                st.text("Error Type:")
                st.code(type(e).__name__)
                st.text("Error Message:")
                st.code(str(e))
                if hasattr(e, '__traceback__'):
                    st.text("Stack Trace:")
                    import traceback
                    st.code(traceback.format_exc())
