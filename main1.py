from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for
import pandas as pd
import os
from werkzeug.utils import secure_filename
import re
import json
from openai import OpenAI
from flask_cors import CORS
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

load_dotenv()

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {str(e)}")
    client = None

# Create Blueprint for the chatbot
class ChatbotBlueprint:
    def __init__(self, name):
        self.name = name
        self.routes = []

    def route(self, route_str, **kwargs):
        def decorator(f):
            self.routes.append((route_str, f, kwargs))
            return f
        return decorator

    def register(self, app, url_prefix=''):
        for route_str, view_func, options in self.routes:
            endpoint = options.pop('endpoint', view_func.__name__)
            app.add_url_rule(url_prefix + route_str, endpoint, view_func, **options)

# Initialize the chatbot blueprint
chatbot_bp = ChatbotBlueprint('chatbot')

# Global variables for chatbot
current_df = None
current_filename = None
original_df = None

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv', 'json'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@chatbot_bp.route('/initialize', methods=['POST'])
def initialize_chatbot():
    """Initialize the chatbot with the dataframe from the main application"""
    global current_df, current_filename, original_df
    
    data = request.json
    if 'filename' in data:
        filename = data['filename']
        current_filename = filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            if filename.endswith('.csv'):
                current_df = pd.read_csv(filepath)
            elif filename.endswith('.json'):
                current_df = pd.read_json(filepath)
            else:
                current_df = pd.read_excel(filepath)
            
            original_df = current_df.copy()
            return jsonify({"status": "success", "message": "Chatbot initialized with dataframe"})
        except Exception as e:
            logger.error(f"Error loading dataframe: {str(e)}")
            return jsonify({"status": "error", "message": f"Error loading dataframe: {str(e)}"}), 400
    else:
        return jsonify({"status": "error", "message": "No filename provided"}), 400

@chatbot_bp.route('/chat', methods=['POST'])
def chat():
    """Process user input and return chatbot response"""
    global current_df, current_filename
    
    if current_df is None:
        return jsonify({"status": "error", "message": "No data loaded. Please upload a dataset first."}), 400
    
    data = request.json
    user_message = data.get('message', '').strip().lower()
    
    if not user_message:
        return jsonify({"status": "error", "message": "No message provided"}), 400
    
    # Handle greetings
    if any(greeting in user_message for greeting in ['hi', 'hello', 'hey']):
        return jsonify({
            "status": "success",
            "message": "Hello! I'm your data assistant. You can ask me about the dataset or give commands to modify it.",
            "dataframe": current_df.head(100).to_dict(orient='records'),
            "columns": current_df.columns.tolist(),
            "preview": current_df.head(5).to_dict(orient='records')
        })
    
    # Handle basic info questions
    if 'how many columns' in user_message:
        return jsonify({
            "status": "success",
            "message": f"The dataset has {len(current_df.columns)} columns: {', '.join(current_df.columns)}",
            "dataframe": current_df.head(100).to_dict(orient='records'),
            "columns": current_df.columns.tolist()
        })
    
    if 'how many rows' in user_message:
        return jsonify({
            "status": "success",
            "message": f"The dataset has {len(current_df)} rows",
            "dataframe": current_df.head(100).to_dict(orient='records'),
            "columns": current_df.columns.tolist()
        })

    # NEW COMMANDS ADDED BELOW
    if 'show summary' in user_message:
        summary = current_df.describe()
        response = "Dataset Summary:\n"
        response += f"1. {len(current_df.columns)} columns, {len(current_df)} rows\n"
        response += f"2. Numeric columns: {', '.join(current_df.select_dtypes(include=['number']).columns)}\n"
        response += f"3. Text columns: {', '.join(current_df.select_dtypes(include=['object']).columns)}\n"
        response += f"4. Total missing values: {current_df.isnull().sum().sum()}\n"
        response += f"5. Duplicate rows: {current_df.duplicated().sum()}"
        return jsonify({
            "status": "success",
            "message": response,
            "dataframe": current_df.head(100).to_dict(orient='records'),
            "columns": current_df.columns.tolist()
        })

    if 'null values' in user_message or 'missing values' in user_message:
        null_counts = current_df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        if len(null_cols) == 0:
            response = "No null/missing values found in the dataset."
        else:
            response = "Columns with null values:\n"
            for col, count in null_cols.items():
                response += f"- {col}: {count} nulls ({count/len(current_df)*100:.1f}%)\n"
        return jsonify({
            "status": "success",
            "message": response.strip(),
            "dataframe": current_df.head(100).to_dict(orient='records'),
            "columns": current_df.columns.tolist()
        })

    if 'duplicate values' in user_message or 'duplicate rows' in user_message:
        dup_count = current_df.duplicated().sum()
        if dup_count == 0:
            response = "No duplicate rows found in the dataset."
        else:
            response = f"Found {dup_count} duplicate rows ({dup_count/len(current_df)*100:.1f}% of data)"
        return jsonify({
            "status": "success",
            "message": response,
            "dataframe": current_df.head(100).to_dict(orient='records'),
            "columns": current_df.columns.tolist()
        })

    if 'data types' in user_message or 'column types' in user_message:
        dtypes = current_df.dtypes
        response = "Column Data Types:\n"
        for col, dtype in dtypes.items():
            response += f"- {col}: {dtype}\n"
        return jsonify({
            "status": "success",
            "message": response.strip(),
            "dataframe": current_df.head(100).to_dict(orient='records'),
            "columns": current_df.columns.tolist()
        })

    if 'show first' in user_message or 'show top' in user_message:
        try:
            n = int(re.search(r'\d+', user_message).group())
            n = min(n, 20)  # Limit to 20 rows for display
            return jsonify({
                "status": "success",
                "message": f"Showing first {n} rows",
                "dataframe": current_df.head(n).to_dict(orient='records'),
                "columns": current_df.columns.tolist()
            })
        except:
            return jsonify({
                "status": "success",
                "message": "Showing first 5 rows",
                "dataframe": current_df.head(5).to_dict(orient='records'),
                "columns": current_df.columns.tolist()
            })

    if 'show last' in user_message or 'show bottom' in user_message:
        try:
            n = int(re.search(r'\d+', user_message).group())
            n = min(n, 20)  # Limit to 20 rows for display
            return jsonify({
                "status": "success",
                "message": f"Showing last {n} rows",
                "dataframe": current_df.tail(n).to_dict(orient='records'),
                "columns": current_df.columns.tolist()
            })
        except:
            return jsonify({
                "status": "success",
                "message": "Showing last 5 rows",
                "dataframe": current_df.tail(5).to_dict(orient='records'),
                "columns": current_df.columns.tolist()
            })

    if 'unique values' in user_message and 'count' not in user_message:
        col_match = re.search(r'unique values in (.+)', user_message)
        if col_match:
            col_name = col_match.group(1).strip()
            if col_name in current_df.columns:
                unique_vals = current_df[col_name].unique()
                response = f"Unique values in '{col_name}':\n"
                response += f"Total: {len(unique_vals)}\n"
                response += f"Values: {', '.join(map(str, unique_vals[:10]))}"  # Show first 10 values
                if len(unique_vals) > 10:
                    response += "..."
                return jsonify({
                    "status": "success",
                    "message": response,
                    "dataframe": current_df.head(100).to_dict(orient='records'),
                    "columns": current_df.columns.tolist()
                })
            else:
                return jsonify({"status": "error", "message": f"Column '{col_name}' not found"})
        else:
            return jsonify({
                "status": "error",
                "message": "Please specify column like: 'unique values in column_name'"
            })

    if 'drop nulls' in user_message:
        initial_count = len(current_df)
        current_df = current_df.dropna()
        dropped = initial_count - len(current_df)
        response = f"Dropped {dropped} rows with null values. {len(current_df)} rows remaining."
        return jsonify({
            "status": "success",
            "message": response,
            "dataframe": current_df.head(100).to_dict(orient='records'),
            "columns": current_df.columns.tolist()
        })

    if 'remove duplicates' in user_message:
        initial_count = len(current_df)
        current_df = current_df.drop_duplicates()
        dropped = initial_count - len(current_df)
        response = f"Removed {dropped} duplicate rows. {len(current_df)} unique rows remaining."
        return jsonify({
            "status": "success",
            "message": response,
            "dataframe": current_df.head(100).to_dict(orient='records'),
            "columns": current_df.columns.tolist()
        })

    if 'reset data' in user_message:
        if original_df is not None:
            current_df = original_df.copy()
            return jsonify({
                "status": "success", 
                "message": "Data reset to original state",
                "dataframe": current_df.head(100).to_dict(orient='records'),
                "columns": current_df.columns.tolist(),
                "preview": current_df.head(5).to_dict(orient='records')
            })
        return jsonify({"status": "error", "message": "No original data to reset to"})
    
    # Handle basic commands
    if user_message.startswith('drop column'):
        col_name = user_message.replace('drop column', '').strip()
        if col_name in current_df.columns:
            current_df = current_df.drop(columns=[col_name])
            return jsonify({
                "status": "success",
                "message": f"Column '{col_name}' dropped successfully",
                "dataframe": current_df.head(100).to_dict(orient='records'),
                "columns": current_df.columns.tolist()
            })
        else:
            return jsonify({"status": "error", "message": f"Column '{col_name}' not found"})
    
    if user_message.startswith('rename column'):
        parts = user_message.replace('rename column', '').split(' to ')
        if len(parts) == 2:
            old_name = parts[0].strip()
            new_name = parts[1].strip()
            if old_name in current_df.columns:
                current_df = current_df.rename(columns={old_name: new_name})
                return jsonify({
                    "status": "success",
                    "message": f"Renamed column '{old_name}' to '{new_name}'",
                    "dataframe": current_df.head(100).to_dict(orient='records'),
                    "columns": current_df.columns.tolist()
                })
            else:
                return jsonify({"status": "error", "message": f"Column '{old_name}' not found"})
        else:
            return jsonify({"status": "error", "message": "Invalid rename format. Use: rename column OLD_NAME to NEW_NAME"})
    
    # If we get here, try using OpenAI if available
    if client:
        try:
            openai_response = get_openai_response(user_message)
            if openai_response.startswith(("drop column", "rename column", "filter rows", "sort by","show first 5 rows")):
                result = execute_command(openai_response)
            else:
                result = {
                    "status": "success",
                    "message": openai_response,
                    "dataframe": current_df.head(100).to_dict(orient='records'),
                    "columns": current_df.columns.tolist(),
                    "preview": current_df.head(5).to_dict(orient='records')
                }
            
            # Save changes if needed
            if result.get("status") == "success" and current_filename:
                try:
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], current_filename)
                    if current_filename.endswith('.csv'):
                        current_df.to_csv(filepath, index=False)
                    elif current_filename.endswith('.json'):
                        current_df.to_json(filepath, orient='records')
                    else:
                        current_df.to_excel(filepath, index=False)
                except Exception as e:
                    logger.error(f"Error saving file: {str(e)}")
            
            return jsonify(result)
        
        except Exception as e:
            logger.error(f"OpenAI error: {str(e)}")
            return jsonify({
                "status": "error",
                "message": "OpenAI service unavailable. Please use direct commands like 'drop column NAME' or 'rename column OLD to NEW'"
            })
    
    return jsonify({
        "status": "error",
        "message": "Command not understood. Try these commands:\n" +
                   "- show summary\n" +
                   "- null values\n" +
                   "- duplicate values\n" +
                   "- data types\n" +
                   "- show first 5\n" +
                   "- show last 5\n" +
                   "- unique values in [column]\n" +
                   "- drop nulls\n" +
                   "- remove duplicates\n" +
                   "- reset data\n" +
                   "- drop column [name]\n" +
                   "- rename column [old] to [new]"
    })

def get_openai_response(user_message):
    """Use OpenAI to interpret user's natural language request and either execute or respond"""
    try:
        prompt = f"""
        You are a senior data analyst assistant that can both answer questions about datasets and perform data manipulations.
        
        Current dataset columns: {current_df.columns.tolist() if current_df is not None else 'No dataset loaded'}
        
        For data manipulation commands, respond with ONLY the appropriate command that should be executed.
        For informational questions, respond with a helpful answer.
        
        Available commands formats:
        - "drop column NAME" - removes a column
        - "rename column OLD_NAME to NEW_NAME" - renames a column
        - "filter rows where COLUMN OPERATOR VALUE" - filters data (operators: ==, >, <, >=, <=, !=, contains, startswith, endswith)
        - "sort by COLUMN asc/desc" - sorts data
        - "add column NAME = EXPRESSION" - creates a new column
        - "show data types" - displays data types
        - "show summary" - shows statistical summary
        - "reset data" - resets to original data
        - "show nulls" - shows columns with null values
        - "drop nulls [COLUMN]" - drops rows with nulls (optional column)
        - "fill nulls [COLUMN] with [VALUE]" - fills nulls
        - "clean [COLUMN]" - cleans a specific column
        - "count unique values in COLUMN" - counts unique values
        - "convert COLUMN to datetime" - converts to datetime
        - "extract year/month/day from COLUMN into NEW_COLUMN" - extracts date parts
        - "fill missing values in COLUMN with VALUE" - fills missing values
        - "remove duplicate rows [based on COLUMN]" - removes duplicates
        - "group by COLUMN and calculate FUNCTION of COLUMN" - group by and aggregate
        - "apply uppercase/lowercase/capitalize to COLUMN" - text transformations
        - "replace OLD_VALUE with NEW_VALUE in COLUMN" - replaces values
        - "split COLUMN into NEW_COL1 and NEW_COL2 by DELIMITER" - splits columns
        - "merge COL1 and COL2 into NEW_COL with separator SEP" - merges columns
        - "calculate correlation between COL1 and COL2" - calculates correlation
        
        For informational questions (like 'what does this column mean?', 'how many rows are there?'), 
        provide a helpful answer based on the dataset.
        
        User request: {user_message}
        
        Response:
        """
        
        response = client.chat.completions.create(
            model="gpt-4",  # Using GPT-4 for better understanding
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return f"Could not process request: {str(e)}"

def execute_command(command):
    """Execute a data manipulation command on the dataframe"""
    global current_df, original_df
    
    try:
        # Normalize command (remove extra spaces, make lowercase)
        normalized_cmd = ' '.join(command.lower().split())
        
        # 1. Show summary statistics with null counts
        if normalized_cmd == "show summary":
            if current_df is None:
                return {"status": "error", "message": "No data loaded"}
            
            summary = current_df.describe().to_dict()
            null_counts = current_df.isnull().sum().to_dict()
            dtypes = current_df.dtypes.astype(str).to_dict()
            
            return {
                "status": "success",
                "message": "Summary statistics with null counts",
                "summary": summary,
                "null_counts": null_counts,
                "dtypes": dtypes,
                "preview": current_df.head(5).to_dict(orient='records')
            }
        
        # 2. Show columns with null values
        elif normalized_cmd == "show nulls":
            if current_df is None:
                return {"status": "error", "message": "No data loaded"}
            
            null_counts = current_df.isnull().sum()
            null_cols = null_counts[null_counts > 0].to_dict()
            total_nulls = null_counts.sum()
            
            if not null_cols:
                return {"status": "success", "message": "No columns contain null values"}
            
            return {
                "status": "success",
                "message": f"Found {total_nulls} null values across {len(null_cols)} columns",
                "null_columns": null_cols,
                "total_nulls": total_nulls
            }
        
        # 3. Drop rows with null values
        elif normalized_cmd.startswith("drop nulls"):
            if current_df is None:
                return {"status": "error", "message": "No data loaded"}
            
            parts = normalized_cmd.split()
            if len(parts) > 2:  # Specific column
                col_name = ' '.join(parts[2:])
                if col_name in current_df.columns:
                    initial_count = len(current_df)
                    current_df = current_df.dropna(subset=[col_name])
                    dropped = initial_count - len(current_df)
                    return {
                        "status": "success",
                        "message": f"Dropped {dropped} rows with nulls in column '{col_name}'",
                        "rows_remaining": len(current_df),
                        "column": col_name,
                        "nulls_remaining": current_df[col_name].isnull().sum()
                    }
                else:
                    return {"status": "error", "message": f"Column '{col_name}' not found"}
            else:  # All nulls
                initial_count = len(current_df)
                current_df = current_df.dropna()
                dropped = initial_count - len(current_df)
                return {
                    "status": "success",
                    "message": f"Dropped {dropped} rows with any null values",
                    "rows_remaining": len(current_df),
                    "remaining_nulls": current_df.isnull().sum().sum()
                }
        
        # 4. Fill null values
        elif normalized_cmd.startswith("fill nulls"):
            if current_df is None:
                return {"status": "error", "message": "No data loaded"}
            
            parts = normalized_cmd.split()
            if len(parts) >= 5 and parts[3] == "with":
                col_name = parts[2]
                fill_value = ' '.join(parts[4:])
                
                if col_name in current_df.columns:
                    null_count = current_df[col_name].isnull().sum()
                    if null_count == 0:
                        return {"status": "success", "message": f"No nulls to fill in column '{col_name}'"}
                    
                    # Try to convert fill_value to column's dtype
                    try:
                        if pd.api.types.is_numeric_dtype(current_df[col_name]):
                            fill_value = float(fill_value)
                        elif pd.api.types.is_bool_dtype(current_df[col_name]):
                            fill_value = fill_value.lower() == 'true'
                    except:
                        pass  # Keep as string if conversion fails
                    
                    current_df[col_name] = current_df[col_name].fillna(fill_value)
                    return {
                        "status": "success",
                        "message": f"Filled {null_count} nulls in '{col_name}' with '{fill_value}'",
                        "column": col_name,
                        "fill_value": str(fill_value),
                        "remaining_nulls": current_df[col_name].isnull().sum()
                    }
                else:
                    return {"status": "error", "message": f"Column '{col_name}' not found"}
            else:
                return {"status": "error", "message": "Invalid fill command format. Use: fill nulls [column] with [value]"}
        
        # 5. Reset to original data
        elif normalized_cmd == "reset data":
            if original_df is not None:
                current_df = original_df.copy()
                return {
                    "status": "success", 
                    "message": "Data reset to original",
                    "preview": current_df.head(5).to_dict(orient='records')
                }
            return {"status": "error", "message": "No original data to reset to"}
        
        # 6. Drop entire dataset
        elif normalized_cmd == "drop dataset":
            global current_filename
            current_df = None
            current_filename = None
            original_df = None
            return {"status": "success", "message": "Dataset cleared"}
        
        # 7. Show all columns
        elif normalized_cmd == "show columns":
            if current_df is None:
                return {"status": "error", "message": "No data loaded"}
            return {
                "status": "success",
                "message": f"Dataset has {len(current_df.columns)} columns",
                "columns": current_df.columns.tolist(),
                "dtypes": current_df.dtypes.astype(str).to_dict(),
                "null_counts": current_df.isnull().sum().to_dict()
            }
        
        # 8. Clean specific column
        elif normalized_cmd.startswith("clean "):
            if current_df is None:
                return {"status": "error", "message": "No data loaded"}
            
            col_name = normalized_cmd[6:].strip()
            if col_name in current_df.columns:
                cleaning_report = {}
                
                # String cleaning
                if pd.api.types.is_string_dtype(current_df[col_name]):
                    # Trim whitespace
                    initial_nulls = current_df[col_name].isnull().sum()
                    current_df[col_name] = current_df[col_name].str.strip()
                    current_df[col_name] = current_df[col_name].replace(r'^\s*$', pd.NA, regex=True)
                    new_nulls = current_df[col_name].isnull().sum() - initial_nulls
                    cleaning_report['trimmed_whitespace'] = True
                    cleaning_report['empty_strings_to_null'] = new_nulls
                
                # Numeric cleaning
                elif pd.api.types.is_numeric_dtype(current_df[col_name]):
                    # Convert to numeric, coercing errors
                    non_numeric = (~pd.to_numeric(current_df[col_name], errors='coerce').notna()).sum()
                    current_df[col_name] = pd.to_numeric(current_df[col_name], errors='coerce')
                    cleaning_report['converted_to_numeric'] = True
                    cleaning_report['non_numeric_values'] = non_numeric
                
                # Datetime cleaning
                elif pd.api.types.is_datetime64_any_dtype(current_df[col_name]):
                    # No specific cleaning for datetime yet
                    cleaning_report['datetime_checked'] = True
                
                return {
                    "status": "success",
                    "message": f"Cleaned column '{col_name}'",
                    "column": col_name,
                    "cleaning_report": cleaning_report,
                    "remaining_nulls": current_df[col_name].isnull().sum()
                }
            else:
                return {"status": "error", "message": f"Column '{col_name}' not found"}
        
        # 9. Drop column command
        elif (match := re.match(r"drop\s+column\s+(?:name\s*=\s*)?(.+)", normalized_cmd)):
            col_name = match.group(1).strip()
            if col_name in current_df.columns:
                current_df = current_df.drop(columns=[col_name])
                return {"status": "success", "message": f"Column '{col_name}' dropped successfully."}
            return {"status": "error", "message": f"Column '{col_name}' not found."}
        
        # 10. Rename column
        elif (match := re.match(r"rename\s+column\s+(.+)\s+to\s+(.+)", normalized_cmd)) or \
             (match := re.match(r"rename\s+(.+)\s+to\s+(.+)", normalized_cmd)):
            old_name = match.group(1).strip()
            new_name = match.group(2).strip()
            if old_name in current_df.columns:
                current_df = current_df.rename(columns={old_name: new_name})
                return {"status": "success", "message": f"Renamed '{old_name}' to '{new_name}'"}
            return {"status": "error", "message": f"Column '{old_name}' not found"}
        
        # 11. Filter rows
        elif (match := re.match(r"filter\s+rows\s+where\s+(.+)\s*(==|!=|>|<|>=|<=|contains|startswith|endswith)\s*(.+)", normalized_cmd)):
            col_name = match.group(1).strip()
            operator = match.group(2).strip()
            value = match.group(3).strip().strip('"\'')
            
            if col_name not in current_df.columns:
                return {"status": "error", "message": f"Column '{col_name}' not found"}
            
            try:
                # Try to convert value to match column type
                col_type = current_df[col_name].dtype
                if col_type == 'float64':
                    value = float(value)
                elif col_type == 'int64':
                    value = int(value)
                elif col_type == 'bool':
                    value = value.lower() == 'true'
            except:
                pass  # Keep as string if conversion fails
            
            if operator == '==':
                current_df = current_df[current_df[col_name] == value]
            elif operator == '!=':
                current_df = current_df[current_df[col_name] != value]
            elif operator == '>':
                current_df = current_df[current_df[col_name] > value]
            elif operator == '<':
                current_df = current_df[current_df[col_name] < value]
            elif operator == '>=':
                current_df = current_df[current_df[col_name] >= value]
            elif operator == '<=':
                current_df = current_df[current_df[col_name] <= value]
            elif operator == 'contains':
                current_df = current_df[current_df[col_name].astype(str).str.contains(value, case=False)]
            elif operator == 'startswith':
                current_df = current_df[current_df[col_name].astype(str).str.startswith(value, case=False)]
            elif operator == 'endswith':
                current_df = current_df[current_df[col_name].astype(str).str.endswith(value, case=False)]
                
            return {"status": "success", "message": f"Filtered rows where {col_name} {operator} {value}"}
        
        # 12. Sort data
        elif (match := re.match(r"sort\s+(?:by\s+)?(.+)\s+(asc|desc)", normalized_cmd)):
            col_name = match.group(1).strip()
            order = match.group(2).strip()
            if col_name in current_df.columns:
                current_df = current_df.sort_values(by=col_name, ascending=(order == 'asc'))
                return {"status": "success", "message": f"Data sorted by {col_name} in {order}ending order"}
            return {"status": "error", "message": f"Column '{col_name}' not found"}
        
        # 13. Add new column
        elif (match := re.match(r"add\s+column\s+(.+)\s*=\s*(.+)", normalized_cmd)):
            new_col = match.group(1).strip()
            expression = match.group(2).strip()
            
            try:
                # Handle simple arithmetic operations
                if expression.startswith('col:'):
                    # Reference another column
                    ref_col = expression[4:].strip()
                    if ref_col in current_df.columns:
                        current_df[new_col] = current_df[ref_col]
                        return {"status": "success", "message": f"Created new column '{new_col}' from column '{ref_col}'"}
                    return {"status": "error", "message": f"Column '{ref_col}' not found"}
                
                # Handle math operations between columns
                elif re.match(r".*[+\-*/].*", expression):
                    # Replace column names with df references
                    expr = expression
                    for col in current_df.columns:
                        expr = expr.replace(col, f"current_df['{col}']")
                    current_df[new_col] = eval(expr)
                    return {"status": "success", "message": f"Created new column '{new_col}' with expression: {expression}"}
                
                # Handle constant value
                else:
                    current_df[new_col] = expression
                    return {"status": "success", "message": f"Created new column '{new_col}' with value: {expression}"}
            except Exception as e:
                return {"status": "error", "message": f"Error creating column: {str(e)}"}
        
        # 14. Show data types
        elif re.match(r"show\s+data\s+types", normalized_cmd):
            dtypes = current_df.dtypes.apply(lambda x: x.name).to_dict()
            return {"status": "success", "message": "Data types:", "details": dtypes}
        
        # 15. Count unique values
        elif (match := re.match(r"count\s+unique\s+values\s+in\s+(.+)", normalized_cmd)):
            col_name = match.group(1).strip()
            if col_name in current_df.columns:
                count = current_df[col_name].nunique()
                return {"status": "success", "message": f"Found {count} unique values in '{col_name}'"}
            return {"status": "error", "message": f"Column '{col_name}' not found"}
        
        # 16. Convert to datetime
        elif (match := re.match(r"convert\s+(.+)\s+to\s+datetime", normalized_cmd)):
            col_name = match.group(1).strip()
            if col_name in current_df.columns:
                current_df[col_name] = pd.to_datetime(current_df[col_name], errors='coerce')
                return {"status": "success", "message": f"Converted '{col_name}' to datetime"}
            return {"status": "error", "message": f"Column '{col_name}' not found"}
        
        # 17. Extract year/month/day
        elif (match := re.match(r"extract\s+(year|month|day)\s+from\s+(.+)\s+into\s+(.+)", normalized_cmd)):
            part = match.group(1)
            col_name = match.group(2)
            new_col = match.group(3)
            
            if col_name in current_df.columns:
                if pd.api.types.is_datetime64_any_dtype(current_df[col_name]):
                    if part == 'year':
                        current_df[new_col] = current_df[col_name].dt.year
                    elif part == 'month':
                        current_df[new_col] = current_df[col_name].dt.month
                    elif part == 'day':
                        current_df[new_col] = current_df[col_name].dt.day
                    return {"status": "success", "message": f"Extracted {part} from '{col_name}' into '{new_col}'"}
                return {"status": "error", "message": f"Column '{col_name}' is not datetime type"}
            return {"status": "error", "message": f"Column '{col_name}' not found"}
        
        # 18. Fill missing values
        elif (match := re.match(r"fill\s+missing\s+values\s+in\s+(.+)\s+with\s+(.+)", normalized_cmd)):
            col_name = match.group(1).strip()
            fill_value = match.group(2).strip()
            
            if col_name in current_df.columns:
                try:
                    # Try to convert fill_value to match column type
                    col_type = current_df[col_name].dtype
                    if col_type == 'float64':
                        fill_value = float(fill_value)
                    elif col_type == 'int64':
                        fill_value = int(fill_value)
                    elif col_type == 'bool':
                        fill_value = fill_value.lower() == 'true'
                    
                    current_df[col_name] = current_df[col_name].fillna(fill_value)
                    return {"status": "success", "message": f"Filled missing values in '{col_name}' with {fill_value}"}
                except:
                    return {"status": "error", "message": f"Couldn't convert fill value to match column type"}
            return {"status": "error", "message": f"Column '{col_name}' not found"}
        
        # 19. Drop rows with missing values
        elif (match := re.match(r"drop\s+rows\s+with\s+missing\s+values\s+(?:in\s+(.+))?", normalized_cmd)):
            if match.group(1):
                col_name = match.group(1).strip()
                if col_name in current_df.columns:
                    current_df = current_df.dropna(subset=[col_name])
                    return {"status": "success", "message": f"Dropped rows with missing values in '{col_name}'"}
                return {"status": "error", "message": f"Column '{col_name}' not found"}
            else:
                current_df = current_df.dropna()
                return {"status": "success", "message": "Dropped all rows with missing values"}
        
        # 20. Group by and aggregate
        elif (match := re.match(r"group\s+by\s+(.+)\s+and\s+calculate\s+(.+)\s+of\s+(.+)", normalized_cmd)):
            group_col = match.group(1).strip()
            agg_func = match.group(2).strip()
            agg_col = match.group(3).strip()
            
            if group_col not in current_df.columns or agg_col not in current_df.columns:
                return {"status": "error", "message": "One or more specified columns not found"}
            
            try:
                if agg_func == 'mean':
                    result = current_df.groupby(group_col)[agg_col].mean()
                elif agg_func == 'sum':
                    result = current_df.groupby(group_col)[agg_col].sum()
                elif agg_func == 'count':
                    result = current_df.groupby(group_col)[agg_col].count()
                elif agg_func == 'min':
                    result = current_df.groupby(group_col)[agg_col].min()
                elif agg_func == 'max':
                    result = current_df.groupby(group_col)[agg_col].max()
                else:
                    return {"status": "error", "message": f"Unknown aggregation function: {agg_func}"}
                
                return {"status": "success", "message": f"Grouped by '{group_col}' and calculated {agg_func} of '{agg_col}'", "details": result.to_dict()}
            except Exception as e:
                return {"status": "error", "message": f"Error in grouping: {str(e)}"}
        
        # 21. Apply function to column
        elif (match := re.match(r"apply\s+(uppercase|lowercase|capitalize)\s+to\s+(.+)", normalized_cmd)):
            operation = match.group(1)
            col_name = match.group(2).strip()
            
            if col_name in current_df.columns:
                if operation == 'uppercase':
                    current_df[col_name] = current_df[col_name].astype(str).str.upper()
                elif operation == 'lowercase':
                    current_df[col_name] = current_df[col_name].astype(str).str.lower()
                elif operation == 'capitalize':
                    current_df[col_name] = current_df[col_name].astype(str).str.capitalize()
                return {"status": "success", "message": f"Applied {operation} to '{col_name}'"}
            return {"status": "error", "message": f"Column '{col_name}' not found"}
        
        # 22. Replace values
        elif (match := re.match(r"replace\s+(.+)\s+with\s+(.+)\s+in\s+(.+)", normalized_cmd)):
            old_val = match.group(1).strip()
            new_val = match.group(2).strip()
            col_name = match.group(3).strip()
            
            if col_name in current_df.columns:
                try:
                    # Try to convert values to match column type
                    col_type = current_df[col_name].dtype
                    if col_type == 'float64':
                        old_val = float(old_val)
                        new_val = float(new_val)
                    elif col_type == 'int64':
                        old_val = int(old_val)
                        new_val = int(new_val)
                    elif col_type == 'bool':
                        old_val = old_val.lower() == 'true'
                        new_val = new_val.lower() == 'true'
                    
                    current_df[col_name] = current_df[col_name].replace(old_val, new_val)
                    return {"status": "success", "message": f"Replaced '{old_val}' with '{new_val}' in '{col_name}'"}
                except:
                    return {"status": "error", "message": f"Couldn't convert values to match column type"}
            return {"status": "error", "message": f"Column '{col_name}' not found"}
        
        # 23. Extract substring
        elif (match := re.match(r"extract\s+first\s+(\d+)\s+characters\s+from\s+(.+)\s+into\s+(.+)", normalized_cmd)):
            length = int(match.group(1))
            col_name = match.group(2).strip()
            new_col = match.group(3).strip()
            
            if col_name in current_df.columns:
                current_df[new_col] = current_df[col_name].astype(str).str[:length]
                return {"status": "success", "message": f"Extracted first {length} characters from '{col_name}' into '{new_col}'"}
            return {"status": "error", "message": f"Column '{col_name}' not found"}
        
        # 24. Calculate correlation
        elif (match := re.match(r"calculate\s+correlation\s+between\s+(.+)\s+and\s+(.+)", normalized_cmd)):
            col1 = match.group(1).strip()
            col2 = match.group(2).strip()
            
            if col1 in current_df.columns and col2 in current_df.columns:
                try:
                    corr = current_df[col1].corr(current_df[col2])
                    return {"status": "success", "message": f"Correlation between '{col1}' and '{col2}': {corr:.2f}"}
                except:
                    return {"status": "error", "message": "Couldn't calculate correlation (non-numeric columns?)"}
            return {"status": "error", "message": "One or both columns not found"}
        
        # 25. Create dummy variables
        elif (match := re.match(r"create\s+dummy\s+variables\s+for\s+(.+)", normalized_cmd)):
            col_name = match.group(1).strip()
            
            if col_name in current_df.columns:
                dummies = pd.get_dummies(current_df[col_name], prefix=col_name)
                current_df = pd.concat([current_df, dummies], axis=1)
                return {"status": "success", "message": f"Created dummy variables for '{col_name}'", "new_columns": dummies.columns.tolist()}
            return {"status": "error", "message": f"Column '{col_name}' not found"}
        
        # 26. Remove duplicates
        elif (match := re.match(r"remove\s+duplicate\s+rows\s+(?:based\s+on\s+(.+))?", normalized_cmd)):
            if match.group(1):
                col_name = match.group(1).strip()
                if col_name in current_df.columns:
                    current_df = current_df.drop_duplicates(subset=[col_name])
                    return {"status": "success", "message": f"Removed duplicate rows based on '{col_name}'"}
                return {"status": "error", "message": f"Column '{col_name}' not found"}
            else:
                current_df = current_df.drop_duplicates()
                return {"status": "success", "message": "Removed all duplicate rows"}
        
        # 27. Change column type
        elif (match := re.match(r"change\s+(.+)\s+to\s+(numeric|string|datetime|boolean)", normalized_cmd)):
            col_name = match.group(1).strip()
            new_type = match.group(2).strip()
            
            if col_name in current_df.columns:
                try:
                    if new_type == 'numeric':
                        current_df[col_name] = pd.to_numeric(current_df[col_name], errors='coerce')
                    elif new_type == 'string':
                        current_df[col_name] = current_df[col_name].astype(str)
                    elif new_type == 'datetime':
                        current_df[col_name] = pd.to_datetime(current_df[col_name], errors='coerce')
                    elif new_type == 'boolean':
                        current_df[col_name] = current_df[col_name].astype(bool)
                    return {"status": "success", "message": f"Changed '{col_name}' to {new_type}"}
                except Exception as e:
                    return {"status": "error", "message": f"Couldn't convert column: {str(e)}"}
            return {"status": "error", "message": f"Column '{col_name}' not found"}
        
        # 28. Calculate value counts
        elif (match := re.match(r"count\s+values\s+in\s+(.+)", normalized_cmd)):
            col_name = match.group(1).strip()
            
            if col_name in current_df.columns:
                counts = current_df[col_name].value_counts().to_dict()
                return {"status": "success", "message": f"Value counts for '{col_name}':", "details": counts}
            return {"status": "error", "message": f"Column '{col_name}' not found"}
        
        # 29. Merge columns
        elif (match := re.match(r"merge\s+(.+)\s+and\s+(.+)\s+into\s+(.+)\s+with\s+separator\s+(.+)", normalized_cmd)):
            col1 = match.group(1).strip()
            col2 = match.group(2).strip()
            new_col = match.group(3).strip()
            separator = match.group(4).strip().replace('"', '').replace("'", "")
            
            if col1 in current_df.columns and col2 in current_df.columns:
                current_df[new_col] = current_df[col1].astype(str) + separator + current_df[col2].astype(str)
                return {"status": "success", "message": f"Merged '{col1}' and '{col2}' into '{new_col}'"}
            return {"status": "error", "message": "One or both columns not found"}
        
        # 30. Split column
        elif (match := re.match(r"split\s+(.+)\s+into\s+(.+)\s+and\s+(.+)\s+by\s+(.+)", normalized_cmd)):
            col_name = match.group(1).strip()
            new_col1 = match.group(2).strip()
            new_col2 = match.group(3).strip()
            delimiter = match.group(4).strip().replace('"', '').replace("'", "")
            
            if col_name in current_df.columns:
                split_data = current_df[col_name].astype(str).str.split(delimiter, n=1, expand=True)
                if len(split_data.columns) >= 2:
                    current_df[new_col1] = split_data[0]
                    current_df[new_col2] = split_data[1]
                    return {"status": "success", "message": f"Split '{col_name}' into '{new_col1}' and '{new_col2}'"}
                return {"status": "error", "message": f"Couldn't split column by '{delimiter}'"}
            return {"status": "error", "message": f"Column '{col_name}' not found"}
        
        # 31. Calculate moving average
        elif (match := re.match(r"calculate\s+(\d+)\s+day\s+moving\s+average\s+for\s+(.+)\s+into\s+(.+)", normalized_cmd)):
            window = int(match.group(1))
            col_name = match.group(2).strip()
            new_col = match.group(3).strip()
            
            if col_name in current_df.columns:
                try:
                    current_df[new_col] = current_df[col_name].rolling(window=window).mean()
                    return {"status": "success", "message": f"Calculated {window}-day moving average for '{col_name}'"}
                except:
                    return {"status": "error", "message": "Couldn't calculate moving average (non-numeric data?)"}
            return {"status": "error", "message": f"Column '{col_name}' not found"}
        
        # 32. Calculate cumulative sum
        elif (match := re.match(r"calculate\s+cumulative\s+sum\s+for\s+(.+)\s+into\s+(.+)", normalized_cmd)):
            col_name = match.group(1).strip()
            new_col = match.group(2).strip()
            
            if col_name in current_df.columns:
                try:
                    current_df[new_col] = current_df[col_name].cumsum()
                    return {"status": "success", "message": f"Calculated cumulative sum for '{col_name}'"}
                except:
                    return {"status": "error", "message": "Couldn't calculate cumulative sum (non-numeric data?)"}
            return {"status": "error", "message": f"Column '{col_name}' not found"}
        
        # 33. Calculate percentage change
        elif (match := re.match(r"calculate\s+percentage\s+change\s+for\s+(.+)\s+into\s+(.+)", normalized_cmd)):
            col_name = match.group(1).strip()
            new_col = match.group(2).strip()
            
            if col_name in current_df.columns:
                try:
                    current_df[new_col] = current_df[col_name].pct_change() * 100
                    return {"status": "success", "message": f"Calculated percentage change for '{col_name}'"}
                except:
                    return {"status": "error", "message": "Couldn't calculate percentage change"}
            return {"status": "error", "message": f"Column '{col_name}' not found"}
        
        # 34. Create bins
        elif (match := re.match(r"create\s+(\d+)\s+bins\s+for\s+(.+)\s+into\s+(.+)", normalized_cmd)):
            num_bins = int(match.group(1))
            col_name = match.group(2).strip()
            new_col = match.group(3).strip()
            
            if col_name in current_df.columns:
                try:
                    current_df[new_col] = pd.cut(current_df[col_name], bins=num_bins)
                    return {"status": "success", "message": f"Created {num_bins} bins for '{col_name}'"}
                except:
                    return {"status": "error", "message": "Couldn't create bins (non-numeric data?)"}
            return {"status": "error", "message": f"Column '{col_name}' not found"}
        
        # 35. Calculate z-score
        elif (match := re.match(r"calculate\s+z-score\s+for\s+(.+)\s+into\s+(.+)", normalized_cmd)):
            col_name = match.group(1).strip()
            new_col = match.group(2).strip()
            
            if col_name in current_df.columns:
                try:
                    current_df[new_col] = (current_df[col_name] - current_df[col_name].mean()) / current_df[col_name].std()
                    return {"status": "success", "message": f"Calculated z-score for '{col_name}'"}
                except:
                    return {"status": "error", "message": "Couldn't calculate z-score (non-numeric data?)"}
            return {"status": "error", "message": f"Column '{col_name}' not found"}
        
        # 36. Calculate rank
        elif (match := re.match(r"calculate\s+rank\s+for\s+(.+)\s+into\s+(.+)", normalized_cmd)):
            col_name = match.group(1).strip()
            new_col = match.group(2).strip()
            
            if col_name in current_df.columns:
                try:
                    current_df[new_col] = current_df[col_name].rank()
                    return {"status": "success", "message": f"Calculated rank for '{col_name}'"}
                except:
                    return {"status": "error", "message": "Couldn't calculate rank"}
            return {"status": "error", "message": f"Column '{col_name}' not found"}
        
        # 37. Sample random rows
        elif (match := re.match(r"get\s+random\s+(\d+)\s+rows", normalized_cmd)):
            n = int(match.group(1))
            if n > 0 and n <= len(current_df):
                sample = current_df.sample(n)
                return {"status": "success", "message": f"Random sample of {n} rows:", "sample_data": sample.to_dict(orient='records')}
            return {"status": "error", "message": f"Invalid number of rows requested (max: {len(current_df)})"}
        
        # 38. Calculate descriptive stats for column
        elif (match := re.match(r"show\s+stats\s+for\s+(.+)", normalized_cmd)):
            col_name = match.group(1).strip()
            
            if col_name in current_df.columns:
                if pd.api.types.is_numeric_dtype(current_df[col_name]):
                    stats = {
                        'mean': current_df[col_name].mean(),
                        'median': current_df[col_name].median(),
                        'min': current_df[col_name].min(),
                        'max': current_df[col_name].max(),
                        'std': current_df[col_name].std(),
                        'count': current_df[col_name].count()
                    }
                    return {"status": "success", "message": f"Statistics for '{col_name}':", "details": stats}
                return {"status": "error", "message": f"Column '{col_name}' is not numeric"}
            return {"status": "error", "message": f"Column '{col_name}' not found"}
        
        # Command not recognized
        else:
            return {"status": "error", "message": f"Command not understood: {command}"}
    
    except Exception as e:
        return {"status": "error", "message": f"Error processing command: {str(e)}"}

# Register the chatbot blueprint
chatbot_bp.register(app, url_prefix='/chatbot')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/modify')
def modify():
    filename = request.args.get('filename')
    if not filename:
        return redirect(url_for('index'))
    return render_template('modify.html')

@app.route('/upload-excel', methods=['POST'])
def upload_excel():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filename.endswith('.json'):
                df = pd.read_json(filepath)
            else:
                df = pd.read_excel(filepath)
            
            # Initialize the global dataframe for chatbot
            global current_df, current_filename, original_df
            current_df = df
            original_df = df.copy()
            current_filename = filename
            
            data = {
                'filename': filename,
                'columns': df.columns.tolist(),
                'sample_data': df.head(100).to_dict(orient='records'),
                'message': 'File uploaded and processed successfully'
            }
            
            return jsonify(data)
            
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/get-data', methods=['GET'])
def get_data():
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            df = pd.read_excel(filepath)
        
        return jsonify({
            'columns': df.columns.tolist(),
            'sample_data': df.head(100).to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-data-quality', methods=['POST'])
def analyze_data_quality():
    try:
        data = request.json
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
            
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Read the file
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # Calculate basic stats
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'duplicate_rows': len(df) - len(df.drop_duplicates()),
            'columns': {}
        }
        
        # Analyze each column
        for col in df.columns:
            col_stats = {
                'missing': df[col].isnull().sum(),
                'unique': df[col].nunique(),
                'dtype': str(df[col].dtype),
                'numeric': pd.api.types.is_numeric_dtype(df[col])
            }
            
            # Calculate numeric stats if applicable
            if col_stats['numeric']:
                col_stats['min'] = df[col].min()
                col_stats['max'] = df[col].max()
                col_stats['mean'] = df[col].mean()
                col_stats['std'] = df[col].std()
                
                # Calculate outliers using IQR method
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                col_stats['outliers'] = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            stats['columns'][col] = col_stats
        
        return jsonify({
            'status': 'success',
            'stats': stats,
            'sample_data': df.head(100).to_dict(orient='records')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-boxplot-data', methods=['POST'])
def get_boxplot_data():
    try:
        data = request.json
        filename = data.get('filename')
        columns = data.get('columns', [])
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
            
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Read the file
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # Filter only numeric columns
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
        if not numeric_cols:
            return jsonify({'status': 'error', 'message': 'No numeric columns found'})
        
        # Prepare boxplot data
        boxplot_data = {}
        for col in numeric_cols:
            values = df[col].dropna().tolist()
            if values:
                boxplot_data[col] = {
                    'values': values,
                    'min': min(values),
                    'q1': np.percentile(values, 25),
                    'median': np.median(values),
                    'q3': np.percentile(values, 75),
                    'max': max(values)
                }
        
        return jsonify({
            'status': 'success',
            'boxplot_data': boxplot_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/transform', methods=['POST'])
def transform_data():
    try:
        data = request.json
        transform_type = data.get('params', {}).get('type')
        filename = data.get('filename')
        params = data.get('params', {})
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
            
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Read the file
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # Apply transformations based on type
        if transform_type == 'drop':
            columns_to_drop = params.get('columns', [])
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
            
        elif transform_type == 'rename':
            old_name = params.get('old_name')
            new_name = params.get('new_name')
            if old_name and new_name and old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
                
        elif transform_type == 'filter':
            column = params.get('column')
            operator = params.get('operator')
            value = params.get('value')
            
            if column and operator and value is not None and column in df.columns:
                try:
                    # Try to convert value to match column type
                    if df[column].dtype == 'float64':
                        value = float(value)
                    elif df[column].dtype == 'int64':
                        value = int(value)
                    elif df[column].dtype == 'bool':
                        value = value.lower() == 'true'
                except:
                    pass  # Keep as string if conversion fails
                
                if operator == '==':
                    df = df[df[column] == value]
                elif operator == '!=':
                    df = df[df[column] != value]
                elif operator == '>':
                    df = df[df[column] > value]
                elif operator == '<':
                    df = df[df[column] < value]
                elif operator == '>=':
                    df = df[df[column] >= value]
                elif operator == '<=':
                    df = df[df[column] <= value]
                elif operator == 'contains':
                    df = df[df[column].astype(str).str.contains(value)]
                elif operator == 'startsWith':
                    df = df[df[column].astype(str).str.startswith(value)]
                elif operator == 'endsWith':
                    df = df[df[column].astype(str).str.endswith(value)]
                    
        elif transform_type == 'sort':
            column = params.get('column')
            ascending = params.get('ascending', True)
            
            if column and column in df.columns:
                df = df.sort_values(by=column, ascending=ascending)
                
        elif transform_type == 'aggregate':
            group_by = params.get('group_by')
            agg_column = params.get('column')
            agg_func = params.get('function')
            
            if group_by and agg_column and agg_func and group_by in df.columns and agg_column in df.columns:
                agg_funcs = {
                    'sum': 'sum',
                    'mean': 'mean',
                    'count': 'count',
                    'min': 'min',
                    'max': 'max'
                }
                if agg_func in agg_funcs:
                    df = df.groupby(group_by).agg({agg_column: agg_funcs[agg_func]}).reset_index()
                    
        elif transform_type == 'fillna':
            column = params.get('column')
            value = params.get('value')
            
            if column and column in df.columns and value is not None:
                try:
                    # Try to convert value to match column type
                    if df[column].dtype == 'float64':
                        value = float(value)
                    elif df[column].dtype == 'int64':
                        value = int(value)
                    elif df[column].dtype == 'bool':
                        value = value.lower() == 'true'
                except:
                    pass  # Keep as string if conversion fails
                
                df[column] = df[column].fillna(value)
                
        elif transform_type == 'type_conversion':
            column = params.get('column')
            new_type = params.get('new_type')
            
            if column and new_type and column in df.columns:
                try:
                    if new_type == 'numeric':
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                    elif new_type == 'string':
                        df[column] = df[column].astype(str)
                    elif new_type == 'datetime':
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                    elif new_type == 'boolean':
                        df[column] = df[column].astype(bool)
                except Exception as e:
                    return jsonify({'error': f'Could not convert column {column} to {new_type}: {str(e)}'}), 400
                    
        elif transform_type == 'dedupe':
            subset = params.get('subset')
            keep = params.get('keep', 'first')
            
            df = df.drop_duplicates(subset=subset if subset else None, keep=keep)
            
        elif transform_type == 'replace':
            column = params.get('column')
            to_replace = params.get('to_replace')
            value = params.get('value')
            
            if column and to_replace is not None and value is not None and column in df.columns:
                try:
                    # Try to convert values to match column type
                    if df[column].dtype == 'float64':
                        to_replace = float(to_replace)
                        value = float(value)
                    elif df[column].dtype == 'int64':
                        to_replace = int(to_replace)
                        value = int(value)
                    elif df[column].dtype == 'bool':
                        to_replace = to_replace.lower() == 'true'
                        value = value.lower() == 'true'
                except:
                    pass  # Keep as string if conversion fails
                
                df[column] = df[column].replace(to_replace, value)
        
        else:
            return jsonify({'error': f'Unknown transformation type: {transform_type}'}), 400
        
        # Update global dataframe for chatbot
        global current_df, current_filename
        current_df = df
        current_filename = filename
        
        # Save the modified file (overwrite the original)
        if filename.endswith('.csv'):
            df.to_csv(filepath, index=False)
        elif filename.endswith('.json'):
            df.to_json(filepath, orient='records')
        else:
            df.to_excel(filepath, index=False)
        
        return jsonify({
            'message': 'Transformation applied successfully',
            'filename': filename,
            'sample_data': df.head(100).to_dict(orient='records'),
            'columns': df.columns.tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download', methods=['GET'])
def download_file():
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Add routes for chatbot static files
@app.route('/chatbot/chatbot.js')
def chatbot_js():
    return """
    // Chatbot UI JavaScript
    document.addEventListener('DOMContentLoaded', function() {
        const chatbotButton = document.getElementById('modify-data-btn');
        if (chatbotButton) {
            chatbotButton.addEventListener('click', initChatbot);
        }
    });

    function initChatbot() {
        // Create chatbot UI if it doesn't exist
        if (!document.getElementById('chatbot-container')) {
            createChatbotUI();
        }
        
        // Show the chatbot
        document.getElementById('chatbot-container').style.display = 'block';
        
        // Initialize with current dataframe
        fetch('/chatbot/initialize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filename: window.currentFilename || null
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                addBotMessage('Hi! I\\'m your data assistant. How would you like to modify your data?');
            } else {
                addBotMessage('Error initializing chatbot: ' + data.message);
            }
        })
        .catch(error => {
            addBotMessage('Error connecting to server: ' + error.message);
        });
    }

    function createChatbotUI() {
        const chatbotHTML = `
            <div id="chatbot-container" class="chatbot-container">
                <div class="chatbot-header">
                    <h3>Modify Data Using Prompt</h3>
                    <button id="chatbot-close" class="chatbot-close-btn">&times;</button>
                </div>
                <div id="chatbot-messages" class="chatbot-messages"></div>
                <div class="chatbot-input">
                    <input type="text" id="chatbot-input-field" placeholder="Type your data command...">
                    <button id="chatbot-send">Send</button>
                </div>
            </div>
        `;
        
        // Add the HTML to the page
        const div = document.createElement('div');
        div.innerHTML = chatbotHTML;
        document.body.appendChild(div.firstChild);
        
        // Add event listeners
        document.getElementById('chatbot-close').addEventListener('click', function() {
            document.getElementById('chatbot-container').style.display = 'none';
        });
        
        document.getElementById('chatbot-send').addEventListener('click', sendMessage);
        document.getElementById('chatbot-input-field').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Add CSS for the chatbot
        const style = document.createElement('style');
        style.textContent = `
            .chatbot-container {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 350px;
                height: 500px;
                background: white;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.2);
                display: flex;
                flex-direction: column;
                overflow: hidden;
                z-index: 1000;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            }
            
            .chatbot-header {
                background: #10a37f;
                color: white;
                padding: 10px 15px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .chatbot-header h3 {
                margin: 0;
                font-size: 16px;
            }
            
            .chatbot-close-btn {
                background: none;
                border: none;
                color: white;
                font-size: 20px;
                cursor: pointer;
            }
            
            .chatbot-messages {
                flex-grow: 1;
                padding: 15px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            
            .chatbot-message {
                padding: 10px 15px;
                border-radius: 18px;
                max-width: 80%;
                word-break: break-word;
            }
            
            .bot-message {
                background: #f0f0f0;
                align-self: flex-start;
            }
            
            .user-message {
                background: #10a37f;
                color: white;
                align-self: flex-end;
            }
            
            .chatbot-input {
                display: flex;
                padding: 10px;
                border-top: 1px solid #e0e0e0;
            }
            
            .chatbot-input input {
                flex-grow: 1;
                padding: 10px;
                border: 1px solid #e0e0e0;
                border-radius: 20px;
                margin-right: 10px;
            }
            
            .chatbot-input button {
                background: #10a37f;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 20px;
                cursor: pointer;
            }
        `;
        document.head.appendChild(style);
    }

    function addUserMessage(message) {
        const messagesContainer = document.getElementById('chatbot-messages');
        const messageElement = document.createElement('div');
        messageElement.classList.add('chatbot-message', 'user-message');
        messageElement.textContent = message;
        messagesContainer.appendChild(messageElement);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function addBotMessage(message) {
        const messagesContainer = document.getElementById('chatbot-messages');
        const messageElement = document.createElement('div');
        messageElement.classList.add('chatbot-message', 'bot-message');
        messageElement.textContent = message;
        messagesContainer.appendChild(messageElement);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function sendMessage() {
        const inputField = document.getElementById('chatbot-input-field');
        const message = inputField.value.trim();
        
        if (message === '') return;
        
        // Add user message to chat
        addUserMessage(message);
        inputField.value = '';
        
        // Add temporary bot message
        addBotMessage('Processing...');
        
        // Send message to server
        fetch('/chatbot/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message
            })
        })
        .then(response => response.json())
        .then(data => {
            // Remove "Processing..." message
            const messages = document.getElementById('chatbot-messages');
            messages.removeChild(messages.lastChild);
            
            // Add bot response
            addBotMessage(data.message || 'Operation completed');
            
            // Update the dataframe in the main app if successful
            if (data.status === 'success' && data.dataframe) {
                // Update the table or data display
                updateDataDisplay(data.dataframe, data.columns);
            }
        })
        .catch(error => {
            // Remove "Processing..." message
            const messages = document.getElementById('chatbot-messages');
            messages.removeChild(messages.lastChild);
            
            // Add error message
            addBotMessage('Error: ' + error.message);
        });
    }

    function updateDataDisplay(data, columns) {
        // This function should update your main UI table or data display
        // Implement according to how your frontend displays data
        if (typeof updateDataTable === 'function') {
            updateDataTable(data, columns);
        } else {
            console.log('Data updated:', data);
            // You might want to dispatch a custom event here that your main app listens for
            const event = new CustomEvent('dataframeUpdated', { 
                detail: { 
                    dataframe: data,
                    columns: columns
                } 
            });
            document.dispatchEvent(event);
        }
    }
    """

@app.route('/chatbot/chatbot.css')
def chatbot_css():
    return """
    /* Additional CSS if needed */
    """

def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    app.run(debug=True)