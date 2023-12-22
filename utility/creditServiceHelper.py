from quart import (
    Blueprint,
    current_app,
    jsonify,
    request
)
from azure.cosmos import CosmosClient, exceptions
from datetime import datetime
from azure.cosmos.errors import CosmosHttpResponseError
import uuid
import os
TOKEN_PER_CREDIT = os.getenv("TOKEN_PER_CREDIT")

def get_current_user_id_from_database():
    container = current_app.config['cosmos_db'].get_container_client("gi_users")

    # Query the Cosmos DB to get the last user record
    query = "SELECT TOP 1 c.id FROM c ORDER BY c.id DESC"
    result = container.query_items(query=query, enable_cross_partition_query=True)

    # Check if there are any records
    try:
        last_user = next(result)
        return int(last_user['id'])
    except StopIteration:
        # If no records found, return 0 as the starting user ID
        return 0
    except CosmosHttpResponseError as cosmos_error:
        print(f"Error querying Cosmos DB: {cosmos_error}")
        return 0
    
def get_current_trans_id_from_database():
    container_transactions = current_app.config['cosmos_db'].get_container_client("transactions")
    query = "SELECT TOP 1 c.surr_no FROM c ORDER BY c.surr_no DESC"
    result = container_transactions.query_items(query=query, enable_cross_partition_query=True)
    try:
        last_trans_id = next(result)
        return last_trans_id['surr_no']
    except StopIteration:
        return 0
    except CosmosHttpResponseError as cosmos_error:
        print(f"Error querying Cosmos DB for transaction ID: {cosmos_error}")
        return 0

def get_user_info(email):
    container_users = current_app.config['cosmos_db'].get_container_client("gi_users")
    query = f"SELECT TOP 1 c.role, c.id FROM c WHERE c.email = '{email}' ORDER BY c.id DESC"
    result = container_users.query_items(query=query, enable_cross_partition_query=True)

    try:
        user_info = next(result)
        return {"role": user_info["role"], "balance": calculate_balance(email)}
    except StopIteration:
        return {"role": None, "balance": 0}
    except CosmosHttpResponseError as cosmos_error:
        print(f"Error querying Cosmos DB for user info: {cosmos_error}")
        return {"role": None, "balance": 0}

def calculate_balance(email):
    # Fetching data from Cosmos DB transactions table
    container_transactions = current_app.config['cosmos_db'].get_container_client("transactions")
    query = f"SELECT * FROM transactions t WHERE t.email = '{email}' ORDER BY t.transaction_ts DESC"
    cosmos_transactions = container_transactions.query_items(
        query=query,
        enable_cross_partition_query=True
    )

    transactions = list(cosmos_transactions)

    # Processing Cosmos transactions
    user_credits = [transaction.get('credit', 0) for transaction in transactions]
    sum_debit = sum(transaction.get('debit', 0) for transaction in transactions)

    # Calculating balance
    balance = sum(user_credits) - sum_debit
    print(balance)
    roundbalance = round(balance,2)
    print(roundbalance)

    return roundbalance

def update_transactions_table(email, balance, service_type, token_usage=None, credit_used = 0, credit_assigned = 0):
    if service_type == "expired":
        transaction_type=2
    else:
        transaction_type=1

    container_transactions = current_app.config['cosmos_db'].get_container_client("transactions")

    try:
        current_trans_id = get_current_trans_id_from_database()
        print(f"current_trans_id: {current_trans_id}")  
        current_utc_datetime = datetime.utcnow()
        formatted_datetime = current_utc_datetime.strftime("%Y-%m-%d %H:%M:%S")
        # Add a new entry to transactions table with zero balance for the new user
        new_transaction = {
            "surr_no" : current_trans_id + 1,
            "id": str(uuid.uuid1()),
            "email": email,
            "credit": credit_assigned,
            "balance": balance,
            "debit": credit_used,
            "purchase_type": 1,
            "service_type": service_type,
            "transaction_type": transaction_type,  # Assuming 1 represents a user creation transaction
            "transaction_ts": formatted_datetime
        }
        # Include "token_usage" only if it's provided
        if credit_assigned is not 0:
            new_transaction["credit"] = credit_assigned
        if token_usage is not None:
            new_transaction["token_usage"] = token_usage
        if credit_used is not 0:
            new_transaction["debit"] = credit_used
        print("New Transaction:", new_transaction)
        container_transactions.create_item(body=new_transaction)

        # Fetch user's role and latest credit balance
        user_info = get_user_info(email)
        return {"message": "Transaction updated successfully", "role": user_info["role"], "balance": user_info["balance"]}

    except exceptions.CosmosHttpResponseError as cosmos_error:
        print(f"Error updating transactions table: {cosmos_error}")
        return {"message": f"Error updating transactions table: {cosmos_error}"}
    except Exception as e:
        print(f"Error updating transactions table: {str(e)}")
        return {"message": f"Error updating transactions table: {str(e)}"}
    

def check_balance(email):
    balance = calculate_balance(email)

    if balance > 0:
        # Continue with the remaining code
        return True
    else:
        # Insufficient balance, return a message
        return False

def get_user_data():
    user_container = current_app.config['cosmos_db'].get_container_client("gi_users")
    query = "SELECT * FROM c"
    result =  user_container.query_items(query, enable_cross_partition_query=True)
    #print("resultttt:",result)
    return result

def get_transaction_data():
    transaction_container = current_app.config['cosmos_db'].get_container_client("transactions")
    query = "SELECT * FROM c"
    result = transaction_container.query_items(query, enable_cross_partition_query=True)
    return result

def get_config_data():
    config_container = current_app.config['cosmos_db'].get_container_client("config")
    query = "SELECT * From c"
    result = config_container.query_items(query, enable_cross_partition_query=True)
    config_item = next(iter(result))
    return dict(config_item)

def update_config_data(new_config_data):
    try:
        config_container = current_app.config['cosmos_db'].get_container_client("config")
        
        # Assuming you have only one record in the 'config' container
        config_item_iterator = config_container.query_items(
            query="SELECT * FROM c",
            enable_cross_partition_query=True
        )
        config_item = next(config_item_iterator)

        # Update the existing config data with the new values
        #config_item["credit_pool_assigned"] = new_config_data.get("credit_pool_assigned", config_item["credit_pool_assigned"])
        config_item["credit_pool_balance"] = new_config_data.get("credit_pool_balance", config_item["credit_pool_balance"])
        config_item["credit_pool_balance"] = round(config_item["credit_pool_balance"], 2)
        #config_item["use_credit_pool"] = new_config_data.get("use_credit_pool", config_item["use_credit_pool"])

        # Replace the existing config item with the updated one
        config_container.replace_item(item=config_item, body=config_item)

    except Exception as e:
        print(f"Error updating config data: {str(e)}")

def credit_used_by_query(token_usage):
    try:
        credit_used = float(round(float(token_usage) / float(TOKEN_PER_CREDIT), 1))
        return credit_used
    except (ValueError, TypeError):
        print("Error: token_usage is not a valid integer")
        return 0.0

async def get_inactive_categories():
    container_users = current_app.config['cosmos_db'].get_container_client("gi_category")
    query = f"SELECT * FROM cat WHERE cat.status = 0"
    cat_list = list(container_users.query_items(query, enable_cross_partition_query=True))
    print("cat list to add in exclude category list:", cat_list)
    return cat_list