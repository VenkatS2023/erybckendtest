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
from utility.creditServiceHelper import get_current_user_id_from_database, update_transactions_table, get_user_info, get_user_data, get_transaction_data, get_config_data, get_current_trans_id_from_database, update_config_data, calculate_balance

bp_user_credit = Blueprint("routes_user_credit", __name__)

@bp_user_credit.route("/user", methods=["POST"])
async def create_or_update_user():
    try:
        data = await request.get_json()
        current_user_id = get_current_user_id_from_database()
        user_id = str(current_user_id + 1)
        username = data.get("name")
        email = data.get("email")
        role = data.get("role", "General") # Default role is set to "general"
        access_token = data.get("accessToken")
        expiration_time = data.get("expirationTime")
        last_logged_in = data.get("lastLoggedIn")
 
        container = current_app.config['cosmos_db'].get_container_client("gi_users")
        query = "SELECT * FROM c"
        user_count = list(container.query_items(query=query, enable_cross_partition_query=True))
        print(len(user_count))
        if len(user_count)==0:
            role = "Admin"
        # Check if the user already exists
        query = f"SELECT * FROM c WHERE c.email = '{email}'"
        existing_users = list(container.query_items(query=query, enable_cross_partition_query=True))
        if existing_users:
            # If the user exists, update the access_token, expiration_time, and last_logged_in
            existing_user = existing_users[0]
            existing_user["access_token"] = access_token
            existing_user["expiration_time"] = expiration_time
            existing_user["last_logged_in"] = last_logged_in
 
            container.upsert_item(existing_user)
            balance = calculate_balance(email)
            response_data = {"message":"User updated successfully","role": existing_user.get("role"), "balance": balance, "status": existing_user.get("status")}         
            return jsonify(response_data)
           
        else:
            # If the user does not exist, create a new user
            new_user = {
                "id": user_id,
                "name": username,
                "email": email,
                "role": role,
                "access_token": access_token,
                "expiration_time": expiration_time,
                "last_logged_in": last_logged_in,
                "status": 1,
                "updated_by": None,
                "updated_at": None
            }
 
            container.create_item(body=new_user)
            update_transactions_table(email, credit_assigned=0, balance = 0, service_type = "New User")
            response_data = {"message":"User updated successfully","role": new_user.get("role"),  "balance": 0, "status": new_user.get("status")}       
            return jsonify(response_data)
 
    except CosmosHttpResponseError as cosmos_error:
        return {"message": f"Error occurred while creating or updating user: {cosmos_error}"}
    except Exception as e:
        return {"message": f"Unexpected error occurred: {str(e)}"}

@bp_user_credit.route("/update_transactions", methods=["POST"])
async def update_transactions():
    try:
        data = await request.get_json()
        
        if data is None:
            return jsonify({"message": "Invalid JSON data in the request"}), 400
        
        email = data.get("email")
        balance = data.get("balance")
        credit_assigned = data.get("credit_assigned")
        service_type = data.get("service_type")

        result = update_transactions_table(email, balance,credit_assigned, service_type)
        return jsonify(result)

    except CosmosHttpResponseError as cosmos_error:
        return {"message": f"Error occurred while updating transactions: {cosmos_error}"}
    except Exception as e:
        return {"message": f"Unexpected error occurred: {str(e)}"}

@bp_user_credit.route("/get_user_balance", methods=["GET"])
async def get_user_balance_api():
    try:
        email = request.args.get("email")
 
        if not email:
            return jsonify({"message": "Missing 'email' parameter in the request"}), 400
 
        user_info = get_user_info(email)
 
        return jsonify({
            "email": email,
            "balance": user_info["balance"],
            "role": user_info["role"]
        })
 
    except CosmosHttpResponseError as cosmos_error:
        return jsonify({"message": f"Error occurred while fetching transactions: {cosmos_error}"}), 500
    except Exception as e:
        return jsonify({"message": f"Unexpected error occurred: {str(e)}"}), 500

@bp_user_credit.route("/getusers/", methods=["GET"])

def getusers():
    try:
        # Fetch config values
        config_values = get_config_data()

        user_data = get_user_data()
        transaction_data = list(get_transaction_data())  # Convert iterator to a list
        container = current_app.config['cosmos_db'].get_container_client("gi_uploads")
        query = f"SELECT * FROM c"
        query_result = container.query_items(query=query, enable_cross_partition_query=True)
        uploaded_files = list(query_result)
        user_info_list = []
        for user in user_data:
            print(f"Processing transactions for user: {user['email']}")
            user_info = {
                "id": user["id"],
                "username": user["name"],
                "email": user["email"],
                "role": user["role"],
                "file_uploaded": 0,
                "query_count": 0,
                "credit_assigned": 0,
                "credit_revoked": 0,
                "credit_used": 0,
                "balance": 0,
                "transactions": [],
                "status": user["status"],
                "updated_at": user["updated_at"],
                "updated_by": user["updated_by"]
            }
            email = user["email"]
            for transaction_item in transaction_data:
                if transaction_item.get("email", "") == user["email"]:
                    try:
                        # Your existing transaction processing logic here...
                        if transaction_item["service_type"] == "Index":
                            user_info["credit_used"] += transaction_item.get("debit", 0)
                        elif transaction_item["service_type"] == "Assigned":
                            user_info["credit_assigned"] += transaction_item.get("credit", 0)
                        elif transaction_item["service_type"] == "Revoked":
                            user_info["credit_revoked"] += transaction_item.get("debit", 0)
                        elif transaction_item["service_type"] == "Query":
                            user_info["query_count"] += 1
                            user_info["credit_used"] += transaction_item["debit"]
                            user_info["unique_queries"] = list(set(transaction_item["id"] for transaction_item in transaction_data if transaction_item["service_type"] == "query"))

                    except KeyError as e:
                        print(f"KeyError: {e}. Skipping transaction.")
            for uploaded_item in uploaded_files:
                if uploaded_item.get("uploaded_by", "") == user["email"]:
                    user_info["file_uploaded"] += 1
            user_info["balance"] = calculate_balance(user["email"])
            user_info["credit_used"] = round(user_info["credit_used"], 2)
            user_info["credit_assigned"] = round(user_info["credit_assigned"], 2)
            user_info["credit_revoked"] = round(user_info["credit_revoked"], 2)
            user_info_list.append(user_info)

        # Create the final response dictionary
        response_dict = {
            "users": user_info_list,
            "config": {
                "credit_pool_assigned": config_values.get("credit_pool_assigned", 0),
                "credit_pool_balance": round(config_values.get("credit_pool_balance", 0), 2),
                "use_credit_pool": config_values.get("use_credit_pool", False),
            }
        }

        return jsonify(response_dict)
    except exceptions.CosmosHttpResponseError as cosmos_error:
        return jsonify({"error": cosmos_error.message}), cosmos_error.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500
        

@bp_user_credit.route("/updateroleandtransactions/", methods=["PUT"])
async def updateroleandtransactions():
    try:
        current_utc_datetime = datetime.utcnow()
        formatted_datetime = current_utc_datetime.strftime("%Y-%m-%d %H:%M:%S")
        data = await request.get_json()
        print("data:", data)
        container = current_app.config["cosmos_db"].get_container_client("gi_users")
        # Step 1: Get list of users with Admin role
        query = "SELECT c.email FROM gi_users c WHERE c.role = 'Admin'"
        admin_list = list(container.query_items(query, enable_cross_partition_query=True))
        count = len(admin_list)
        container2 = current_app.config["cosmos_db"].get_container_client("transactions")
        config_data = get_config_data()
        user_id = data.get("id")
        email = data.get("email")
        new_role = data.get("new_role")
        CreditAssigned = data.get("CreditAssigned")
        Credit_Revoked = data.get("Credit_Revoked")
        credit_pool_balance = config_data.get("credit_pool_balance", 0) or 0
        if credit_pool_balance is None:
            credit_pool_balance = 0
        print("credit pool balance:", credit_pool_balance)
        current_trans_id = get_current_trans_id_from_database()
        print(f"current_trans_id: {current_trans_id}")
        latest_balance = calculate_balance(email)
        print("latestbalncacee:",latest_balance)
        # Handle the case when the balance is 0 or not found
        if latest_balance is None:
            latest_balance = 0

        # Update user role if needed
        parameters = [{"name": "@user_id", "value": user_id}]
        items = list(
            container.query_items(
                f"SELECT * FROM gi_users c WHERE c.id = @user_id",
                parameters=parameters,
                enable_cross_partition_query=True,
            )
        )
        response={}
        if items:
            # Process financial transactions
            if CreditAssigned is not None and CreditAssigned > 0:
                CreditAssigned = round(CreditAssigned, 2)
                # Update credit pool balance if use_credit_pool is True
                if config_data.get("use_credit_pool", False):
                    if CreditAssigned <= credit_pool_balance:
                        # Calculate the new balance
                        new_balance = round(latest_balance + CreditAssigned, 2)

                        record = {
                            "surr_no": current_trans_id + 1,
                            "id": str(uuid.uuid4()),
                            "email": email,
                            "credit": CreditAssigned,
                            "balance": new_balance,
                            "debit": 0,
                            "purchase_type": 1,
                            "service_type": "Assigned",
                            "transaction_type": 1,
                            "transaction_ts": formatted_datetime
                            # Add other columns as needed
                        }
                        response = container2.create_item(body=record)
                        credit_pool_balance -= CreditAssigned
                        update_config_data({"credit_pool_balance": credit_pool_balance})
                    else:
                        return (
                            jsonify(
                                {"error": f"Credit assigned exceeds credit pool balance {credit_pool_balance}"}
                            ),
                            400,
                        )
                else:
                    new_balance = latest_balance + CreditAssigned

                    record = {
                        "surr_no": current_trans_id + 1,
                        "id": str(uuid.uuid4()),
                        "email": email,
                        "credit": CreditAssigned,
                        "balance": new_balance,
                        "debit": 0,
                        "purchase_type": 1,
                        "service_type": "Assigned",
                        "transaction_type": 1,
                        "transaction_ts": formatted_datetime
                        # Add other columns as needed
                    }
                    response = container2.create_item(body=record)

            elif CreditAssigned is not None and CreditAssigned < 0:
                return (
                    jsonify(
                        {"error": "Cannot assign credits less than zero"}
                        ),
                        400,
                    )

            if Credit_Revoked is not None and Credit_Revoked > 0:
                Credit_Revoked = round(Credit_Revoked, 2)
                # Ensure that Credit_Revoked is not Sgreater than the latest_balance
                if Credit_Revoked is not None and Credit_Revoked <= latest_balance:
                    # Calculate the new balance after revoking credit
                    new_balance_after_revoke = round(latest_balance - Credit_Revoked,2)
                    record1 = {
                        "surr_no": current_trans_id + 1,
                        "id": str(uuid.uuid4()),
                        "email": email,
                        "credit": 0,
                        "balance": new_balance_after_revoke,
                        "debit": Credit_Revoked,
                        "purchase_type": 1,
                        "service_type": "Revoked",
                        "transaction_type": 2,
                        "transaction_ts": formatted_datetime
                        # Add other columns as needed
                    }
                    response = container2.create_item(body=record1)
                    # print("response:::", response)

                    # Update credit pool balance if use_credit_pool is True
                    if config_data.get("use_credit_pool", False):
                        credit_pool_balance += Credit_Revoked
                        update_config_data({"credit_pool_balance": credit_pool_balance})
                else:
                    return (
                        jsonify({"error": "Insufficient credit balance to revoke"}),
                        400,
                    )
            elif Credit_Revoked is not None and Credit_Revoked < 0:
                return (
                    jsonify(
                        {"error": "Cannot revoke negative credits"}
                        ),
                        400,
                    )

            # Step 2: Check conditions and update role if needed
            service_type = response.get("service_type","")
            if any(user_dict.get("email") == email for user_dict in admin_list) and new_role == "General":
                print("count:", count)
                count -= 1  # Decrement count since we found an admin user with the required condition
                print("count after decrement:", count)
                if count > 0:
                    print("inside if condition")
                    # Perform the role update
                    document_to_update = items[0]
                    if document_to_update["role"] != new_role:
                        # Update the role only once
                        document_to_update["role"] = new_role
                        container.replace_item(document_to_update, document_to_update)
                        if service_type == "Revoked":
                            response["message"] = f"Role for user updated Successfully. New role: {new_role} and Credit Revoked."
                        elif service_type == "Assigned":
                            response["message"] = f"Role for user updated Successfully. New role: {new_role} and Credit Assigned."
                        else:
                            response["message"] = f"Role for user updated Successfully. New role: {new_role}"
                
                else:
                    print("inside else condition")
                    response["message"] = "User Role Not Updated. At least 1 admin is required in the system."
            if new_role == "Admin":
                document_to_update = items[0]
                if document_to_update["role"] != new_role:
                    # Update the role only once
                    document_to_update["role"] = new_role
                    container.replace_item(document_to_update, document_to_update)
                    if service_type == "Revoked":
                        response["message"] = f"Role for user updated Successfully. New role: {new_role} and Credit Revoked."
                    elif service_type == "Assigned":
                        response["message"] = f"Role for user updated Successfully. New role: {new_role} and Credit Assigned."
                    else:
                        response["message"] = f"Role for user updated Successfully. New role: {new_role}"

            if config_data.get("use_credit_pool", False):
                response["credit_pool_balance"] = round(credit_pool_balance, 2)
            return jsonify(response)

    except exceptions.CosmosHttpResponseError as cosmos_error:
        return jsonify({"error": cosmos_error.message}), cosmos_error.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp_user_credit.route("/transaction_details/", methods=["GET"])
 
def transaction_details():
    email = request.args.get("email")
    container_users = current_app.config['cosmos_db'].get_container_client("transactions")
    query = f"SELECT * FROM c WHERE c.email = '{email}' AND c.service_type != 'New User' ORDER BY c.transaction_ts DESC"
    items = list(container_users.query_items(query, enable_cross_partition_query=True))
 
    trans_info_list = []
    for transaction in items:
        print(f"Processing transactions for user: {transaction['email']}")
        trans_info = {
            "Reference Id": transaction["surr_no"],
            "Date": transaction["transaction_ts"],
            "Transaction Type": transaction["service_type"],
            "Credit Used" : transaction["debit"],
            "Credit Assigned" : transaction["credit"],
            "Balance" : transaction["balance"],
            "email" : transaction["email"]
        }
        trans_info_list.append(trans_info)
    response_dict = {
        "transactions": trans_info_list,
    }
 
    return jsonify(response_dict)        


@bp_user_credit.route("/update_user_status", methods=["PUT"])
async def update_user_status():
    try:
        data = await request.get_json()
        email = data.get("email")
        status = data.get("status")
 
        container = current_app.config['cosmos_db'].get_container_client("gi_users")
 
        # Check if the user exists
        query = f"SELECT * FROM c WHERE c.email = '{email}'"
        existing_users = list(container.query_items(query=query, enable_cross_partition_query=True))
 
        if not existing_users:
            return {"message": f"User with email {email} not found", "status": 404}
 
        existing_user = existing_users[0]
 
        # Update user status to 0 and set updated_by and updated_at
        existing_user["status"] = status
        existing_user["updated_by"] = email
        existing_user["updated_at"] = datetime.utcnow().isoformat()
 
        # Update the user in the container
        container.upsert_item(existing_user)
 
        balance = calculate_balance(email)
        response_data = {
            "message": "User status updated successfully",
            "role": existing_user.get("role"),
            "balance": balance,
            "status": existing_user.get("status"),
        }
        return jsonify(response_data)
    except CosmosHttpResponseError as cosmos_error:
        return {"message": f"Error occurred while updating user status: {cosmos_error}"}
    except Exception as e:
        return {"message": f"Unexpected error occurred: {str(e)}"}