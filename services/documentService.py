import re
from quart import (
	Blueprint,
	current_app,
	jsonify,
	request
)
import json
import uuid,logging
import os
import time
import traceback
from datetime import datetime
from werkzeug.utils import secure_filename
from utility.acshelper import qna_indexing
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.cosmos import CosmosClient, exceptions
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from utility.creditServiceHelper import calculate_balance

bp_doc = Blueprint("routes_doc", __name__, static_folder='static')
service_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]
azure_search_admin_key = os.environ["AZURE_SEARCH_ADMIN_KEY"]
azure_search_credential = AzureKeyCredential(azure_search_admin_key)
search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=azure_search_credential)



#This API fetches the list of uploaded files and returns it as a response.
@bp_doc.route("/uploadedFilesList/", methods=["GET"])
async def uploaded_files_list():
	try:
		# Extract email from the request
		args = request.args
		email = args.get('email')
		print("email_from_frontend:",email)
		balance = calculate_balance(email)

		# Fetch user details including role
		user_container = current_app.config['cosmos_db'].get_container_client("gi_users")
		user_query = f"SELECT * FROM c WHERE c.email = '{email}'"
		user_query_result = user_container.query_items(query=user_query, enable_cross_partition_query=True)
		user_details = list(user_query_result)
		# print("user_details:",user_details)

		if not user_details:
			return jsonify({"message": "User not found."})

		# Extract the role from user details
		user_role = user_details[0]['role']

		# Check if the user role is 'Admin'
		if user_role == 'Admin':
			# Fetch all uploads
			uploads_container = current_app.config['cosmos_db'].get_container_client("gi_uploads")
			upload_query = "SELECT * FROM c"
			query_result = uploads_container.query_items(query=upload_query, enable_cross_partition_query=True)
		else:
			# Fetch uploads based on the user's email
			uploads_container = current_app.config['cosmos_db'].get_container_client("gi_uploads")
			upload_query = f"SELECT * FROM c WHERE c.uploaded_by = '{email}'"
			query_result = uploads_container.query_items(query=upload_query, enable_cross_partition_query=True)
		uploaded_files = list(query_result)
		response_data = {"Files": uploaded_files, "balance": balance }
		return jsonify(response_data)

	except exceptions.CosmosHttpResponseError as e:
		# Handle Cosmos DB HTTP response errors
		return jsonify({"message": f"Cosmos DB error: {str(e)}"})
	except Exception as e:
		# Handle other exceptions
		return jsonify({"message": f"Error occurred: {str(e)}"})



@bp_doc.route("/addCategory/")
async def add_category():
	args = request.args
	category_code = args.get('category_code')
	category_name = args.get('category_name')
	container = current_app.config['cosmos_db'].get_container_client("gi_category")
	try:
		user_name = 'super-admin'
		query = "SELECT * FROM gi_category r WHERE r.category_name = @category_name"
		query_params = [{"name": "@category_name", "value": str(category_name)}]

		# existing_category = db.session.query(gi_category).filter(gi_category.category_name == category_name).first()
		existing_category = None
		query_result = container.query_items(
			query=query,
			parameters=query_params,
			enable_cross_partition_query=True
		)

		for item in query_result:
			existing_category = item

		if existing_category is not None:
			return {"message": "Category already exists"}

		# new_category = gi_category(category_code= category_code, category_name=category_name, created_by=user_name, created_at=datetime.now())
		# db.session.add(new_category)
		# db.session.commit()

		category_id = uuid.uuid4()
		item = {
			'id': str(category_id),
			'category_code': category_code,
			'category_name': category_name,
			'created_by': "super-admin",
			'created_at': str(datetime.now()),
			'status': 1
		}
		container.create_item(body=item)
		return {"message": "Category added successfully"}
	except Exception as e:
		return {"message": "Error occurred while updating category: " + str(e)}


@bp_doc.route("/getCategories/")
async def get_categories():
	args = request.args
	status_flag = int(args.get('status_flag'))
	container = current_app.config['cosmos_db'].get_container_client("gi_category")

	# status_flag = 0
	# if status_flag_request == True:
	# 	status_flag = 1

	category_list = None

	if status_flag == 1:
		query = "SELECT * FROM gi_category r WHERE r.status = @status_flag"
		query_params = [{"name": "@status_flag", "value": status_flag}]

		query_result = container.query_items(
			query=query,
			parameters=query_params,
			enable_cross_partition_query=True
		)
		category_list = list(query_result)
	else:
		query = "SELECT * FROM gi_category r"

		query_result = container.query_items(
			query=query,
			# parameters=query_params,
			enable_cross_partition_query=True
		)

		category_list = list(query_result)
	if category_list:
		return {"CategoryList": category_list}
	else:
		return {"CategoryList": []}


def insert_file_in_db(files, files_exist, user_name, remarks, index_format, category_id):
	dir_list = []
	for file in files:
		file_path = os.path.join(os.getcwd(), 'uploads', secure_filename(file.filename))
		file_name= file.filename
		file_id = uuid.uuid4()
		files_exist.append(file.filename)

		print(file_path)
		file.save(file_path)
		# with open(file_path, "wb") as buffer:
		# 	shutil.copyfileobj(file.file, buffer)
		file_info = os.stat(file_path)
		file_size = file_info.st_size
		dir_list.append(file_path)
		# insert_dataset_file_query(file_id, file_name, file_size, user_name, remarks, index_format, category_id)

		container = current_app.config['cosmos_db'].get_container_client("gi_uploads")

		item = {
			'id': str(file_id),
			'file_name': file_name,
			'file_size': file_size,
			'uploaded_by': "super-admin",
			'uploaded_at': str(datetime.now()),
			'chunk_ids': '',
			'token_used': '',
			'credit_used': '',
			'category_id': category_id,
			'ex_time': time.time(),
			'status': 0
		}
		container.create_item(body=item)

	return dir_list

def upload_files_to_blob_storage(files_to_upload):
	# logger.info(f"Upload file to bolo started {str(files_to_upload)}")
	# Create a blob_service_client
	# blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
	container_client = current_app.config['blob_container_client']

	# files_to_upload = list_files_in_directory(source_directory)

	for file_to_upload in files_to_upload:
		# source_file_path = os.path.join(source_directory, file_to_upload)

		# Use the filename as the blob name
		blob_name = os.path.basename(file_to_upload)
		# logger.info(f"Blob Name {str(blob_name)}")
		# Get the BlobClient for the file
		blob_client = container_client.get_blob_client(blob_name)
		# logger.info(f"blob_client {str(blob_client)}")
		# Upload the file to Blob Storage
		with open(file_to_upload, "rb") as data:
			blob_client.upload_blob(data, overwrite=True)

		# logger.info(f"Uploaded {blob_name} to Azure Blob Storage.")

@bp_doc.route("/uploadFile", methods=["POST"])
async def create_upload_file():
	try:
		files = (await request.files).getlist('files')
		check_file = current_app.config['cosmos_db'].get_container_client("gi_uploads")
		query = f"SELECT c.file_name FROM gi_uploads c"
		file_list = list(check_file.query_items(query, enable_cross_partition_query=True))
		# print("file_list:", file_list)

		dup_list = []
		for file in files:
			file_name = file.filename.replace(" ", "_").replace("'", "")
			for file_item in file_list:
				existing_file_name = file_item['file_name']
				if existing_file_name == file_name:
					dup_list.append(file_name)

		# print("dup_list:", dup_list)

		request_form = (await request.form)
		email = request_form.get("email")
		remarks = request_form.get('remarks')
		category_id = request_form.get('category_id')

		files_exist = []
		uploaded_files = []
		not_uploaded_files = []
		index_format = 'acs'

		dir_list = []
		for file in files:
			file_name = file.filename.replace(" ","_").replace("'","")
			file_id = uuid.uuid4()
			files_exist.append(file.filename)

			try:
				if not dup_list or file_name not in dup_list:
					uploaded_files.append(file_name)

					file_path = os.path.join(os.getcwd(), 'uploads', secure_filename(file.filename))
					print(file_path)
					await file.save(file_path)

					file_info = os.stat(file_path)
					file_size = file_info.st_size
					dir_list.append(file_path)

					container = current_app.config['cosmos_db'].get_container_client("gi_uploads")

					item = {
						'id': str(file_id),
						'file_name': file_name,
						'file_size': file_size,
						'uploaded_by': email,
						'uploaded_at': str(datetime.now()),
						'chunk_ids': '',
						'token_used': '',
						'credit_used': '',
						'category_id': category_id,
						'ex_time': time.time(),
						'status': 0
					}
					container.create_item(body=item)

					print("dir_list:", dir_list)
					if "acs" in index_format:
						container_client = current_app.config['blob_container_client']

						for file_to_upload in dir_list:
							blob_name = os.path.basename(file_to_upload)
							blob_client = container_client.get_blob_client(blob_name)
							with open(file_to_upload, "rb") as data:
								await blob_client.upload_blob(data, overwrite=True)
					for file_name in dir_list:
						file_src = os.path.join(os.getcwd(), 'uploads', f'{file_name}')
						os.remove(file_src)
				else:
					not_uploaded_files.append(file_name)

			except Exception as file_processing_error:
				traceback.print_exc()
				return {"message": f"Error processing file {file_name}: {str(file_processing_error)}", "status": "error"}

		if not dup_list:
			return {"message": "Files uploaded successfully.", "filename": ", ".join(uploaded_files), "status": "success"}
		elif not uploaded_files:
			return {
				"message": "No files uploaded. Files already exists, may have been uploaded by other user.",
				"filename": "",
				"status": "success"
			}
		else:
			return {
				"message": "Files have already been uploaded by other users. The remaining files have been processed successfully.",
				"filename": ", ".join(not_uploaded_files),
				"status": "success"
			}

	except Exception as e:
		traceback.print_exc()
		return {"message": "Error occurred while updating category: " + str(e), "status": "error"}


@bp_doc.route('/deleteFile', methods=['POST'])
async def delete_chunks_by_title():
	try:
		raw_data=await request.get_data()
		# #------STEP 1: DELETE INDEXES FROM ACS------
		json_str = raw_data.decode('utf8').replace("'", '"')
		data = json.loads(json_str)
		ids=data["chunk_ids"]
		docid=[]
		# Now, you can work with the JSON data as a Python dictionary
		for i in ids:
			docid.append({'id':i})
		search_client.delete_documents(documents=docid)
		print("step1 finished")

		#------STEP 2: DELETE FILE FROM BLOB STORAGE------


		container_name = os.getenv("BLOB_STORAGE_CONTAINER_NAME")
		#container_client = current_app.config['blob_container_client']
		blob_name = data.get("blob_name")
		print(blob_name)
		BLOB_STORAGE_CONNECTION_STRING = os.getenv("BLOB_STORAGE_CONNECTION_STRING")
		blob_service_client = BlobServiceClient.from_connection_string(BLOB_STORAGE_CONNECTION_STRING)
		blob_client =blob_service_client.get_blob_client(container =container_name,blob =blob_name)
		blob_client.delete_blob()
		print(f"Deleted blob: {blob_name}")



		#------STEP 3: DELETE FILE ENTRY FROM COSMOS_DB TABLE (gi_uploads)------
		# user_name ="super-admin"
		container = current_app.config['cosmos_db'].get_container_client("gi_uploads")
		query = "SELECT * FROM gi_uploads r WHERE r.file_name = @blob_name"
		query_params = [{"name": "@blob_name", "value": str(blob_name)}]
		# Execute the parameterized query
		items = list(container.query_items(query=query, parameters=query_params, enable_cross_partition_query=True))
        # items = list(container.query_items(query=query, enable_cross_partition_query=False))
		for item in items:
			container.delete_item(item,partition_key = {})
		return jsonify(f"File(s) entry with file_name '{blob_name}' deleted successfully.")
	except Exception as e:
		print(f"An error occurred while deleting cosmos file entry: {str(e)}")
		return jsonify({"message": f"Deleted {len(docid)}  chunks from the index."})
	except Exception as e:
		return jsonify({"message": "Error occurred while deleting chunks: " + str(e)})

#This API updates the feedback of a previously asked question by ID.
@bp_doc.route("/questionFeedback" , methods=["PUT"])
async def update_feedback():
	try:
		request_json = await request.get_json()
		id = request_json.get('id')
		feedback = request_json.get('feedback')

		user_name ="super-admin"
		container = current_app.config['cosmos_db'].get_container_client("gi_qa")
		query = "SELECT * FROM gi_qa r WHERE r.id = @id"
		query_params = [{"name": "@id", "value": str(id)}]

		# Execute the parameterized query
		query_result = container.query_items(
			query=query,
			parameters=query_params,
			enable_cross_partition_query=True
		)

		# Print the query results
		qa_record = None
		for item in query_result:
			qa_record = item
		if not qa_record:
			return {"message": "Question with ID {} not found".format(id)}

		qa_record['feedback'] = feedback
		qa_record['updatedBy'] = user_name
		qa_record['updatedAt'] = str(datetime.now())
		response = container.replace_item(item=qa_record, body=qa_record)

		if feedback==1:
			str_id = str(qa_record['id'])
			qna_indexing(str_id,qa_record['question'],qa_record['answer'],qa_record['thoughts'],str(qa_record['data_points']), str(qa_record['exclude_category']))

		return {"message": "Feedback updated successfully for question with ID {}".format(id)}
	except Exception as e:

		return {"message": "Error occurred while updating feedback: " + str(e)}
	
@bp_doc.route("/updateCategory", methods = ["PUT"])
async def updateCategory():
	try:
		category_id = request.args.get('category_id')
		status_flag = request.args.get('status_flag')
		category_name = request.args.get('category_name')
		category_code = request.args.get('category_code')
		container = current_app.config['cosmos_db'].get_container_client("gi_category")
		query = "SELECT * FROM gi_category cat WHERE cat.id = @category_id"
		query_params = [{"name": "@category_id", "value": str(category_id)}]

		# Execute the parameterized query
		categories = container.query_items(
			query=query,
			parameters=query_params,
			enable_cross_partition_query=True
		)
		category = next(categories, None)

		if category is None:
			return {"message": "Category not found"}
		
		status_mapping = {"false": 0, "true": 1}
		category['status'] = status_mapping.get(status_flag, 1)
		if category_name:
			category["category_name"] = category_name
		if category_code:
			category["category_code"] = category_code

		response = container.replace_item(item=category, body=category)

		response_message = ""
		if status_flag:
			response_message += "Category Activated." if category['status'] == 1 else "Category Deactivated."
		response_message += "Category is edited." if category_name or category_code  else ""
		return {"message": response_message}
	
	except Exception as e:
		return {"message": "Error occurred while updating category: " + str(e)}
