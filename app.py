import io
import logging
import mimetypes
import os
import time

import aiohttp
import openai
from azure.identity.aio import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.monitor.opentelemetry import configure_azure_monitor
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.aio import SearchClient
from azure.storage.blob.aio import BlobServiceClient
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
from quart import (
	Blueprint,
	Quart,
	abort,
	current_app,
	jsonify,
	request,
	send_file,
	send_from_directory,
)
from azure.search.documents.indexes.models import (
	SearchIndex,
	SearchField,
	SearchFieldDataType,
	SimpleField,
	SearchableField,
	SearchIndex,
	SemanticConfiguration,
	PrioritizedFields,
	SemanticField,
	SearchField,
	SemanticSettings,
	VectorSearch,
	VectorSearchAlgorithmConfiguration,
)


from approaches.chatreadretrieveread import ChatReadRetrieveReadApproach
from approaches.readdecomposeask import ReadDecomposeAsk
from approaches.readretrieveread import ReadRetrieveReadApproach
from approaches.retrievethenread import RetrieveThenReadApproach
from dotenv import load_dotenv
from quart_cors import cors
load_dotenv("acs.env")


CONFIG_OPENAI_TOKEN = "openai_token"
CONFIG_CREDENTIAL = "azure_credential"
CONFIG_ASK_APPROACHES = "ask_approaches"
CONFIG_CHAT_APPROACHES = "chat_approaches"
CONFIG_BLOB_CONTAINER_CLIENT = "blob_container_client"

bp = Blueprint("routes", __name__, static_folder='static')
bp = cors(bp, allow_origin="*")

@bp.route("/")
async def index():
	return await bp.send_static_file("index.html")

@bp.route("/favicon.ico")
async def favicon():
	return await bp.send_static_file("favicon.ico")

@bp.route("/assets/<path:path>")
async def assets(path):
	return await send_from_directory("static/assets", path)

# Serve content files from blob storage from within the app to keep the example self-contained.
# *** NOTE *** this assumes that the content files are public, or at least that all users of the app
# can access all the files. This is also slow and memory hungry.
@bp.route("/content/<path>")
async def content_file(path):
	blob_container_client = current_app.config[CONFIG_BLOB_CONTAINER_CLIENT]
	blob = await blob_container_client.get_blob_client(path).download_blob()
	if not blob.properties or not blob.properties.has_key("content_settings"):
		abort(404)
	mime_type = blob.properties["content_settings"]["content_type"]
	if mime_type == "application/octet-stream":
		mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
	blob_file = io.BytesIO()
	await blob.readinto(blob_file)
	blob_file.seek(0)
	return await send_file(blob_file, mimetype=mime_type, as_attachment=False, attachment_filename=path)

@bp.route("/ask", methods=["POST"])
async def ask():
	try:
		print("-----------------Ask API Call happened-------------")
		if not request.is_json:
			return jsonify({"error": "request must be json"}), 415
		request_json = await request.get_json()
		approach = request_json["approach"]

		impl = current_app.config[CONFIG_ASK_APPROACHES].get(approach)
		print("-----Ask Step 1---------------")
		if not impl:
			return jsonify({"error": "unknown approach"}), 400
		# Workaround for: https://github.com/openai/openai-python/issues/371
		async with aiohttp.ClientSession() as s:
			openai.aiosession.set(s)
			print("-----Ask Step 2---------------")
			r = await impl.run(request_json["question"], request_json.get("overrides") or {})
		return jsonify(r)
	except Exception as e:
		print(str(e))
		logging.exception("Exception in /ask")
		return jsonify({"error": str(e)}), 500

@bp.route("/chat", methods=["POST"])
async def chat():
	print("-----------------Chat API Call happened-------------")
	if not request.is_json:
		return jsonify({"error": "request must be json"}), 415
	request_json = await request.get_json()
	approach = request_json["approach"]
	try:
		impl = current_app.config[CONFIG_CHAT_APPROACHES].get(approach)
		print("-----Chat Step 1---------------")
		if not impl:
			return jsonify({"error": "unknown approach"}), 400
		# Workaround for: https://github.com/openai/openai-python/issues/371
		async with aiohttp.ClientSession() as s:
			openai.aiosession.set(s)
			print("-----Chat Step 1---------------")
			r = await impl.run(request_json["history"], request_json.get("overrides") or {})
		return jsonify(r)
	except Exception as e:
		logging.exception("Exception in /chat")
		return jsonify({"error": str(e)}), 500

#@bp.before_request
#async def ensure_openai_token():
#    openai_token = current_app.config[CONFIG_OPENAI_TOKEN]
#    if openai_token.expires_on < time.time() + 60:
#        openai_token = await current_app.config[CONFIG_CREDENTIAL].get_token("https://cognitiveservices.azure.com/.default")
#        #current_app.config[CONFIG_OPENAI_TOKEN] = openai_token
#        openai.api_key = openai_token.token

@bp.before_app_serving
async def setup_clients():

	# Replace these with your own values, either in environment variables or directly here
	AZURE_SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
	AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX_NAME")
	AZURE_QNA_INDEX = os.getenv("AZURE_QA_INDEX_NAME")
	AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")

	AZURE_STORAGE_ACCOUNT = os.getenv("BLOB_STORAGE_CONNECTION_STRING")
	AZURE_STORAGE_CONTAINER = os.getenv("BLOB_STORAGE_CONTAINER_NAME")

	AZURE_OPENAI_SERVICE_BASE = os.getenv("AZURE_OPENAI_API_BASE")
	AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
	AZURE_OPENAI_TYPE = os.getenv("OPENAI_API_TYPE")
	AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.getenv("GPT3_LLM_MODEL_DEPLOYMENT_NAME")
	AZURE_OPENAI_CHATGPT_MODEL = os.getenv("GPT3_LLM_MODEL_NAME")
	AZURE_OPENAI_EMB_DEPLOYMENT = os.getenv("EMBEDDING_MODEL_DEPLOYMENT_NAME_1")
	AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")

	KB_FIELDS_CONTENT = 'content'
	KB_FIELDS_SOURCEPAGE = 'sourcepage'


	print("--------------Starting Backend Setup-------------")
	#--------------------Creating INDEXES in ACS--------------------
	key = AZURE_SEARCH_ADMIN_KEY
	azure_search_credential = AzureKeyCredential(key)

	vector_search = VectorSearch(
		algorithm_configurations=[
			VectorSearchAlgorithmConfiguration(
				name="my-vector-config",
				kind="hnsw",
				hnsw_parameters={
					"m": 4,
					"efConstruction": 400,
					"efSearch": 1000,
					"metric": "cosine"
				}
			)
		]
	)


	#-------- STEP 1 SEARCH INDEX-----------
	index_client = SearchIndexClient(endpoint=AZURE_SEARCH_SERVICE_ENDPOINT, credential=azure_search_credential)
	fields = [
		SimpleField(name="id", type=SearchFieldDataType.String, key=True),
		SearchableField(name="title", type=SearchFieldDataType.String,searchable=True, retrievable=True),
		SearchableField(name="content", type=SearchFieldDataType.String,searchable=True, retrievable=True),
		SearchableField(name="category", type=SearchFieldDataType.String,filterable=True, searchable=True, retrievable=True),
		SearchableField(name="sourcepage", type=SearchFieldDataType.String,filterable=True, searchable=True, retrievable=True),
		SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),searchable=True, vector_search_dimensions=1536, vector_search_configuration="my-vector-config"),
	]

	semantic_config = SemanticConfiguration(
		name="my-semantic-config",
		prioritized_fields=PrioritizedFields(
			title_field=SemanticField(field_name="title"),
			prioritized_content_fields=[SemanticField(field_name="content")]
		)
	)

	semantic_settings = SemanticSettings(configurations=[semantic_config])

	# Create the search index with the semantic settings
	index = SearchIndex(name=AZURE_SEARCH_INDEX, fields=fields,vector_search=vector_search, semantic_settings=semantic_settings)
	index_client.create_or_update_index(index)
	print('---------- Search Index Created------------------')

	#-------- QNA INDEX-----------
	qa_index_client = SearchIndexClient(endpoint=AZURE_SEARCH_SERVICE_ENDPOINT, credential=azure_search_credential)

	qa_fields = [
		SimpleField(name="id", type=SearchFieldDataType.String, key=True),
		SearchableField(name="question", type=SearchFieldDataType.String,searchable=True, retrievable=True),
		SearchableField(name="answer", type=SearchFieldDataType.String,searchable=True, retrievable=True),
		SearchableField(name="category_id", type=SearchFieldDataType.String,filterable=True, searchable=True, retrievable=True),
		SearchableField(name="index_format", type=SearchFieldDataType.String,filterable=True, searchable=True, retrievable=True),
		SearchableField(name="tracing", type=SearchFieldDataType.String,filterable=True, searchable=True, retrievable=True),
		SearchableField(name="cost_saved", type=SearchFieldDataType.String,filterable=True, searchable=True, retrievable=True),
		SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),searchable=True, vector_search_dimensions=1536, vector_search_configuration="my-vector-config"),
	]

	qa_semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    prioritized_fields=PrioritizedFields(
        title_field=SemanticField(field_name="answer"),
        prioritized_content_fields=[SemanticField(field_name="question")]
    )
)

	# Create the semantic settings with the configuration
	qa_semantic_settings = SemanticSettings(configurations=[qa_semantic_config])

	qa_index = SearchIndex(name=AZURE_QNA_INDEX, fields=qa_fields,vector_search=vector_search, semantic_settings=qa_semantic_settings)
	qa_index_client.create_or_update_index(qa_index)
	print('---------- QNA Index Created------------------')

	#-------- STEP 2 CREATE ACS SEARCH CLIENT-----------
	# Set up clients for Cognitive Search and Storage
	search_client = SearchClient(endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,index_name=AZURE_SEARCH_INDEX, credential=azure_search_credential)
	qna_client = SearchClient(endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,index_name=AZURE_QNA_INDEX, credential=azure_search_credential)
	print('---------- ACS SEARCH CLIENT Created------------------')


	#-------- STEP 3 BLOB STORAGE SETTING-----------
	blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_ACCOUNT)
	blob_container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER)
	print('---------- Blob Storage Setup Done------------------')

	#-------- STEP 4 AZURE OPEN AI SETTING-----------
	openai.api_base = AZURE_OPENAI_SERVICE_BASE
	openai.api_version = AZURE_OPENAI_VERSION
	openai.api_type = AZURE_OPENAI_TYPE
	openai.api_key = AZURE_OPENAI_KEY
	print('---------- Azure OpenAI Setup Done------------------')


	#-------- STEP 5 SETTING UP CURRENT CONFIG-----------
	# Various approaches to integrate GPT and external knowledge, most applications will use a single one of these patterns
	# or some derivative, here we include several for exploration purposes
	current_app.config[CONFIG_ASK_APPROACHES] = {
		"rtr": RetrieveThenReadApproach(
			search_client,
			AZURE_OPENAI_CHATGPT_DEPLOYMENT,
			AZURE_OPENAI_CHATGPT_MODEL,
			AZURE_OPENAI_EMB_DEPLOYMENT,
			KB_FIELDS_SOURCEPAGE,
			KB_FIELDS_CONTENT
		),
		"rrr": ReadRetrieveReadApproach(
			search_client,
			AZURE_OPENAI_CHATGPT_DEPLOYMENT,
			AZURE_OPENAI_EMB_DEPLOYMENT,
			KB_FIELDS_SOURCEPAGE,
			KB_FIELDS_CONTENT
		),
		"rda": ReadDecomposeAsk(search_client,
			AZURE_OPENAI_CHATGPT_DEPLOYMENT,
			AZURE_OPENAI_EMB_DEPLOYMENT,
			KB_FIELDS_SOURCEPAGE,
			KB_FIELDS_CONTENT
		)
	}

	current_app.config[CONFIG_CHAT_APPROACHES] = {
		"rrr": ChatReadRetrieveReadApproach(
			search_client,
			AZURE_OPENAI_CHATGPT_DEPLOYMENT,
			AZURE_OPENAI_CHATGPT_MODEL,
			AZURE_OPENAI_EMB_DEPLOYMENT,
			KB_FIELDS_SOURCEPAGE,
			KB_FIELDS_CONTENT,
		)
	}
	print("-------------------STEPUP COMPLETED------------------")


	# Use the current user identity to authenticate with Azure OpenAI, Cognitive Search and Blob Storage (no secrets needed,
	# just use 'az login' locally, and managed identity when deployed on Azure). If you need to use keys, use separate AzureKeyCredential instances with the
	# keys for each service
	# If you encounter a blocking error during a DefaultAzureCredential resolution, you can exclude the problematic credential by using a parameter (ex. exclude_shared_token_cache_credential=True)
	azure_credential = DefaultAzureCredential(exclude_shared_token_cache_credential = True)
	current_app.config[CONFIG_CREDENTIAL] = azure_credential
	current_app.config[CONFIG_BLOB_CONTAINER_CLIENT] = blob_container_client



def create_app():
	if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
		configure_azure_monitor()
		AioHttpClientInstrumentor().instrument()
	app = Quart(__name__)
	app = cors(app, allow_origin="*")
	app.register_blueprint(bp)
	app.asgi_app = OpenTelemetryMiddleware(app.asgi_app)
	return app
