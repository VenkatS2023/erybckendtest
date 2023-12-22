import os
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
from quart import Quart
from dotenv import load_dotenv
from quart_cors import cors

from services.setup import bp
from services.documentService import bp_doc
from services.botmessage import bp_message
from services.userCreditService import bp_user_credit
load_dotenv("acs.env")

bp = cors(bp, allow_origin="*")

CONFIG_OPENAI_TOKEN = "openai_token"
CONFIG_CREDENTIAL = "azure_credential"
CONFIG_ASK_APPROACHES = "ask_approaches"
CONFIG_CHAT_APPROACHES = "chat_approaches"
CONFIG_BLOB_CONTAINER_CLIENT = "blob_container_client"




def create_app():
	if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
		configure_azure_monitor()
		AioHttpClientInstrumentor().instrument()
	app = Quart(__name__)
	app = cors(app, allow_origin="*")
	app.register_blueprint(bp)
	app.register_blueprint(bp_doc, url_prefix = "/documentService")
	app.register_blueprint(bp_user_credit, url_prefix='/userCreditService')
	app.register_blueprint(bp_message)
	
	app.asgi_app = OpenTelemetryMiddleware(app.asgi_app)
	return app
