from quart import (
	Blueprint,
	current_app,
	jsonify,
	request,Response
)
import logging
import sys
import traceback
from datetime import datetime
import os
from botbuilder.core import BotFrameworkAdapterSettings,TurnContext, BotFrameworkAdapter
from .bot import MyBot
from botbuilder.schema import Activity, ActivityTypes
from dotenv import load_dotenv
load_dotenv("acs.env")

APP_ID=os.getenv("APP_ID")
APP_PASSWORD=os.getenv("APP_PASSWORD")
BASEURL=os.getenv("BOT_URL_DEV")
BOT_APPROACH=os.getenv("BOT_APPROACH")
BOT_RETRIEVAL_MODE=os.getenv("BOT_RETRIEVAL_MODE")
BOT_SEMANTIC_RANKER=os.getenv("BOT_SEMANTIC_RANKER")
BOT_CAPTIONS=os.getenv("BOT_CAPTIONS")
BOT_TOP=os.getenv("BOT_TOP")
BOT_TEMPERATURE=os.getenv("BOT_TEMPERATURE")
BOT_PROMPT_TEMPLATE=os.getenv("BOT_PROMPT_TEMPLATE")

SETTINGS = BotFrameworkAdapterSettings(APP_ID, APP_PASSWORD)
ADAPTER = BotFrameworkAdapter(SETTINGS)
bp_message = Blueprint("routes_message", __name__, static_folder='static')
print("--------------BOT MESSAGES------------")
@bp_message.route("/api/messages", methods=["POST"])
async def botmessages():
    try:
        async def on_error(context: TurnContext, error: Exception):
            # This check writes out errors to console log .vs. app insights.
            print(f"\n [on_turn_error] unhandled error: {error}", file=sys.stderr)
            traceback.print_exc()

             # Send a message to the user
            await context.send_activity("Sorry, the bot is not available at the moment, Please contact your administrator or try again later.")

    # Send a trace activity if we're talking to the Bot Framework Emulator
            if context.activity.channel_id == "emulator":
        # Create a trace activity that contains the error object
                trace_activity = Activity(
                    label="TurnError",
                    name="on_turn_error Trace",
                    timestamp=datetime.utcnow(),
                    type=ActivityTypes.trace,
                    value=f"{error}",
                    value_type="https://www.botframework.com/schemas/error",
                )
        # Send a trace activity, which will be displayed in Bot Framework Emulator
                await context.send_activity(trace_activity)


        ADAPTER.on_turn_error = on_error
        BOT = MyBot()
        async def messages():
            if "application/json" in request.headers["Content-Type"]:
                body = await request.json
            else:
                return Response(status=415)

            activity = Activity().deserialize(body)
            auth_header = request.headers["Authorization"] if "Authorization" in request.headers else ""

            response = await ADAPTER.process_activity(activity, auth_header, BOT.on_turn)
            if response:
                return jsonify(data=response.body, status=response.status)
            return jsonify(status=201)
        await messages()
        response = Response('This is an example response', status=200)
        return response
    except Exception as e:
        print(e,"APP_ID")
        logging.info(f"Error in bot end point {e}")
	   








