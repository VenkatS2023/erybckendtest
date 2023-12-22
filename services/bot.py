
import os
from botbuilder.core import ActivityHandler, TurnContext, MessageFactory, CardFactory
from botbuilder.schema import ChannelAccount, Attachment, MediaUrl, CardAction, CardImage, HeroCard, ActionTypes
import logging
import requests
import aiohttp
import openai
from quart import (
    Blueprint,
    abort,
    current_app,
    jsonify,
    request,
    send_file,
    send_from_directory,
)

from dotenv import load_dotenv
load_dotenv("acs.env")


CONFIG_OPENAI_TOKEN = "openai_token"
CONFIG_CREDENTIAL = "azure_credential"
CONFIG_ASK_APPROACHES = "ask_approaches"
CONFIG_CHAT_APPROACHES = "chat_approaches"
CONFIG_BLOB_CONTAINER_CLIENT = "blob_container_client"

APP_ID = os.getenv("APP_ID")
APP_PASSWORD = os.getenv("APP_PASSWORD")
BASEURL = os.getenv("BOT_URL_DEV")
BOT_APPROACH = os.getenv("BOT_APPROACH")
BOT_RETRIEVAL_MODE = os.getenv("BOT_RETRIEVAL_MODE")
BOT_SEMANTIC_RANKER = os.getenv("BOT_SEMANTIC_RANKER")
BOT_CAPTIONS = os.getenv("BOT_CAPTIONS")
BOT_TOP = os.getenv("BOT_TOP")
BOT_TEMPERATURE = os.getenv("BOT_TEMPERATURE")
BOT_PROMPT_TEMPLATE = os.getenv("BOT_PROMPT_TEMPLATE")


# BOT LOGIC
class MyBot(ActivityHandler):
    def __init__(self):
        pass
        # self.adapter = BotFrameworkAdapter(APP_ID, APP_PASSWORD)
        # pass

    async def on_message_activity(self, turn_context: TurnContext):
        logging.info(f"hitting in bot file")

        body = {
            "history": [{"user": turn_context.activity.text}],
            "approach": BOT_APPROACH,
            "overrides": {
                "retrieval_mode": BOT_RETRIEVAL_MODE,
                "semantic_ranker": BOT_SEMANTIC_RANKER,
                "semantic_captions": BOT_CAPTIONS,
                "top": BOT_TOP,
                "temperature": BOT_TEMPERATURE,
                "prompt_template": BOT_PROMPT_TEMPLATE,
                # "prompt_template_prefix": options.overrides?.promptTemplatePrefix,
                # "prompt_template_suffix": options.overrides?.promptTemplateSuffix,
                # "exclude_category": options.overrides?.excludeCategory,
                # "suggest_followup_questions": options.overrides?.suggestFollowupQuestions
            }
        }
        bodya = {
            "retrieval_mode": BOT_RETRIEVAL_MODE,
            "semantic_ranker": BOT_SEMANTIC_RANKER,
            "semantic_captions": BOT_CAPTIONS,
            "top": BOT_TOP,
            "temperature": BOT_TEMPERATURE,
            "prompt_template": BOT_PROMPT_TEMPLATE,
            # "prompt_template_prefix": options.overrides?.promptTemplatePrefix,
            # "prompt_template_suffix": options.overrides?.promptTemplateSuffix,
            # "exclude_category": options.overrides?.excludeCategory,
            # "suggest_followup_questions": options.overrides?.suggestFollowupQuestions
        }
        print(body, "body")
        print("printing here*******************", BASEURL+"/chat")
        # response=requests.post(BASEURL+'/chat',json=body)
        try:
            impl = current_app.config[CONFIG_CHAT_APPROACHES].get(BOT_APPROACH)
            print("----- log 1---------------")
            if not impl:
                return jsonify({"error": "unknown approach"}), 400
            # Workaround for: https://github.com/openai/openai-python/issues/371
            async with aiohttp.ClientSession() as s:
                openai.aiosession.set(s)
                print("-----log 2---------------")
                r = await impl.run( [{"user": turn_context.activity.text}],  {})
                answer=r.get("answer")
                print(r,"answer")
                await turn_context.send_activity(f"{answer}")
            # return jsonify(r)
        except Exception as e:
            print("error in bot")
            logging.exception("Exception in /chat")
            # return jsonify({"error": str(e)}), 500
        # try:
        #     print(f"responsestatus_code---------{response.status_code}")
        # if response.status_code==200:
        #     print("20000000000000000htting call the api afterwards")
        #     jsondata=response.get_json()
        #     answer=jsondata.get("answer")

        #     await turn_context.send_activity(f"{answer}")

        # except Exception as e:
            # print("Error ~~~~~~~~~~",e)
            # await turn_context.send_activity(f"Error: {e}")
        # await turn_context.send_activity(f"{body}")

    # GREETING MSG
    def create_hero_card(self) -> Attachment:
        herocard = HeroCard(text="Hello!  I am Eryl!. How can I help you today?")
        return CardFactory.hero_card(herocard)
    # ON MEMEBERS ADD TRIGGER GREET FUNCTION

    async def on_members_added_activity(
        self,
        members_added: ChannelAccount,
        turn_context: TurnContext
    ):
        for member_added in members_added:
            if member_added.id != turn_context.activity.recipient.id:
                cardatt = self.create_hero_card()
                msg_activity = MessageFactory.attachment(cardatt)
                await turn_context.send_activity(msg_activity)


bot = MyBot()
