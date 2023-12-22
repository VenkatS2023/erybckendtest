import logging
import os
import re
import openai
import json
import time
import base64
import html
# from pypdf import PdfReader
from tenacity import retry, wait_random_exponential, stop_after_attempt
import azure.functions as func
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import fitz
from azure.cosmos import CosmosClient, exceptions
import azure.functions as func
from azure.storage.blob import BlobServiceClient
import requests
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from datetime import datetime
from azure.cosmos.errors import CosmosHttpResponseError
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)  # Set the desired logging level
# Define a logger
logger = logging.getLogger(__name__)

MAX_SECTION_LENGTH = 1000
SENTENCE_SEARCH_LIMIT = 100
SECTION_OVERLAP = 100

COSMOS_DATABASE_ID = os.environ["COSMOS_DATABASE_ID"]
COSMOS_UPLOAD_CONTAINER = os.environ["COSMOS_UPLOAD_CONTAINER"]
COSMOS_ENDPOINT = os.environ["COSMOS_ENDPOINT"]
COSMOS_KEY = os.environ["COSMOS_KEY"]

client = CosmosClient(url=COSMOS_ENDPOINT, credential=COSMOS_KEY)
database = client.get_database_client(COSMOS_DATABASE_ID)
container = database.get_container_client(COSMOS_UPLOAD_CONTAINER)
tran_container = database.get_container_client("transactions")

AZURE_OPENAI_API_REGION = os.environ["AZURE_OPENAI_API_REGION"]
EMBEDDING_MODEL_DEPLOYMENT_NAME = os.environ["EMBEDDING_MODEL_DEPLOYMENT_NAME"]
openai.api_type = os.environ["OPENAI_API_TYPE"]
openai.api_version = os.environ["AZURE_OPENAI_API_VERSION"]
openai.api_base = os.environ["AZURE_OPENAI_API_BASE"]
openai.api_key = os.environ["AZURE_OPENAI_API_KEY"]

chunk_size = int(os.environ["chunk_size"])
blob_storage_connection_string = os.environ["blob_storage_connection_string"]
blob_storage_container_name = os.environ["blob_storage_container_name"]

blob_service_client = BlobServiceClient.from_connection_string(
    blob_storage_connection_string)
container_client = blob_service_client.get_container_client(
    blob_storage_container_name)

service_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]
azure_search_admin_key = os.environ["AZURE_SEARCH_ADMIN_KEY"]
azure_search_credential = AzureKeyCredential(azure_search_admin_key)

search_client = SearchClient(
    endpoint=service_endpoint, index_name=index_name, credential=azure_search_credential)


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
# Function to generate embeddings for title and content fields, also used for query embeddings
def generate_embeddings(text):
    """
    Create embeddings of chunks of text
    """
    try:
        # logger.info("Entered embedding function ^^^^^^^^^^^^^^^^^^^^^ ****************")
        # logger.info(f"Starting embedding of:*** {text}  *** with type {type(text)}")
        response = openai.Embedding.create(
            input=text, engine=EMBEDDING_MODEL_DEPLOYMENT_NAME)
        # logger.info("Response returned from azure openAi embedd****************")
        embeddings = response['data'][0]['embedding']
        token_used = response['usage']['total_tokens']
        # logging.info(f"Embedding Response {str(response)}")
        logging.info(f"Embedding Response1 {str(response['usage'])}")
        logger.info(f"Embedding Response2 {str(response['usage'])}")
        # logger.info("Azure OpenAI API call successful")
        return embeddings,token_used
    except openai.error.OpenAIError as e:
        # Handle OpenAI API errors
        logger.warning(f"Error calling OpenAI API:{e}", exc_info=True)
        raise e
    except Exception as e:
        # Handle other exceptions
        logger.error(f"An unexpected error occurred:{e}", exc_info=True)
        raise e


def modify_string(input_string):
    # Remove characters that don't match the allowed set: letters, digits, underscore, dash, equal sign, for Index key(id)
    modified_string = re.sub(r'[^a-zA-Z0-9_=\-]', '_', input_string)

    return modified_string

def table_to_html(table):
    table_html = "<table>"
    rows = [sorted([cell for cell in table.cells if cell.row_index == i], key=lambda cell: cell.column_index) for i in range(table.row_count)]
    for row_cells in rows:
        table_html += "<tr>"
        for cell in row_cells:
            tag = "th" if (cell.kind == "columnHeader" or cell.kind == "rowHeader") else "td"
            cell_spans = ""
            if cell.column_span > 1: cell_spans += f" colSpan={cell.column_span}"
            if cell.row_span > 1: cell_spans += f" rowSpan={cell.row_span}"
            table_html += f"<{tag}{cell_spans}>{html.escape(cell.content)}</{tag}>"
        table_html +="</tr>"
    table_html += "</table>"
    return table_html

def fetch_file_from_azure_blob(blob_name):
    """
    Read blob pdf and convert it into text
    """
    try:

        blob_client = container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob()
        file_content = blob_data.readall()
        pdf_text = ""
        offset = 0
        page_map = []
        form_recognizer_client = DocumentAnalysisClient(endpoint=f"https://formrecogniser0001.cognitiveservices.azure.com/", credential=AzureKeyCredential("f3310f3aeeb1488da72019492ff47365"), headers={"x-ms-useragent": "azure-search-chat-demo/1.0.0"})
        # with open(blob_name, "rb") as f:
        poller = form_recognizer_client.begin_analyze_document("prebuilt-layout", document = file_content)
        form_recognizer_results = poller.result()
        
        # pdf_document = fitz.open("pdf", file_content)
        for page_num, page in enumerate(form_recognizer_results.pages):
            logging.info(f"page_num - {str(page_num)}")
            tables_on_page = [table for table in form_recognizer_results.tables if table.bounding_regions[0].page_number == page_num + 1]

            # mark all positions of the table spans in the page
            page_offset = page.spans[0].offset
            page_length = page.spans[0].length
            table_chars = [-1]*page_length
            for table_id, table in enumerate(tables_on_page):
                for span in table.spans:
                    # replace all table spans with "table_id" in table_chars array
                    for i in range(span.length):
                        idx = span.offset - page_offset + i
                        if idx >=0 and idx < page_length:
                            table_chars[idx] = table_id

            # build page text by replacing characters in table spans with table html
            page_text = ""
            added_tables = set()
            for idx, table_id in enumerate(table_chars):
                if table_id == -1:
                    page_text += form_recognizer_results.content[page_offset + idx]
                elif table_id not in added_tables:
                    page_text += table_to_html(tables_on_page[table_id])
                    added_tables.add(table_id)

            page_text += " "
            page_map.append((page_num+1, offset, page_text))
            offset += len(page_text)
        # pdf_document = fitz.open("pdf", file_content)
        # num_pages = pdf_document.page_count
        # for page_num in range(num_pages):

        #     page = pdf_document.load_page(page_num)
        #     page_text = page.get_text()

        #     # # Fix newlines in the middle of sentences
        #     # page_text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", page_text.strip())
        #     # # Remove multiple newlines
        #     # page_text = re.sub(r"\n\s*\n", "\n\n", page_text)

        #     page_map.append((page_num, offset, page_text))
        #     offset += len(page_text)

        return page_map

    except Exception as e:
        logger.info(f"An error occurred in fetch_file_from_azure_blob: {e}")

# def get_document_text(blob_name):
#     try:

#         blob_client = container_client.get_blob_client(blob_name)
#         blob_data = blob_client.download_blob()
#         file_content = blob_data.readall()
#         offset = 0
#         page_map = []
#         reader = PdfReader(blob_name)
#         pages = reader.pages
#         for page_num, p in enumerate(pages):
#             page_text = p.extract_text()
#             page_map.append((page_num, offset, page_text))
#             offset += len(page_text)

#         return page_map

#     except Exception as e:
#         logger.info(f"An error occurred in fetch_file_from_azure_blob: {e}")

def split_text(page_map):
    SENTENCE_ENDINGS = [".", "!", "?"]
    WORDS_BREAKS = [",", ";", "(", ")", "[", "]", "{", "}", "\t", "\n"]
    # WORDS_BREAKS = [",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n"]

    def find_page(offset):
        num_pages = len(page_map)
        for i in range(num_pages - 1):
            if offset >= page_map[i][1] and offset < page_map[i + 1][1]:
                return i
        return num_pages - 1

    all_text = "".join(p[2] for p in page_map)
    length = len(all_text)
    start = 0
    end = length
    while start + SECTION_OVERLAP < length:
        last_word = -1
        end = start + MAX_SECTION_LENGTH

        if end > length:
            end = length
        else:
            # Try to find the end of the sentence
            while end < length and (end - start - MAX_SECTION_LENGTH) < SENTENCE_SEARCH_LIMIT and all_text[end] not in SENTENCE_ENDINGS:
                if all_text[end] in WORDS_BREAKS:
                    last_word = end
                end += 1
            if end < length and all_text[end] not in SENTENCE_ENDINGS and last_word > 0:
                end = last_word # Fall back to at least keeping a whole word
        if end < length:
            end += 1

        # Try to find the start of the sentence or at least a whole word boundary
        last_word = -1
        while start > 0 and start > end - MAX_SECTION_LENGTH - 2 * SENTENCE_SEARCH_LIMIT and all_text[start] not in SENTENCE_ENDINGS:
            if all_text[start] in WORDS_BREAKS:
                last_word = start
            start -= 1
        if all_text[start] not in SENTENCE_ENDINGS and last_word > 0:
            start = last_word
        if start > 0:
            start += 1

        section_text = all_text[start:end]
        yield (section_text, find_page(start))

        last_table_start = section_text.rfind("<table")
        if (last_table_start > 2 * SENTENCE_SEARCH_LIMIT and last_table_start > section_text.rfind("</table")):
            # If the section ends with an unclosed table, we need to start the next section with the table.
            # If table starts inside SENTENCE_SEARCH_LIMIT, we ignore it, as that will cause an infinite loop for tables longer than MAX_SECTION_LENGTH
            # If last table starts inside SECTION_OVERLAP, keep overlapping
            # if args.verbose: print(f"Section ends with unclosed table, starting next section with the table at page {find_page(start)} offset {start} table start {last_table_start}")
            start = min(end - SECTION_OVERLAP, start + last_table_start)
        else:
            start = end - SECTION_OVERLAP

    if start + SECTION_OVERLAP < end:
        yield (all_text[start:end], find_page(start))

def blob_name_from_file_page(filename, page = 0):
    if os.path.splitext(filename)[1].lower() == ".pdf":
        return os.path.splitext(os.path.basename(filename))[0] + ".pdf" + f"-{page}" 
    else:
        return os.path.basename(filename)

def create_sections(blob_name, page_map, file_id, category_id):
    # file_id = filename_to_id(filename)
    input_data = []
    for i, (content, pagenum) in enumerate(split_text(page_map)):
        item = {
            'id': f"{file_id}_{i+1}",
            'title': blob_name,
            'category': category_id,
            "sourcepage": blob_name_from_file_page(blob_name, pagenum),
            # 'blob_name': blob_name,
            'content': content
        }
        input_data.append(item)

    return input_data

def chunking_file(category_id, blob_name, file_id):
    # chunk_size = int(chunk_size)
    # input_data = []

    page_map = fetch_file_from_azure_blob(blob_name)
    # logger.info(f"Type of text in chunking_file function: {type(text)}")
    # Divide text into chunks
    # chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # # Create items for each chunk
    # for i, chunk in enumerate(chunks):
    #     item = {
    #         'id': f"{file_id}_{i+1}",
    #         'title': blob_name,
    #         # 'blob_name': blob_name,
    #         'content': chunk
    #     }
    #     input_data.append(item)
    input_data = create_sections(blob_name, page_map, file_id, category_id)
    return input_data       # --> chunks_list


def main(blob: func.InputStream):
    try:
        # Get the blob name from the blob trigger event
        blob_name = blob.name
        blob_name = blob_name.split('/', 1)[1]       # blob name
        query = "SELECT * FROM gi_uploads r WHERE r.file_name = @blob_name"
        query_params = [{"name": "@blob_name", "value": blob_name}]
        
        # Execute the parameterized query
        query_result = container.query_items(
            query=query,
            parameters=query_params,
            enable_cross_partition_query=True
        )
        logging.info(f"File_item is :    {str(query_result)}   ")

        # Print the query results
        file_item = []
        for item in query_result:
            file_item = item
        
        logging.info(f"File_item is :    {str(file_item)}   ")
        logging.info(f"Blob name is :    {blob_name}   ")

        file_id = blob_name
        # Modifying file_id according to index document key requirement
        # file_id = modify_string(file_id)
        file_id= file_id.replace(" ","_").replace(".","_")

        chunks_list = chunking_file(file_item['category_id'], blob_name, file_id)
        logger.info(f"Number of chunks created for file {file_id} : {len(chunks_list)}")

        # Generate embeddings for title and content fields for each chunk
        chunk_ids=[]
        total_tokens = 0
        for item in chunks_list:
            # title = item['title']
            # item['title'] = files_dictionary[blob_name]
            chunk_ids.append(item['id'])
            content = item['content']
            # logger.info(f"Starting embedding of:*** {content}  *** with type {type(content)}")
            # title_embeddings = generate_embeddings(title)
            try:
                content_embeddings, token_used = generate_embeddings(content)
                total_tokens = total_tokens + token_used
                # logger.info(f"Embeddings: {content_embeddings}")
                item['contentVector'] = content_embeddings
            except Exception as e:
                logger.error(
                    f"Error in generate_embeddings function: {e}", exc_info=True)

            # item['titleVector'] = title_embeddings
            item['@search.action'] = 'upload'
            
            # logger.info(f"Embeddings created and inserted in cosmos DB for file {file_id}")
            
        end_time = time.time()
        ex_time = end_time - file_item['ex_time']
        credit_used = round(total_tokens/1000,1) 

        file_item['chunk_ids'] = str(chunk_ids) 
        file_item['token_used'] = total_tokens 
        file_item['credit_used'] = credit_used
        file_item['ex_time'] = ex_time
        file_item['status'] = 1

        response = container.replace_item(item=file_item, body=file_item)
        logging.info(f"Response {str(response)}")
        # container.create_item(body=item)
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    batch_size = 1000  # Maximum batch size for uploading documents
    # embedd_chunks = [{key: value for key, value in item.items(
    # ) if key != 'blob_name'} for item in chunks_list]    # To exclude blob_name in index entry
    embedd_chunks = chunks_list

    # Split embedd_chunks into batches of maximum batch_size
    num_embedd_chunks = len(embedd_chunks)

    all_batches_uploaded = True  # Flag to track successful embedd_chunks upload

    for i in range(0, num_embedd_chunks, batch_size):
        batch = embedd_chunks[i:i + batch_size]
        try:
            search_client.upload_documents(batch)
            logging.info(
                f"Uploaded batch {i//batch_size + 1}/{(num_embedd_chunks-1)//batch_size + 1}")
        except Exception as e:
            logging.info(f"Error uploading batch {i//batch_size + 1}: {e}")
            all_batches_uploaded = False  # Set flag to False on upload error
            break  # Exit the loop on error

    if all_batches_uploaded:
        logging.info(f"Uploaded {len(embedd_chunks)} documents")

    email = file_item['uploaded_by']
    logging.info(f"Email-------------- {str(email)}")
    balance = calculate_balance(email)- credit_used
    rbalance = round(balance,2)
    logging.info(f"Balance ----------- {str(rbalance)}")
    update_transactions_table(email = email, balance = rbalance, service_type = "Index", token_usage=total_tokens, credit_used = credit_used, credit_assigned = 0)
    logging.info("End of the function.")

def get_current_trans_id_from_database():
    query = "SELECT Top 1 * FROM transactions t ORDER BY t.surr_no DESC"
    
    # Execute the parameterized query
    query_result = tran_container.query_items(
        query=query,
        enable_cross_partition_query=True
    )
    logging.info(f"Tran item is :    {str(query_result)}   ")

    # Print the query results
    tran_item = []
    for item in query_result:
        tran_item = item

    try:
        return tran_item['surr_no']
    except StopIteration:
        return 0
    except CosmosHttpResponseError as cosmos_error:
        print(f"Error querying Cosmos DB for transaction ID: {cosmos_error}")
        return 0

    
def calculate_balance(email):
    query = "SELECT Top 1 * FROM transactions t WHERE t.email = @email ORDER BY t.transaction_ts DESC"
    query_params = [{"name": "@email", "value": email}]
    
    # Execute the parameterized query
    query_result = tran_container.query_items(
        query=query,
        parameters=query_params,
        enable_cross_partition_query=True
    )
    logging.info(f"Tran item is :    {str(query_result)}   ")

    # Print the query results
    tran_item = []
    for item in query_result:
        tran_item = item

    # # Processing Cosmos transactions
    # user_credits = [transaction.get('credit', 0) for transaction in transactions]
    # sum_debit = sum(transaction.get('debit', 0) for transaction in transactions)

    # # Calculating balance
    # balance = sum(user_credits) - sum_debit
    logging.info(f"Current credit balance is :    {str(tran_item['balance'])}   ")

    return tran_item['balance']

def update_transactions_table(email, balance, service_type, token_usage=None, credit_used = 0, credit_assigned = 0):

    try:
        current_trans_id = get_current_trans_id_from_database()
        logging.info(f"current_trans_id: {str(current_trans_id)}") 
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
            "transaction_type": 2,  # Assuming 1 represents a user creation transaction
            "transaction_ts": str(formatted_datetime)
        }
        # Include "token_usage" only if it's provided
        if credit_assigned is not 0:
            new_transaction["credit"] = credit_assigned
        if token_usage is not None:
            new_transaction["token_usage"] = token_usage
        if credit_used is not 0:
            new_transaction["debit"] = credit_used
        logging.info(f"New Transaction: {str(new_transaction)}")
        tran_container.create_item(body=new_transaction)
        logging.info(f"Record added successfully")
        
    except exceptions.CosmosHttpResponseError as cosmos_error:
        logging.info(f"Error updating transactions table: {cosmos_error}")
    except Exception as e:
        logging.info(f"Error updating transactions table: {str(e)}")