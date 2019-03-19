import os, uuid, sys
from azure.storage.blob import BlockBlobService, PublicAccess
from flask_app.entrypoint import app

def initialize_azure_blob():
    try:
        # Create the BlockBlockService that is used to call the Blob service for the storage account
        block_blob_service = BlockBlobService(
            account_name='accountname', 
            account_key='accountkey'
        )
    except Exception as e:
        app.logger.warning(e)