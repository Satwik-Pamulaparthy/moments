from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from azure.core.exceptions import HttpResponseError
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get credentials and endpoint from environment variables
endpoint = os.getenv("Endpoint")
key = os.getenv("Key")

# Set up credentials and the client
credentials = CognitiveServicesCredentials(key)

client = ComputerVisionClient(endpoint=endpoint, credentials=credentials)

# Function to generate captions (descriptions) for the image
def generate_caption(image_path):
    """
    Generates a caption (description) for an image using Azure Vision API.
    :param image_path: Path to the image file.
    :return: caption (description) if successful, None otherwise.
    """
    try:
        with open(image_path, "rb") as image_stream:
            # Request description (caption) feature from the API
            analysis = client.describe_image_in_stream(image_stream)

        caption = analysis.captions[0].text if analysis.captions else None
        return caption

    except HttpResponseError as e:
        print(f"Error occurred while analyzing the image: {e}")
        return None

# Function to generate tags for the image
def generate_tags(image_path):
    """
    Generates tags for an image using Azure Vision API.
    :param image_path: Path to the image file.
    :return: list of tags if successful, None otherwise.
    """
    try:
        with open(image_path, "rb") as image_stream:
            # Request tags feature from the API
            analysis = client.analyze_image_in_stream(image_stream, visual_features=["Tags"])

        tags = [tag.name for tag in analysis.tags]
        return tags

    except HttpResponseError as e:
        print(f"Error occurred while analyzing the image: {e}")
        return None
