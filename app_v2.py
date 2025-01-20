import os
import tempfile
import logging
from io import BytesIO
from flask import Flask, request, jsonify
import requests
from PIL import Image
import cv2
import numpy as np
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import re
from urllib.parse import urlparse
from main_v1 import get_category_ids
from dotenv import load_dotenv
from parser import clean_and_format_json
import json

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

def classify_urls(urls):
    """
    Classifies URLs into video and image URLs based on their file extensions or patterns
    
    Args:
        urls (list): List of URLs to classify
        
    Returns:
        tuple: (image_urls, video_urls)
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.3gp', '.flv', '.webm', '.wmv'}
    
    image_urls = []
    video_urls = []
    
    for url in urls:
        # Clean the URL
        url = url.strip()
        if not url:
            continue
            
        # Parse the URL and get the path
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        # Check for file extensions
        _, ext = os.path.splitext(path)
        
        # Classify based on extension
        if ext in image_extensions:
            image_urls.append(url)
        elif ext in video_extensions:
            video_urls.append(url)
        else:
            # If no extension, try to identify by common patterns
            if any(pattern in url.lower() for pattern in ['img', 'image', 'photo', 'pic']):
                image_urls.append(url)
            elif any(pattern in url.lower() for pattern in ['video', 'vid', 'watch']):
                video_urls.append(url)
            else:
                # Default to image if can't determine
                image_urls.append(url)
    
    return image_urls, video_urls

# Initialize Vertex AI
vertexai.init(
    project=os.getenv('VERTEX_PROJECT'), 
    location=os.getenv('VERTEX_LOCATION')
)

# system_instruction = """
# You are an expert content moderator at a social media company. Your job is to examine post images, captions, and videos to identify potential violations of platform policies and categorize the content accordingly. 

# Read the provided text, examine the image or video, and analyze its content. Categorize each piece of content based on the given categories. 

# Categories:
# - Identify and flag any violations or sensitive content.
# - Assign severity levels to each identified category.
# - Conduct sentiment analysis for the captions or text associated with the post.

# Below is the updated list of moderation categories to classify the content:

# moderation_categories = [
#     "Harmful or Dangerous Content",
#     "Hate Speech",
#     "Harassment and Bullying",
#     "Violence and Threats",
#     "Misinformation and Fake News",
#     "Sexual Content and Nudity",
#     "Child Exploitation and Abuse",
#     "Spam and Scams",
#     "Intellectual Property Violations",
#     "Impersonation",
#     "Drugs, Alcohol, and Tobacco",
#     "Regulated Goods",
#     "Privacy Violations",
#     "Cultural Sensitivity",
#     "Platform Abuse",
#     "Environmental Harm",
#     "Sensitive or Disturbing Content",
#     "Unauthorized Commercial Activities",
#     "Non-Consensual Content",
#     "Other Violations"
# ]

# return the categories in list of dictionires 
# also add percenatge score which matching the category
# also return sentiment analysis with score
# also return inappropriate_flag true or false
# if content is not matching with moderate categories then return empty string
# """

system_instruction = """
You are an expert content moderator at a travel social media company. Your job is to examine post images, captions, and videos to identify potential violations of platform policies and categorize the content accordingly. 

Read the provided caption for the post, examine the image or video, and analyze its content. Categorize each piece of content based on the given categories. 

Categories:

Identify and flag any violations or sensitive content that is not appropriate for a travel social platform.
Assign severity levels to each identified category.
Conduct sentiment analysis for the overall post.
Below is a list of moderation categories to classify the content:

moderation_categories = [
    "Harmful or Dangerous Content",
    "Hate Speech",
    "Harassment and Bullying",
    "Violence and Threats",
    "Misinformation and Fake News",
    "Sexual Content and Nudity",
    "Child Exploitation and Abuse",
    "Spam and Scams",
    "Intellectual Property Violations",
    "Impersonation",
    "Drugs, Alcohol, and Tobacco",
    "Regulated Goods",
    "Privacy Violations",
    "Cultural Sensitivity",
    "Platform Abuse",
    "Environmental Harm",
    "Sensitive or Disturbing Content",
    "Unauthorized Commercial Activities",
    "Non-Consensual Content",
    "Other Violations"
]
You can add other categories according to the post 
return the categories in list of dictionaries 
also add a score which matching the category (0 to 1)
also return sentiment analysis with score in range -1 to 1 floating-point number, scores can be between -1 to 1 as well)
also return inappropriate_flag true or false
if content is not matching with any moderate categories then return empty string
example : ```
  {
    "category": "Violence and Threats",
    "score": 0.89,
    "sentiment": -0.4746
  },s
  {
    "category": "Sensitive or Disturbing Content",
    "score": 0.79,
    "sentiment": -0.5859375
  }
]
inappropriate_flag: True
```
return exactly like this in format
"""

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.2, 
    "top_p": 0.95,
}

model = GenerativeModel(model_name="gemini-1.5-flash-001", system_instruction=[system_instruction])


def fetch_and_preprocess_image(image_path):
    """
    Fetch and preprocess an image from a given path or URL
    
    Args:
        image_path (str): Path or URL of the image
    
    Returns:
        Part: Processed image part for Vertex AI
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/85.0.4183.121 Safari/537.36"
        )
    }
    
    try:
        if image_path.startswith(("http://", "https://")):
            try:
                response = requests.get(image_path, stream=True, headers=headers, verify=True)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            except requests.RequestException as e:
                logging.error(f"Error fetching image {image_path}: {e}")
                raise
        else:
            image = Image.open(image_path)

        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Resize and process the image
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        return Part.from_data(mime_type="image/jpeg", data=image_bytes)

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise


def fetch_and_preprocess_video(video_path):
    """
    Fetch and preprocess a video from a given path or URL
    
    Args:
        video_path (str): Path or URL of the video
    
    Returns:
        list: List of video frame parts for Vertex AI
    """
    temp_file_path = None
    supported_formats = ['.mp4', '.mov', '.avi', '.mkv', '.3gp', '.flv']

    try:
        if video_path.startswith(("http://", "https://")):
            response = requests.get(video_path, stream=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mkv") as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
        else:
            temp_file_path = video_path

        if not any(temp_file_path.endswith(ext) for ext in supported_formats):
            logging.error(f"Unsupported video format: {video_path}")
            raise ValueError(f"Unsupported video format: {video_path}")

        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Failed to open video file: {video_path}")

        frames = []
        success, frame = cap.read()
        frame_interval = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 5)

        while success and len(frames) < 5:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG")
            frames.append(Part.from_data(mime_type="image/jpeg", data=buffer.getvalue()))
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_interval)
            success, frame = cap.read()

        cap.release()
        return frames

    except Exception as e:
        logging.error(f"Error processing video: {e}")
        raise
    finally:
        if temp_file_path and temp_file_path != video_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

import re
import json

# def predict_categories(urls, caption=""):
#     """
#     Predicts categories based on URLs and optional caption
    
#     Args:
#         urls (list): List of URLs to process
#         caption (str, optional): The caption text. Defaults to empty string.
        
#     Returns:
#         list: Predicted categories with counts and metadata
#     """
#     # Classify URLs
#     image_urls, video_urls = classify_urls(urls)

#     # Initialize contents with caption if provided
#     contents = [Part.from_text(caption)] if caption else [Part.from_text("Null")]
#     categories_count = {}
#     inappropriate_flag = False
#     sentiment_analysis = 0  # Default sentiment score (adjust based on your implementation)

#     # Process images
#     for image_url in image_urls:
#         print("processing image", image_url)
#         try:
#             image_part = fetch_and_preprocess_image(image_url)
#             content = contents + [image_part]
#             response = model.generate_content(content, generation_config=generation_config)
#             raw_response = response.text.strip()
#             print(f"this is raw respon{raw_response}")

#             # Use regular expressions to extract individual JSON blocks
#             json_blocks = re.findall(r'```json\n(.*?)\n```', raw_response, re.DOTALL)

#             for block in json_blocks:
#                 try:
#                     parsed_data = json.loads(block)
#                     if isinstance(parsed_data, list):
#                         for item in parsed_data:
#                             category_name = item.get("moderation_category", "")
#                             score = item.get("score", 0)
#                             categories_count[category_name] = categories_count.get(category_name, 0) + 1
#                             if "Sexual Content and Nudity" in category_name:
#                                 inappropriate_flag = True
#                     elif isinstance(parsed_data, dict):
#                         if "sentiment" in parsed_data:
#                             sentiment_analysis = parsed_data.get("sentiment", 0)
#                         if "inappropriate_flag" in parsed_data:
#                             inappropriate_flag = parsed_data.get("inappropriate_flag", False)

#                 except json.JSONDecodeError:
#                     logging.error(f"Failed to decode JSON block: {block}")

#         except Exception as e:
#             logging.error(f"Error processing image URL {image_url}: {str(e)}")

#     # Process videos
#     for video_url in video_urls:
#         try:
#             video_frames = fetch_and_preprocess_video(video_url)
#             for frame in video_frames:
#                 content = contents + [frame]
#                 response = model.generate_content(content, generation_config=generation_config)
#                 raw_response = response.text.strip()

#                 # Use regular expressions to extract individual JSON blocks
#                 json_blocks = re.findall(r'```json\n(.*?)\n```', raw_response, re.DOTALL)

#                 for block in json_blocks:
#                     try:
#                         parsed_data = json.loads(block)
#                         if isinstance(parsed_data, list):
#                             for item in parsed_data:
#                                 category_name = item.get("category", "")
#                                 score = item.get("score", 0)
#                                 categories_count[category_name] = categories_count.get(category_name, 0) + 1
#                                 if "Sexual Content and Nudity" in category_name:
#                                     inappropriate_flag = True
#                         elif isinstance(parsed_data, dict):
#                             if "sentiment" in parsed_data:
#                                 sentiment_analysis = parsed_data.get("sentiment", 0)
#                             if "inappropriate_flag" in parsed_data:
#                                 inappropriate_flag = parsed_data.get("inappropriate_flag", False)

#                     except json.JSONDecodeError:
#                         logging.error(f"Failed to decode JSON block: {block}")

#         except Exception as e:
#             logging.error(f"Error processing video URL {video_url}: {str(e)}")

#     # Create structured data instead of raw JSON-formatted strings
#     result = [
#         {
#             "categories": [{"category": category, "score": count} for category, count in categories_count.items()],
#             "sentiment_analysis": sentiment_analysis,
#             "inappropriate_flag": inappropriate_flag
#         }
#     ]

#     return result

import json
import logging



# def predict_categories(urls, caption=""):
#     """
#     Predicts categories based on URLs and optional caption.
    
#     Args:
#         urls (list): List of URLs to process.
#         caption (str, optional): The caption text. Defaults to an empty string.
        
#     Returns:
#         dict: Predicted categories and inappropriate flag.
#     """
#     safty_settings=False
#     # Classify URLs
#     image_urls, video_urls = classify_urls(urls)
    
#     # Initialize contents with caption if provided
#     contents = [Part.from_text(caption)] if caption else [Part.from_text("Null")]
#     predicted_categories = []
#     inappropriate_flag = False
    
#     # Helper function to clean and parse the response
#     def clean_and_parse_response(raw_response):
#         try:
#             # Extract content between ```json blocks if present
#             if "```json" in raw_response:
#                 start_index = raw_response.find("```json") + len("```json")
#                 end_index = raw_response.rfind("```")
#                 json_content = raw_response[start_index:end_index].strip()
#             else:
#                 json_content = raw_response.strip()
            
#             # Split the content to separate JSON and inappropriate_flag
#             parts = json_content.split('inappropriate_flag:')
            
#             # Parse the categories JSON
#             categories_json = parts[0].strip()
#             categories = json.loads(categories_json)
            
#             # Get inappropriate flag value
#             flag = False
#             if len(parts) > 1:
#                 flag_value = parts[1].strip().lower()
#                 flag = flag_value == 'true'
            
#             return categories, flag
#         except json.JSONDecodeError as e:
#             logging.error(f"Failed to decode JSON response: {raw_response}")
#             return None, False
    
#     # Process images
#     for image_url in image_urls:
#         print("Processing image:", image_url)
#         try:
#             image_part = fetch_and_preprocess_image(image_url)
#             content = contents + [image_part]
#             response = model.generate_content(content, generation_config=generation_config)
#             # new_response=response.json()
#             print(f"safty respone model{response.candidates}")
#             # if response.candidates[0].finish_reason=="SAFETY":
#             if response.candidates:
#                 safty_settings=True
#             print(f"this is response fro m model {response}")
#             raw_response = response.text.strip()
            
#             print(f"this is raw response{raw_response}")
            
#             # Clean and parse the response
#             categories, flag = clean_and_parse_response(raw_response)
#             if categories:
#                 inappropriate_flag |= flag
#                 predicted_categories.extend(categories)
#         except Exception as e:
#             logging.error(f"Error processing image URL {image_url}: {str(e)}")
    
#     # Process videos
#     for video_url in video_urls:
#         print("Processing video:", video_url)
#         try:
#             video_frames = fetch_and_preprocess_video(video_url)
#             for frame in video_frames:
#                 content = contents + [frame]
#                 response = model.generate_content(content, generation_config=generation_config)
#                 if response.candidates[0].finish_reason=="SAFETY":
#                     safty_settings=True

#                 print(f"this is resposen from model{response}")
#                 raw_response = response.text.strip()
                
#                 # Clean and parse the response
#                 categories, flag = clean_and_parse_response(raw_response)
#                 if categories:
#                     inappropriate_flag |= flag
#                     predicted_categories.extend(categories)
#         except Exception as e:
#             logging.error(f"Error processing video URL {video_url}: {str(e)}")
#     if safty_settings:
#         return {
#             "reason":"due to safty settings"
#         }
#     else:
#         return {
#             "predicted_categories": predicted_categories,
#             "inappropriate_flag": inappropriate_flag
#         }





# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     """
#     Endpoint to predict categories from provided URLs and an optional caption.
#     """
#     try:
#         # Parse the request JSON data
#         data = request.get_json()
#         if not data:
#             return jsonify({"error": "Invalid JSON payload."}), 400

#         # Extract URLs and caption from the payload
#         urls = data.get('urls', [])
#         caption = data.get('caption', '')  # Optional caption

#         # Validate the URLs input
#         if not urls or not isinstance(urls, list):
#             return jsonify({"error": "Please provide a list of URLs."}), 400

#         # Call the prediction function
#         predicted_categories = predict_categories(urls, caption)
#         print(f"Predicted categories: {predicted_categories}")

#         # if predicted_categories:
#         #     return jsonify({
#         #         "reason":""
#         #     })
        
#         # else:



#         # Return the JSON response with proper formatting
#         return jsonify({
#             "predicted_categories": predicted_categories,
#             "predicted_category_ids": None  # Adjust this if necessary
#         }), 200


#     except Exception as e:
#         # Log the error and return a 500 response
#         logging.error(f"Prediction error: {str(e)}", exc_info=True)
#         return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500


# def predict_categories(urls, caption=""):
#     """
#     Predicts categories based on URLs and optional caption.
#     Returns safety ratings if content is flagged.
    
#     Args:
#         urls (list): List of URLs to process.
#         caption (str, optional): The caption text. Defaults to empty string.
        
#     Returns:
#         dict: Either safety ratings or predicted categories with inappropriate flag.
#     """
#     # Classify URLs
#     image_urls, video_urls = classify_urls(urls)
    
#     # Initialize contents with caption if provided
#     contents = [Part.from_text(caption)] if caption else [Part.from_text("Null")]
#     predicted_categories = []
#     inappropriate_flag = False
    
#     def extract_safety_ratings(candidates):
#         """
#         Extract safety ratings from response candidates.
#         """
#         safety_data = []
#         for rating in candidates[0].safety_ratings:
#             safety_data.append({
#                 "category": rating.category,
#                 "probability": rating.probability,
#                 "probability_score": rating.probability_score,
#                 "severity": rating.severity,
#                 "severity_score": rating.severity_score,
#                 "blocked": getattr(rating, 'blocked', False)
#             })
#         return safety_data

#     def clean_and_parse_response(raw_response):
#         try:
#             if "```json" in raw_response:
#                 start_index = raw_response.find("```json") + len("```json")
#                 end_index = raw_response.rfind("```")
#                 json_content = raw_response[start_index:end_index].strip()
#             else:
#                 json_content = raw_response.strip()
            
#             parts = json_content.split('inappropriate_flag:')
#             categories_json = parts[0].strip()
#             categories = json.loads(categories_json)
            
#             flag = False
#             if len(parts) > 1:
#                 flag_value = parts[1].strip().lower()
#                 flag = flag_value == 'true'
            
#             return categories, flag
#         except json.JSONDecodeError as e:
#             logging.error(f"Failed to decode JSON response: {raw_response}")
#             return None, False
    
#     # Process images
#     for image_url in image_urls:
#         print("Processing image:", image_url)
#         try:
#             image_part = fetch_and_preprocess_image(image_url)
#             content = contents + [image_part]
#             response = model.generate_content(content, generation_config=generation_config)
#             response_dict=response.to_dict()
#             print(f"response dict {response_dict}")
#             print(type(response_dict))

            
#             print(f"Safety response: {response.candidates[0]}")
#             print(type(response))
            
#             # Check for safety ratings
#             if response.candidates and response.candidates[0].finish_reason == "SAFETY":
#                 safety_ratings = extract_safety_ratings(response.candidates)
#                 return {
#                     "status": "safety_flagged",
#                     "safety_ratings": safety_ratings,
#                     "model_version": response.model_version
#                 }
            
#             # print(f"Model response: {response}")
#             raw_response = response.text.strip()
            
#             # print(f"Raw response: {raw_response}")
            
#             categories, flag = clean_and_parse_response(raw_response)
#             if categories:
#                 inappropriate_flag |= flag
#                 predicted_categories.extend(categories)
#         except Exception as e:
#             logging.error(f"Error processing image URL {image_url}: {str(e)}")
    
#     # Process videos
#     for video_url in video_urls:
#         print("Processing video:", video_url)
#         try:
#             video_frames = fetch_and_preprocess_video(video_url)
#             for frame in video_frames:
#                 content = contents + [frame]
#                 response = model.generate_content(content, generation_config=generation_config)
                
#                 # Check for safety ratings
#                 if response.candidates and response.candidates[0].finish_reason == "SAFETY":
#                     safety_ratings = extract_safety_ratings(response.candidates)
#                     return {
#                         "status": "safety_flagged",
#                         "safety_ratings": safety_ratings,
#                         "model_version": response.model_version
#                     }

#                 # print(f"Model response: {response}")
#                 raw_response = response.text.strip()
                
#                 categories, flag = clean_and_parse_response(raw_response)
#                 if categories:
#                     inappropriate_flag |= flag
#                     predicted_categories.extend(categories)
#         except Exception as e:
#             logging.error(f"Error processing video URL {video_url}: {str(e)}")
    
#     # Return normal response if no safety flags were raised
#     return {
#         "status": "success",
#         "predicted_categories": predicted_categories,
#         "inappropriate_flag": inappropriate_flag
#     }


# def predict_categories(urls, caption=""):
#     """
#     Predicts categories based on URLs and optional caption.
    
#     Args:
#         urls (list): List of URLs to process.
#         caption (str, optional): The caption text. Defaults to empty string.
        
#     Returns:
#         dict: Either safety ratings or predicted categories with inappropriate flag.
#     """
#     # Classify URLs
#     image_urls, video_urls = classify_urls(urls)
    
#     # Initialize contents with caption if provided
#     contents = [Part.from_text(caption)] if caption else [Part.from_text("Null")]
#     predicted_categories = []
#     inappropriate_flag = False
    
#     def clean_and_parse_response(raw_response):
#         try:
#             if "```json" in raw_response:
#                 start_index = raw_response.find("```json") + len("```json")
#                 end_index = raw_response.rfind("```")
#                 json_content = raw_response[start_index:end_index].strip()
#             else:
#                 json_content = raw_response.strip()
            
#             parts = json_content.split('inappropriate_flag:')
#             categories_json = parts[0].strip()
#             categories = json.loads(categories_json)
            
#             flag = False
#             if len(parts) > 1:
#                 flag_value = parts[1].strip().lower()
#                 flag = flag_value == 'true'
            
#             return categories, flag
#         except json.JSONDecodeError as e:
#             logging.error(f"Failed to decode JSON response: {raw_response}")
#             return None, False
    
#     # Process images
#     for image_url in image_urls:
#         print("Processing image:", image_url)
#         try:
#             image_part = fetch_and_preprocess_image(image_url)
#             content = contents + [image_part]
#             response = model.generate_content(content, generation_config=generation_config)
            
#             # Convert response to dictionary
#             response_dict = response.to_dict()
#             print(f"Response dict: {response_dict}")
            
#             # Check for safety flags
#             if (response_dict.get('candidates') and 
#                 response_dict['candidates'][0].get('finish_reason') == 'SAFETY'):
                
#                 return {
#                     "status": "safety_flagged",
#                     "safety_ratings": response_dict['candidates'][0]['safety_ratings'],
#                     "model_version": response_dict.get('model_version', 'unknown')
#                 }
            
#             raw_response = response.text.strip()
#             categories, flag = clean_and_parse_response(raw_response)
#             if categories:
#                 inappropriate_flag |= flag
#                 predicted_categories.extend(categories)
#         except Exception as e:
#             logging.error(f"Error processing image URL {image_url}: {str(e)}")
    
#     # Process videos
#     for video_url in video_urls:
#         print("Processing video:", video_url)
#         try:
#             video_frames = fetch_and_preprocess_video(video_url)
#             for frame in video_frames:
#                 content = contents + [frame]
#                 response = model.generate_content(content, generation_config=generation_config)
                
#                 # Convert response to dictionary
#                 response_dict = response.to_dict()
                
#                 # Check for safety flags
#                 if (response_dict.get('candidates') and 
#                     response_dict['candidates'][0].get('finish_reason') == 'SAFETY'):
                    
#                     return {
#                         "status": "safety_flagged",
#                         "safety_ratings": response_dict['candidates'][0]['safety_ratings'],
#                         "model_version": response_dict.get('model_version', 'unknown')
#                     }

#                 raw_response = response.text.strip()
#                 categories, flag = clean_and_parse_response(raw_response)
#                 if categories:
#                     inappropriate_flag |= flag
#                     predicted_categories.extend(categories)
#         except Exception as e:
#             logging.error(f"Error processing video URL {video_url}: {str(e)}")
    
#     # Return normal response if no safety flags were raised
#     return {
#         "status": "success",
#         "predicted_categories": predicted_categories,
#         "inappropriate_flag": inappropriate_flag
#     }

def predict_categories(urls, caption=""):
    """
    Predicts categories based on URLs and optional caption.
    
    Args:
        urls (list): List of URLs to process.
        caption (str, optional): The caption text. Defaults to empty string.
        
    Returns:
        dict: Predicted categories with scores, whether from safety ratings or normal prediction
    """
    # Classify URLs
    image_urls, video_urls = classify_urls(urls)
    
    # Initialize contents with caption if provided
    contents = [Part.from_text(caption)] if caption else [Part.from_text("Null")]
    predicted_categories = []
    inappropriate_flag = False

    def extract_categories_from_safety(safety_ratings):
        """
        Extract categories and scores from safety ratings into simplified format.
        """
        categories = []
        for rating in safety_ratings:
            # Only include categories with significant probability (not NEGLIGIBLE)
            if rating['probability'] != 'NEGLIGIBLE':
                categories.append({
                    'category': rating['category'].replace('HARM_CATEGORY_', ''),  # Remove prefix for cleaner output
                    'score': rating['probability_score'],
                    'sentiment': rating['severity_score']
                })
        return categories
    
    def clean_and_parse_response(raw_response):
        try:
            if "```json" in raw_response:
                start_index = raw_response.find("```json") + len("```json")
                end_index = raw_response.rfind("```")
                json_content = raw_response[start_index:end_index].strip()
            else:
                json_content = raw_response.strip()
            
            parts = json_content.split('inappropriate_flag:')
            categories_json = parts[0].strip()
            categories = json.loads(categories_json)
            
            flag = False
            if len(parts) > 1:
                flag_value = parts[1].strip().lower()
                flag = flag_value == 'true'
            
            return categories, flag
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON response: {raw_response}")
            return None, False
    
    # Process images
    for image_url in image_urls:
        print("Processing image:", image_url)
        try:
            image_part = fetch_and_preprocess_image(image_url)
            content = contents + [image_part]
            response = model.generate_content(content, generation_config=generation_config)
            
            # Convert response to dictionary
            response_dict = response.to_dict()
            print(f"Response dict: {response_dict}")
            
            # Check for safety flags
            if (response_dict.get('candidates') and 
                response_dict['candidates'][0].get('finish_reason') == 'SAFETY'):
                
                safety_categories = extract_categories_from_safety(
                    response_dict['candidates'][0]['safety_ratings']
                )
                
                return {
                    "status": "success",
                    "predicted_categories": safety_categories,
                    "category_source": "safety_ratings",
                    "model_version": response_dict.get('model_version', 'unknown')
                }
            
            raw_response = response.text.strip()
            categories, flag = clean_and_parse_response(raw_response)
            if categories:
                inappropriate_flag |= flag
                predicted_categories.extend(categories)
        except Exception as e:
            logging.error(f"Error processing image URL {image_url}: {str(e)}")
    
    # Process videos
    for video_url in video_urls:
        print("Processing video:", video_url)
        try:
            video_frames = fetch_and_preprocess_video(video_url)
            for frame in video_frames:
                content = contents + [frame]
                response = model.generate_content(content, generation_config=generation_config)
                
                response_dict = response.to_dict()
                
                if (response_dict.get('candidates') and 
                    response_dict['candidates'][0].get('finish_reason') == 'SAFETY'):
                    
                    safety_categories = extract_categories_from_safety(
                        response_dict['candidates'][0]['safety_ratings']
                    )
                    
                    return {
                        "status": "success",
                        "predicted_categories": safety_categories,
                        "category_source": "safety_ratings",
                        "model_version": response_dict.get('model_version', 'unknown')
                    }

                raw_response = response.text.strip()
                categories, flag = clean_and_parse_response(raw_response)
                if categories:
                    inappropriate_flag |= flag
                    predicted_categories.extend(categories)
        except Exception as e:
            logging.error(f"Error processing video URL {video_url}: {str(e)}")
    
    # Return normal response
    return {
        "status": "success",
        "predicted_categories": predicted_categories,
        "category_source": "normal_prediction",
        "inappropriate_flag": inappropriate_flag
    }



# @app.route('/predict', methods=['POST'])
# def predict():
#     """
#     Endpoint to predict categories from provided URLs and caption.
#     """
#     try:
#         data = request.get_json()
#         if not data:
#             return jsonify({"status": "error", "error": "Invalid JSON payload"}), 400

#         urls = data.get('urls', [])
#         caption = data.get('caption', '')

#         if not urls or not isinstance(urls, list):
#             return jsonify({"status": "error", "error": "Please provide a list of URLs"}), 400

#         result = predict_categories(urls, caption)
#         print(f"Prediction result: {result}")

#         # Return appropriate response based on status
#         if result["status"] == "safety_flagged":
#             return jsonify({
#                 "status": "safety_flagged",
#                 "safety_ratings": result["safety_ratings"],
#                 "model_version": result.get("model_version")
#             }), 200
#         else:
#             return jsonify({
#                 "status": "success",
#                 "predicted_categories": result["predicted_categories"],
#                 "predicted_category_ids": None,
#                 "inappropriate_flag": result["inappropriate_flag"]
#             }), 200

#     except Exception as e:
#         logging.error(f"Prediction error: {str(e)}", exc_info=True)
#         return jsonify({
#             "status": "error",
#             "error": "An internal server error occurred",
#             "details": str(e)
#         }), 500

app=Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict categories from provided URLs and caption.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "error": "Invalid JSON payload"}), 400

        urls = data.get('urls', [])
        caption = data.get('caption', '')

        if not urls or not isinstance(urls, list):
            return jsonify({"status": "error", "error": "Please provide a list of URLs"}), 400

        result = predict_categories(urls, caption)
        print(f"Prediction result: {result}")

        response_data = {
            "status": "success",
            "predicted_categories": result["predicted_categories"],
            # "predicted_category_ids": None,
        }

        # Add additional fields based on result type
        if "category_source" in result:
            response_data["category_source"] = result["category_source"]
        if "model_version" in result:
            response_data["model_version"] = result["model_version"]
        if "inappropriate_flag" in result:
            response_data["inappropriate_flag"] = result["inappropriate_flag"]
        else:
            response_data["inappropriate_flag"] = True  # Default value when not present

        return jsonify(response_data), 200

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": "An internal server error occurred",
            "details": str(e)
        }), 500



if __name__ == '__main__':
    app.run(debug=True, port=3400)