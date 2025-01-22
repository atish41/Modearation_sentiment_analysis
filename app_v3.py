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

# [Previous imports and setup code remains the same...]


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




system_instruction = """
You are an expert content moderator at a travel social media company. Your job is to examine post images, captions, and videos to identify potential violations of platform policies and categorize the content accordingly. 

Read the provided caption for the post, examine the image or video, and analyze its content. Categorize each piece of content based on the given categories. 

Categories:

Identify and flag any violations or sensitive content that is not appropriate for a travel social platform.
Assign severity levels to each identified category.
Conduct sentiment analysis for both individual categories and the overall post content.
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
also return overall_sentiment as a separate field with a score between -1 to 1
if content is not matching with any moderate categories then return empty string
example : ```
{
  "categories": [
    {
      "category": "Violence and Threats",
      "score": 0.89,
      "sentiment": -0.4746
    },
    {
      "category": "Sensitive or Disturbing Content",
      "score": 0.79,
      "sentiment": -0.5859375
    }
  ],
  "inappropriate_flag": true,
  "overall_sentiment": 0.2
}
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

def clean_and_parse_response(raw_response):
    try:
        if "```json" in raw_response:
            start_index = raw_response.find("```json") + len("```json")
            end_index = raw_response.rfind("```")
            json_content = raw_response[start_index:end_index].strip()
        else:
            json_content = raw_response.strip()
        
        # Parse the JSON content
        response_data = json.loads(json_content)
        
        # Extract categories and flags
        categories = response_data.get('categories', [])
        inappropriate_flag = response_data.get('inappropriate_flag', False)
        overall_sentiment = response_data.get('overall_sentiment', 0.0)
        
        return categories, inappropriate_flag, overall_sentiment
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON response: {raw_response}")
        return None, False, 0.0

def predict_categories(urls, caption=""):
    """
    Predicts categories based on URLs and optional caption.
    
    Args:
        urls (list): List of URLs to process.
        caption (str, optional): The caption text. Defaults to empty string.
        
    Returns:
        dict: Predicted categories with scores, whether from safety ratings or normal prediction
    """
    image_urls, video_urls = classify_urls(urls)
    contents = [Part.from_text(caption)] if caption else [Part.from_text("Null")]
    predicted_categories = []
    inappropriate_flag = False
    overall_sentiments = []

    def extract_categories_from_safety(safety_ratings):
        categories = []
        overall_sentiment = 0.0
        
        for rating in safety_ratings:
            if rating['probability'] != 'NEGLIGIBLE':
                sentiment = -(rating['severity_score'])
                categories.append({
                    'category': rating['category'].replace('HARM_CATEGORY_', ''),
                    'score': rating['probability_score'],
                    'sentiment': sentiment
                })
                overall_sentiment += sentiment
        
        # Calculate average sentiment if there are categories
        if categories:
            overall_sentiment /= len(categories)
            
        return categories, overall_sentiment

    # Process images
    for image_url in image_urls:
        try:
            image_part = fetch_and_preprocess_image(image_url)
            content = contents + [image_part]
            response = model.generate_content(content, generation_config=generation_config)
            
            response_dict = response.to_dict()
            
            if (response_dict.get('candidates') and 
                response_dict['candidates'][0].get('finish_reason') == 'SAFETY'):
                
                safety_categories, safety_sentiment = extract_categories_from_safety(
                    response_dict['candidates'][0]['safety_ratings']
                )
                
                return {
                    "status": "success",
                    "predicted_categories": safety_categories,
                    "category_source": "safety_ratings",
                    "model_version": response_dict.get('model_version', 'unknown'),
                    "overall_sentiment": safety_sentiment
                }
            
            raw_response = response.text.strip()
            categories, flag, sentiment = clean_and_parse_response(raw_response)
            if categories:
                inappropriate_flag |= flag
                predicted_categories.extend(categories)
                overall_sentiments.append(sentiment)
        except Exception as e:
            logging.error(f"Error processing image URL {image_url}: {str(e)}")

    # Process videos
    for video_url in video_urls:
        try:
            video_frames = fetch_and_preprocess_video(video_url)
            frame_sentiments = []
            
            for frame in video_frames:
                content = contents + [frame]
                response = model.generate_content(content, generation_config=generation_config)
                
                response_dict = response.to_dict()
                print(f"this is resposne safty {response_dict}")
                
                if (response_dict.get('candidates') and 
                    response_dict['candidates'][0].get('finish_reason') == 'SAFETY'):
                    
                    safety_categories, safety_sentiment = extract_categories_from_safety(
                        response_dict['candidates'][0]['safety_ratings']
                    )
                    
                    return {
                        "status": "success",
                        "predicted_categories": safety_categories,
                        "category_source": "safety_ratings",
                        "model_version": response_dict.get('model_version', 'unknown'),
                        "overall_sentiment": safety_sentiment
                    }

                raw_response = response.text.strip()
                categories, flag, sentiment = clean_and_parse_response(raw_response)
                if categories:
                    inappropriate_flag |= flag
                    predicted_categories.extend(categories)
                    frame_sentiments.append(sentiment)
            
            # Average sentiment across frames
            if frame_sentiments:
                overall_sentiments.append(sum(frame_sentiments) / len(frame_sentiments))
                
        except Exception as e:
            logging.error(f"Error processing video URL {video_url}: {str(e)}")

    # Calculate final overall sentiment
    final_overall_sentiment = 0.0
    if overall_sentiments:
        final_overall_sentiment = sum(overall_sentiments) / len(overall_sentiments)

    return {
        "status": "success",
        "predicted_categories": predicted_categories,
        "category_source": "normal_prediction",
        "inappropriate_flag": inappropriate_flag,
        "overall_sentiment": final_overall_sentiment
    }

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

        response_data = {
            "status": "success",
            "predicted_categories": result["predicted_categories"],
            "overall_sentiment": result["overall_sentiment"],
            "inappropriate_flag": result.get("inappropriate_flag", True)
        }

        if "category_source" in result:
            response_data["category_source"] = result["category_source"]
        if "model_version" in result:
            response_data["model_version"] = result["model_version"]

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