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

# Updated system instruction
system_instruction = """You are a social media content moderator. Analyze images, videos, or captions (if given), 
and moderate them by flagging inappropriate content such as drugs, smoking, threats, killing, etc. 
Return moderation categories with associated scores indicating the likelihood of inappropriate content."""

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

def predict_categories(urls, caption=""):
    """
    Predicts categories based on URLs and optional caption
    
    Args:
        urls (list): List of URLs to process
        caption (str, optional): The caption text. Defaults to empty string.
        
    Returns:
        list: Predicted moderation categories with scores
    """
    # Classify URLs
    image_urls, video_urls = classify_urls(urls)
    
    # Initialize contents with caption if provided
    contents = [Part.fr  om_text(caption)] if caption else [Part.from_text("Null")]
    moderation_results = []

    # Process images
    for image_url in image_urls:
        print("processing image", image_url)
        try:
            image_part = fetch_and_preprocess_image(image_url)
            content = contents + [image_part]
            response = model.generate_content(content, generation_config=generation_config)
            moderation_results.append({"url": image_url, "result": response.text.strip()})
        except Exception as e:
            logging.error(f"Error processing image URL {image_url}: {str(e)}")

    # Process videos
    for video_url in video_urls:
        try:
            video_frames = fetch_and_preprocess_video(video_url)
            for frame in video_frames:
                content = contents + [frame]
                response = model.generate_content(content, generation_config=generation_config)
                moderation_results.append({"url": video_url, "result": response.text.strip()})
        except Exception as e:
            logging.error(f"Error processing video URL {video_url}: {str(e)}")

    return moderation_results


app = Flask(__name__)

@app.route('/moderate', methods=['POST'])
def moderate_content():
    """
    Endpoint to moderate content based on URLs and an optional caption.
    
    Expected JSON body:
    {
        "urls": ["https://example.com/image.jpg", "https://example.com/video.mp4"],
        "caption": "Optional caption text"
    }
    
    Returns:
        JSON response containing moderation results.
    """
    try:
        data = request.get_json()
        if not data or 'urls' not in data:
            return jsonify({"error": "Invalid input. 'urls' is required."}), 400
        
        urls = data.get('urls', [])
        caption = data.get('caption', "")
        
        # Validate URLs input
        if not isinstance(urls, list) or not all(isinstance(url, str) for url in urls):
            return jsonify({"error": "'urls' must be a list of strings."}), 400
        
        # Call the moderation function
        results = predict_categories(urls, caption)
        return jsonify({"moderation_results": results})
    
    except Exception as e:
        logging.error(f"Error in /moderate endpoint: {e}")
        return jsonify({"error": "An error occurred while processing the request."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=3400)
