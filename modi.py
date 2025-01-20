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
from urllib.parse import urlparse
from dotenv import load_dotenv
import re

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Define standard moderation categories
MODERATION_CATEGORIES = {
    'violence': {
        'subcategories': ['graphic_violence', 'weapons', 'blood', 'injury'],
        'threshold': 0.7
    },
    'adult': {
        'subcategories': ['nudity', 'sexual_content', 'suggestive'],
        'threshold': 0.6
    },
    'hate': {
        'subcategories': ['hate_symbols', 'hate_speech', 'extremism'],
        'threshold': 0.7
    },
    'drugs': {
        'subcategories': ['drug_use', 'drug_paraphernalia', 'smoking'],
        'threshold': 0.6
    },
    'harassment': {
        'subcategories': ['bullying', 'threats', 'intimidation'],
        'threshold': 0.7
    },
    'self_harm': {
        'subcategories': ['suicide', 'self_injury', 'eating_disorders'],
        'threshold': 0.8
    },
    'spam': {
        'subcategories': ['spam', 'scams', 'misleading'],
        'threshold': 0.8
    }
}

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

# system_instruction = """You are a content moderation AI. Analyze the provided image/video and return:
# 1. Detailed identification of potentially inappropriate content
# 2. Confidence scores (0-100) for each identified category
# 3. Specific details about concerning elements
# 4. Overall determination if content is inappropriate
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

# - use these categories for results
# Format response as JSON:
# {
#     "categories": [
#         {
#             "name": "category_name",
#             "confidence": score,
#             "details": "specific_details"
#         }
#     ],
#     "is_inappropriate": boolean,
#     "overall_assessment": "explanation"
# }"""
system_instruction = """You are a content analysis AI. Analyze the provided image/video and return:
1. Content Moderation Analysis:
   - Identify potentially inappropriate or concerning content
   - Provide confidence scores (0-100) for each category
   - Flag specific elements of concern

2. Sentiment Analysis:
   - Overall emotional tone
   - Mood indicators
   - Key emotional elements
   - Engagement potential

Content Categories:
- violence_gore
- adult_content
- hate_speech
- harassment
- self_harm
- drugs_alcohol
- spam_misleading
- graphic_content
- weapons
- extremism

Sentiment Categories:
- positive
- negative
- neutral
- controversial
- emotional_intensity

Format response as JSON:
{
    "content_moderation": {
        "categories": {
            "category_name": {
                "score": 0-100,
                "details": "specific_details"
            }
        },
        "flags": ["list_of_specific_concerns"],
        "is_inappropriate": boolean
    },
    "sentiment_analysis": {
        "overall_tone": "positive/negative/neutral",
        "emotional_scores": {
            "positive": 0-100,
            "negative": 0-100,
            "neutral": 0-100,
            "controversial": 0-100
        },
        "intensity": 0-100,
        "key_emotions": ["emotion1", "emotion2"],
        "engagement_level": "high/medium/low"
    },
    "overall_assessment": "explanation"
}"""
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.1,
    "top_p": 0.95,
}

model = GenerativeModel(
    model_name="gemini-1.5-flash-001",
    system_instruction=[system_instruction]
)

def parse_raw_response(response_text):
    """
    Parse the raw text response from the model into a structured format
    
    Args:
        response_text (str): Raw text response from the model
        
    Returns:
        dict: Structured response data
    """
    try:
        # First try direct JSON parsing
        import json
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Fallback: Parse the text response
        categories = []
        is_inappropriate = False
        overall_assessment = ""
        
        # Common patterns for confidence scores
        confidence_patterns = [
            r'(\d+)%',
            r'(\d+)\s*percent',
            r'confidence:\s*(\d+)',
            r'score:\s*(\d+)',
        ]
        
        # Process response line by line
        lines = response_text.split('\n')
        current_category = None
        
        for line in lines:
            line = line.strip().lower()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check for inappropriate content indicators
            if any(term in line for term in ['inappropriate', 'violated', 'violation', 'flagged']):
                is_inappropriate = True
            
            # Look for category indicators
            for category, info in MODERATION_CATEGORIES.items():
                if category in line or any(sub in line for sub in info['subcategories']):
                    # Extract confidence score
                    confidence = 0
                    for pattern in confidence_patterns:
                        matches = re.findall(pattern, line)
                        if matches:
                            confidence = float(matches[0])
                            break
                    
                    # If no explicit confidence found, use presence as 90% confidence
                    if confidence == 0:
                        confidence = 90
                    
                    # Extract or generate details
                    details = line
                    if ':' in line:
                        details = line.split(':', 1)[1].strip()
                    
                    categories.append({
                        'name': category,
                        'confidence': confidence,
                        'details': details
                    })
                    current_category = category
                    break
            
            # If no category found, add details to current category if exists
                elif current_category and ':' not in line:
                    if categories:
                        categories[-1]['details'] += f" {line}"
        
        # If we found categories but no explicit inappropriate flag,
        # check if any category exceeds its threshold
        if categories and not is_inappropriate:
            for cat in categories:
                cat_info = MODERATION_CATEGORIES.get(cat['name'])
                if cat_info and cat['confidence'] >= cat_info['threshold'] * 100:
                    is_inappropriate = True
                    break
        
        return {
            'categories': categories,
            'is_inappropriate': is_inappropriate,
            'overall_assessment': overall_assessment or "Analysis based on detected content and patterns"
        }
        
    except Exception as e:
        logging.error(f"Error in parse_raw_response: {str(e)}")
        # Return a minimal valid response
        return {
            'categories': [],
            'is_inappropriate': False,
            'overall_assessment': f"Error parsing response: {str(e)}"
        }

def parse_moderation_response(response_text):
    """
    Parse and standardize the moderation response
    
    Args:
        response_text (str): Response text from the model
        
    Returns:
        dict: Standardized moderation results
    """
    try:
        # Get structured data from raw response
        result = parse_raw_response(response_text)
        
        # Standardize categories and scores
        standardized_categories = []
        is_inappropriate = result.get('is_inappropriate', False)
        
        for category in result.get('categories', []):
            category_name = category['name'].lower()
            confidence = float(category.get('confidence', 90))  # Default to 90 if not specified
            
            # Find matching standard category
            for std_category, info in MODERATION_CATEGORIES.items():
                if category_name in [std_category] + info['subcategories']:
                    if confidence >= info['threshold'] * 100:
                        is_inappropriate = True
                    
                    standardized_categories.append({
                        'category': std_category,
                        'subcategory': category_name,
                        'confidence': confidence,
                        'details': category.get('details', ''),
                        'exceeds_threshold': confidence >= info['threshold'] * 100
                    })
                    break
        
        return {
            'categories': standardized_categories,
            'is_inappropriate': is_inappropriate,
            'overall_assessment': result.get('overall_assessment', '')
        }
    except Exception as e:
        logging.error(f"Error in parse_moderation_response: {str(e)}")
        return {
            'categories': [],
            'is_inappropriate': False,
            'overall_assessment': f"Error standardizing response: {str(e)}"
        }

def moderate_content(urls, caption=""):
    """
    Moderate content based on URLs and optional caption
    
    Args:
        urls (list): List of URLs to process
        caption (str, optional): The caption text
        
    Returns:
        dict: Moderation results
    """
    image_urls, video_urls = classify_urls(urls)
    contents = [Part.from_text(caption)] if caption else [Part.from_text("Analyze this content for moderation")]
    
    all_results = []
    # for value in all_results:
    #     print(f"this safty setting is{value['Candidate']}")
    # Process images
    for image_url in image_urls:
        try:
            image_part = fetch_and_preprocess_image(image_url)
            content = contents + [image_part]
            response = model.generate_content(content, generation_config=generation_config)
            result = parse_moderation_response(response.text)
            if result:
                result['url'] = image_url
                all_results.append(result)
        except Exception as e:
            logging.error(f"Error moderating image {image_url}: {e}"),000
    
    # Process videos
    for video_url in video_urls:
        try:
            video_frames = fetch_and_preprocess_video(video_url)
            frame_results = []
            for frame in video_frames:
                content = contents + [frame]
                response = model.generate_content(content, generation_config=generation_config)
                result = parse_moderation_response(response.text)
                if result:
                    frame_results.append(result)
            
            # Aggregate video frame results
            if frame_results:
                aggregated_result = {
                    'url': video_url,
                    'is_inappropriate': any(r['is_inappropriate'] for r in frame_results),
                    'categories': [],
                    'overall_assessment': "Combined assessment from multiple video frames"
                }
                
                # Combine and average category scores
                all_categories = {}
                for result in frame_results:
                    for category in result['categories']:
                        cat_key = (category['category'], category['subcategory'])
                        if cat_key not in all_categories:
                            all_categories[cat_key] = {
                                'confidence_sum': 0,
                                'count': 0,
                                'details': set(),
                                'exceeds_threshold': False
                            }
                        all_categories[cat_key]['confidence_sum'] += category['confidence']
                        all_categories[cat_key]['count'] += 1
                        all_categories[cat_key]['details'].add(category['details'])
                        all_categories[cat_key]['exceeds_threshold'] |= category['exceeds_threshold']
                
                # Calculate averages and format categories
                for (category, subcategory), data in all_categories.items():
                    avg_confidence = data['confidence_sum'] / data['count']
                    aggregated_result['categories'].append({
                        'category': category,
                        'subcategory': subcategory,
                        'confidence': avg_confidence,
                        'details': ' | '.join(data['details']),
                        'exceeds_threshold': data['exceeds_threshold']
                    })
                
                all_results.append(aggregated_result)
        except Exception as e:
            logging.error(f"Error moderating video {video_url}: {e}")
    
    return all_results

app = Flask(__name__)

@app.route('/moderate', methods=['POST'])
def moderate():
    data = request.get_json()
    urls = data.get('urls', [])
    caption = data.get('caption', '')
    
    if not urls:
        return jsonify({"error": "Please provide at least one URL."}), 400
    
    try:
        moderation_results = moderate_content(urls, caption)
        print(f"this is results{moderation_results}")
        return jsonify({
            "results": moderation_results,
            "summary": {
                "total_urls": len(urls),
                "inappropriate_content_found": any(r['is_inappropriate'] for r in moderation_results),
                "processed_urls": len(moderation_results)
            }
        }), 200
    except Exception as e:
        logging.error(f"Moderation error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=3400)