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

system_instruction = """
You are an expert content categorizer at a social media company. Your job is to look at post images and their respective captions and derive categories related to them for better recommendations. 

Read the text and examine the image. Categorize each based on their content.
you can return multiple , separate categories for single post containing very different captions, image, and videos

Categories:
- Determine the main theme or topic of the text.
- Identify the main subject and context of the image.
- Identify the main subject and context of the video.
Below is a list of categories where each category is separated by a comma. Match the exact name of the category
CATEGORIES_LIST={Gardening, Cooking and Baking, DIY and Crafts, Photography, Reading and Book Clubs, Gaming, Collecting (e.g., stamps, coins), Knitting and Sewing, Painting and Drawing, Running, Yoga and Pilates, Cycling, Hiking and Outdoor Activities, Team Sports (e.g., soccer, basketball), Swimming, Fitness and Bodybuilding, Martial Arts, Dance, Movies and TV Shows, Music and Concerts, Theater and Performing Arts, Comedy, Celebrity News and Gossip, Anime and Manga, Podcasts, Fan Clubs (e.g., specific bands, actors), Smartphones and Mobile Devices, Computers and Laptops, Smart Home Devices, Wearable Technology, Virtual Reality (VR) and Augmented Reality (AR), Gaming Consoles and Accessories, Software and Apps, Tech News and Reviews, Astronomy and Space, Biology and Medicine, Environmental Science, Physics and Chemistry, History and Archaeology, Mathematics, Language Learning, Educational Courses and Tutorials, Nutrition and Diet, Mental Health, Meditation and Mindfulness, Alternative Medicine, Fitness Challenges, Personal Development, Sleep and Relaxation, Wellness Retreats, Adventure Travel, Cultural Travel, Budget Travel, Luxury Travel, Road Trips, Travel Tips and Hacks, Travel Photography, Destination Reviews, Parenting, Dating and Relationships, Home Decor and Interior Design, Fashion and Style, Personal Finance, Minimalism, Eco-Friendly Living, Urban Living, Gourmet Cooking, Baking, Vegan and Vegetarian, Wine and Beer Tasting, Coffee Lovers, Food Photography, Restaurant Reviews, International Cuisine, Literature and Poetry, Visual Arts, Music and Instrumental, Theater and Performing Arts, Film and Documentary, Cultural Festivals, Art Exhibitions, Craftsmanship, Entrepreneurship, Freelancing, Networking, Career Development, Industry-Specific Groups (e.g., tech, finance), Job Hunting, Mentorship, Work-Life Balance, Environmental Activism, Human Rights, Animal Welfare, Political Activism, Community Service, Charitable Organizations, Sustainable Living, Diversity and Inclusion, Specific Fandoms (e.g., Harry Potter, Star Wars), Niche Collecting (e.g., rare books, vintage items), Unique Hobbies (e.g., urban beekeeping, rock balancing), Esoteric Interests (e.g., cryptozoology, paranormal), Startup Founders, Small Business Owners, Investment and Venture Capital, Business Strategy and Management, Marketing and Sales, E-commerce, Business Networking, Leadership and Mentoring, Home Renovation, Furniture Making, Landscaping and Gardening, DIY Home Decor, Plumbing and Electrical Projects, Sustainable Living Projects, Tool and Equipment Reviews, Upcycling and Recycling, Car Enthusiasts, Motorcycles, Electric Vehicles, Car Restoration, Off-Roading, Automotive News and Reviews, Motorsport, Vehicle Maintenance and Repair, Dog Owners, Cat Lovers, Exotic Pets, Animal Rescue and Adoption, Pet Training and Behavior, Pet Nutrition and Health, Aquariums and Fishkeeping, Bird Watching, Fiction Writing, Poetry, Non-Fiction Writing, Book Clubs, Literary Analysis, Writing Workshops, Publishing and Self-Publishing, Writing Prompts and Challenges, Goal Setting, Time Management, Productivity Hacks, Mindset and Motivation, Public Speaking, Journaling, Coaching and Mentoring, Life Skills, Skincare and Makeup, Fashion Trends, Personal Styling, Beauty Tutorials, Sustainable Fashion, Haircare, Nail Art, Fashion Design, Meditation and Mindfulness, Yoga and Spiritual Practices, Religious Study Groups, Comparative Religion, Spiritual Growth, Astrology and Horoscopes, Spiritual Healing, Rituals and Ceremonies, Web Development, Mobile App Development, Data Science and Machine Learning, Cybersecurity, Cloud Computing, Software Engineering, Programming Languages, Hackathons and Coding Challenges, Historical Events, Archaeology, Genealogy, Cultural Studies, Historical Reenactments, Ancient Civilizations, Military History, Preservation and Restoration, Renewable Energy, Zero Waste Lifestyle, Sustainable Agriculture, Green Building, Environmental Policy, Eco-Friendly Products, Climate Change Action, Conservation Efforts, Stock Market, Cryptocurrency, Real Estate Investment, Personal Finance Management, Retirement Planning, Budgeting and Saving, Financial Independence, Investment Strategies, New Parents, Single Parenting, Parenting Teens, Child Development, Educational Resources for Kids, Work-Life Balance for Parents, Parenting Support Groups, Family Activities and Outings, Language Learning (e.g., Spanish, French, Mandarin), Cultural Exchange, Translation and Interpretation, Linguistics, Language Immersion Programs, Dialects and Regional Languages, Multilingual Communities, Language Teaching Resources, Mental Health Awareness, Physical Fitness Challenges, Holistic Health, Sports Psychology, Body Positivity, Mind-Body Connection, Stress Management, Chronic Illness Support, Camping and Backpacking, Bird Watching, Nature Photography, Rock Climbing, Fishing and Hunting, Wildcrafting and Foraging, Stargazing, National Parks Exploration, Pottery and Ceramics, Jewelry Making, Scrapbooking, Candle Making, Textile Arts, Glass Blowing, Woodworking, Paper Crafts, Independent Filmmaking, Screenwriting, Animation and VFX, Documentary Filmmaking, Video Editing, Cinematography, Media Critique and Analysis, Podcast Production}
Return the categories as terms(single words as given in the list) separated by commas. DON'T RETURN FULL SENTENCES.
Return the categories from the {CATEGORIES_LIST}
Return the categories as terms separated by commas in a list from the {CATEGORIES_LIST}. 
return the catgories of image urls also and video urls also and caption if provided
return results in one list

"""


generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.2, 
    "top_p": 0.95,
}

model = GenerativeModel(model_name="gemini-1.5-flash-001", system_instruction=[system_instruction])

# def fetch_and_preprocess_image(image_path):
#     """
#     Fetch and preprocess an image from a given path or URL
    
#     Args:
#         image_path (str): Path or URL of the image
    
#     Returns:
#         Part: Processed image part for Vertex AI
#     """
#     headers = {
#         "User-Agent": (
#             "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
#             "Chrome/85.0.4183.121 Safari/537.36"
#         )
#     }
    
#     try:
#         if image_path.startswith(("http://", "https://")):
#             try:
#                 response = requests.get(image_path, stream=True, headers=headers, verify=True)
#                 response.raise_for_status()
#                 image = Image.open(BytesIO(response.content))
#             except requests.RequestException as e:
#                 logging.error(f"Error fetching image {image_path}: {e}")
#                 raise
#         else:
#             image = Image.open(image_path)

#         # Resize and process the image
#         image = image.resize((224, 224), Image.Resampling.LANCZOS)
#         buffer = BytesIO()
#         image.save(buffer, format="JPEG")
#         image_bytes = buffer.getvalue()
#         return Part.from_data(mime_type="image/jpeg", data=image_bytes)

#     except Exception as e:
#         logging.error(f"Error processing image: {e}")
#         raise


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
        list: Predicted categories
    """
    # Classify URLs
    image_urls, video_urls = classify_urls(urls)
    
    # Initialize contents with caption if provided
    contents = [Part.from_text(caption)] if caption else [Part.from_text("Null")]
    unique_categories = set()

    # Process images
    for image_url in image_urls:
        print("processing image",image_url)
        try:
            image_part = fetch_and_preprocess_image(image_url)
            content = contents + [image_part]
            response = model.generate_content(content, generation_config=generation_config)
            categories = response.text.strip().split(", ")
            unique_categories.update(categories)
        except Exception as e:
            logging.error(f"Error processing image URL {image_url}: {str(e)}")

    # Process videos
    for video_url in video_urls:
        try:
            video_frames = fetch_and_preprocess_video(video_url)
            for frame in video_frames:
                content = contents + [frame]
                response = model.generate_content(content, generation_config=generation_config)
                categories = response.text.strip().split(", ")
                unique_categories.update(categories)
        except Exception as e:
            logging.error(f"Error processing video URL {video_url}: {str(e)}")

    return list(unique_categories)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    urls = data.get('urls', [])
    caption = data.get('caption', '')  # Optional caption
    
    if not urls:
        return jsonify({"error": "Please provide at least one URL."}), 400

    try:
        predicted_categories = predict_categories(urls, caption)
        predicted_category_ids = get_category_ids({"predicted_categories": predicted_categories})
        
        return jsonify({
            "predicted_categories": predicted_categories,
            "predicted_category_ids": predicted_category_ids
        }), 200
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=3400)