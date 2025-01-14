import json
from typing import Dict, List, Union

# Define the category mapping with just IDs and names
CATEGORIES = {
    1: {'name': 'Gardening'},
    2: {'name': 'Cooking and Baking'},
    3: {'name': 'DIY and Crafts'},
    4: {'name': 'Photography'},
    5: {'name': 'Reading and Book Clubs'},
    6: {'name': 'Gaming'},
    7: {'name': 'Collecting'},
    8: {'name': 'Knitting and Sewing'},
    9: {'name': 'Painting and Drawing'},
    10: {'name': 'Running'},
    11: {'name': 'Yoga and Pilates'},
    12: {'name': 'Cycling'},
    13: {'name': 'Hiking and Outdoor Activities'},
    14: {'name': 'Team Sports'},
    15: {'name': 'Swimming'},
    16: {'name': 'Fitness and Bodybuilding'},
    17: {'name': 'Martial Arts'},
    18: {'name': 'Dance'},
    19: {'name': 'Movies and TV Shows'},
    20: {'name': 'Music and Concerts'},
    21: {'name': 'Theater and Performing Arts'},
    22: {'name': 'Comedy'},
    23: {'name': 'Celebrity News and Gossip'},
    24: {'name': 'Anime and Manga'},
    25: {'name': 'Podcasts'},
    26: {'name': 'Fan Clubs'},
    27: {'name': 'Smartphones and Mobile Devices'},
    28: {'name': 'Computers and Laptops'},
    29: {'name': 'Smart Home Devices'},
    30: {'name': 'Wearable Technology'},
    31: {'name': 'Virtual Reality and Augmented Reality'},
    32: {'name': 'Gaming Consoles and Accessories'},
    33: {'name': 'Software and Apps'},
    34: {'name': 'Tech News and Reviews'},
    35: {'name': 'Astronomy and Space'},
    36: {'name': 'Biology and Medicine'},
    37: {'name': 'Environmental Science'},
    38: {'name': 'Physics and Chemistry'},
    39: {'name': 'History and Archaeology'},
    40: {'name': 'Mathematics'},
    41: {'name': 'Language Learning'},
    42: {'name': 'Educational Courses and Tutorials'},
    43: {'name': 'Nutrition and Diet'},
    44: {'name': 'Mental Health'},
    45: {'name': 'Meditation and Mindfulness'},
    46: {'name': 'Alternative Medicine'},
    47: {'name': 'Fitness Challenges'},
    48: {'name': 'Personal Development'},
    49: {'name': 'Sleep and Relaxation'},
    50: {'name': 'Wellness Retreats'},
    51: {'name': 'Adventure Travel'},
    52: {'name': 'Cultural Travel'},
    53: {'name': 'Budget Travel'},
    54: {'name': 'Luxury Travel'},
    55: {'name': 'Road Trips'},
    56: {'name': 'Travel Tips and Hacks'},
    57: {'name': 'Travel Photography'},
    58: {'name': 'Destination Reviews'},
    59: {'name': 'Parenting'},
    60: {'name': 'Dating and Relationships'},
    61: {'name': 'Home Decor and Interior Design'},
    62: {'name': 'Fashion and Style'},
    63: {'name': 'Personal Finance'},
    64: {'name': 'Minimalism'},
    65: {'name': 'Eco-Friendly Living'},
    66: {'name': 'Urban Living'},
    67: {'name': 'Gourmet Cooking'},
    68: {'name': 'Baking'},
    69: {'name': 'Vegan and Vegetarian'},
    70: {'name': 'Wine and Beer Tasting'},
    71: {'name': 'Coffee Lovers'},
    72: {'name': 'Food Photography'},
    73: {'name': 'Restaurant Reviews'},
    74: {'name': 'International Cuisine'},
    75: {'name': 'Literature and Poetry'},
    76: {'name': 'Visual Arts'},
    77: {'name': 'Music and Instrumental'},
    78: {'name': 'Theater and Performing Arts'},
    79: {'name': 'Film and Documentary'},
    80: {'name': 'Cultural Festivals'},
    81: {'name': 'Art Exhibitions'},
    82: {'name': 'Craftsmanship'},
    83: {'name': 'Entrepreneurship'},
    84: {'name': 'Freelancing'},
    85: {'name': 'Networking'},
    86: {'name': 'Career Development'},
    87: {'name': 'Industry-Specific Groups'},
    88: {'name': 'Job Hunting'},
    89: {'name': 'Mentorship'},
    90: {'name': 'Work-Life Balance'},
    91: {'name': 'Environmental Activism'},
    92: {'name': 'Human Rights'},
    93: {'name': 'Animal Welfare'},
    94: {'name': 'Political Activism'},
    95: {'name': 'Community Service'},
    96: {'name': 'Charitable Organizations'},
    97: {'name': 'Sustainable Living'},
    98: {'name': 'Diversity and Inclusion'},
    99: {'name': 'Specific Fandoms'},
    100: {'name': 'Niche Collecting'},
    101: {'name': 'Unique Hobbies'},
    102: {'name': 'Esoteric Interests'},
    103: {'name': 'Startup Founders'},
    104: {'name': 'Small Business Owners'},
    105: {'name': 'Investment and Venture Capital'},
    106: {'name': 'Business Strategy and Management'},
    107: {'name': 'Marketing and Sales'},
    108: {'name': 'E-commerce'},
    109: {'name': 'Business Networking'},
    110: {'name': 'Leadership and Mentoring'},
    111: {'name': 'Home Renovation'},
    112: {'name': 'Furniture Making'},
    113: {'name': 'Landscaping and Gardening'},
    114: {'name': 'DIY Home Decor'},
    115: {'name': 'Plumbing and Electrical Projects'},
    116: {'name': 'Sustainable Living Projects'},
    117: {'name': 'Tool and Equipment Reviews'},
    118: {'name': 'Upcycling and Recycling'},
    119: {'name': 'Car Enthusiasts'},
    120: {'name': 'Motorcycles'},
    121: {'name': 'Electric Vehicles'},
    122: {'name': 'Car Restoration'},
    123: {'name': 'Off-Roading'},
    124: {'name': 'Automotive News and Reviews'},
    125: {'name': 'Motorsport'},
    126: {'name': 'Vehicle Maintenance and Repair'},
    127: {'name': 'Dog Owners'},
    128: {'name': 'Cat Lovers'},
    129: {'name': 'Exotic Pets'},
    130: {'name': 'Animal Rescue and Adoption'},
    131: {'name': 'Pet Training and Behavior'},
    132: {'name': 'Pet Nutrition and Health'},
    133: {'name': 'Aquariums and Fishkeeping'},
    134: {'name': 'Bird Watching'},
    135: {'name': 'Fiction Writing'},
    136: {'name': 'Poetry'},
    137: {'name': 'Non-Fiction Writing'},
    138: {'name': 'Book Clubs'},
    139: {'name': 'Literary Analysis'},
    140: {'name': 'Writing Workshops'},
    141: {'name': 'Publishing and Self-Publishing'},
    142: {'name': 'Writing Prompts and Challenges'},
    143: {'name': 'Goal Setting'},
    144: {'name': 'Time Management'},
    145: {'name': 'Productivity Hacks'},
    146: {'name': 'Mindset and Motivation'},
    147: {'name': 'Public Speaking'},
    148: {'name': 'Journaling'},
    149: {'name': 'Coaching and Mentoring'},
    150: {'name': 'Life Skills'},
    151: {'name': 'Skincare and Makeup'},
    152: {'name': 'Fashion Trends'},
    153: {'name': 'Personal Styling'},
    154: {'name': 'Beauty Tutorials'},
    155: {'name': 'Sustainable Fashion'},
    156: {'name': 'Haircare'},
    157: {'name': 'Nail Art'},
    158: {'name': 'Fashion Design'},
    159: {'name': 'Meditation and Mindfulness'},
    160: {'name': 'Yoga and Spiritual Practices'},
    161: {'name': 'Religious Study Groups'},
    162: {'name': 'Comparative Religion'},
    163: {'name': 'Spiritual Growth'},
    164: {'name': 'Astrology and Horoscopes'},
    165: {'name': 'Spiritual Healing'},
    166: {'name': 'Rituals and Ceremonies'},
    167: {'name': 'Web Development'},
    168: {'name': 'Mobile App Development'},
    169: {'name': 'Data Science and Machine Learning'},
    170: {'name': 'Cybersecurity'},
    171: {'name': 'Artificial Intelligence'},
    172: {'name': 'Cybersecurity'},
    173: {'name': 'Cloud Computing'},
    174: {'name': 'Software Engineerin'},
    175: {'name': 'Programming Languages'},
    176: {'name': 'Hackathons and Coding Challenges'},
    177: {'name': 'Historical Events'},
    178: {'name': 'Archaeology'},
    179: {'name': 'Genealogy'},
    180: {'name': 'Cultural Studies'},
    181: {'name': 'Historical Reenactments'},
    182: {'name': 'Ancient Civilizations'},
    183: {'name': 'Military History'},
    184: {'name': 'Preservation and Restoration'},
    185: {'name': 'Renewable Energy'},
    186: {'name': 'Zero Waste Lifestyle'},
    187: {'name': 'Sustainable Agriculture'},
    188: {'name': 'Green Building'},
    189: {'name': 'Environmental Policy'},
    190: {'name': 'Eco-Friendly Products'},
    191: {'name': 'Climate Change Action'},
    192: {'name': 'Conservation Efforts'},
    193: {'name': 'Stock Market'},
    194: {'name': 'Cryptocurrency'},
    195: {'name': 'Real Estate Investment'},
    196: {'name': 'Personal Finance Management'},
    197: {'name': 'Retirement Planning'},
    198: {'name': 'Budgeting and Saving'},
    199: {'name': 'Financial Independence'},
    200: {'name': 'Investment Strategie'},
    201: {'name': 'New Parents'},
    202: {'name': 'Single Parenting'},
    203: {'name': 'Parenting Teens'},
    204: {'name': 'Child Development'},
    205: {'name': 'Educational Resources for Kids'},
    206: {'name': 'Work-Life Balance for Parents'},
    207: {'name': 'Parenting Support Groups'},
    208: {'name': 'Family Activities and Outings'},
    209: {'name': 'Family Activities and Outings'},
    210: {'name': 'Language Learning'},
    211: {'name': 'Cultural Exchange'},
    212: {'name': 'Translation and Interpretation'},
    213: {'name': 'Linguistics'},
    214: {'name': 'Language Immersion Programs'},
    215: {'name': 'Dialects and Regional Languages'},
    216: {'name': 'Multilingual Communities'},
    217: {'name': 'Language Teaching Resources'},
    218: {"name":"Mental Health Awareness"},
    219: {"name":"Physical Fitness Challenges"},
    220: {"name":"Holistic Health"},
    221: {"name":"Sports Psychology"},
    222: {"name":"Body Positivity"},
    223: {"name":"Mind-Body Connection"},
    224: {"name":"Stress Management"},
    225: {"name":"Chronic Illness Support"},
    226: {"name":"Camping and Backpacking"},
    227: {"name":"Bird Watching"},
    228: {"name":"Nature Photography"},
    229: {"name":"Rock Climbing"},
    230: {"name":"Fishing and Hunting"},
    231: {"name":"Wildcrafting and Foraging"},
    232: {"name":"Stargazing"},
    233: {"name":"National Parks Exploration"},
    234: {"name":"Pottery and Ceramics"},
    235: {"name":"Jewelry Making"},
    236: {"name":"Scrapbooking"},
    237: {"name":"Candle Making"},
    238: {"name":"Textile Arts"},
    239: {"name":"Glass Blowing"},
    240: {"name":"Woodworking"},
    241: {"name":"Paper Crafts"},
    242: {"name":"Independent Filmmaking"},
    243: {"name":"Screenwriting"},
    244: {"name":"Animation and VFX"},
    245: {"name":"Documentary Filmmaking"},
    246: {"name":"Video Editing"},
    247: {"name":"Cinematography"},
    248: {"name":"Media Critique and Analysis"},
    249: {"name":"Podcast Production"},
}


def get_category_ids(input_data: Union[str, Dict]) -> List[int]:
    """
    Get category IDs from JSON input containing predicted categories.
    
    Args:
        input_data: Either a JSON string or dictionary with 'predicted_categories' key
        
    Returns:
        List[int]: List of category IDs
    """
    # Handle string input
    if isinstance(input_data, str):
        try:
            data = json.loads(input_data)
        except json.JSONDecodeError:
            return []
    else:
        data = input_data
    
    # Extract categories
    categories = data.get('predicted_categories', [])
    
    # Get IDs
    ids = []
    for name in categories:
        normalized_name = name.strip()
        for cat_id, cat_info in CATEGORIES.items():
            if cat_info['name'] == normalized_name:
                ids.append(cat_id)
                break
    
    return ids

if __name__=="__main__":

    # Example usage:
    input_json = '''
    {
        "predicted_categories": [
            "Photography",
            "Hiking and Outdoor Activities",
            "Dog Owners",
            "Travel Photography"
        ]
    }
    '''

    # Test with string input
    ids = get_category_ids(input_json)
    print(ids)
    # Output: [4, 13, 101, 57]

    # Test with dictionary input
    # input_dict = {
    #     "predicted_categories": [
    #         "Photography",
    #         "Hiking and Outdoor Activities",
    #         "Dog Owners",
    #         "Travel Photography"
    #     ]
    # }

    input={
        "predicted_categories": [
            "Animal Welfare",
            "Photography",
            "Bird Watching"
        ]
    }
    ids = get_category_ids(input)
    print(ids)
    Output: [4, 13, 101, 57]