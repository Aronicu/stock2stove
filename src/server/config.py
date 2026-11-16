import os

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SERVER_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
TEMPLATES_DIR = os.path.join(DATA_DIR, "templates")
RAW_HOME_DIR = os.path.join(DATA_DIR, "raw_home")
DATASET_YOLO_DIR = os.path.join(DATA_DIR, "dataset_yolo")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

FINE_TUNED_PT_PATH = os.path.join(MODELS_DIR, "yolov8n_stock2stove.pt")
FINE_TUNED_ONNX_PATH = os.path.join(MODELS_DIR, "yolov8n_stock2stove.onnx")
YOLO_WORLD_PT_PATH = os.path.join(MODELS_DIR, "yolov8s-worldv2.pt")
YOLO_COCO_PT_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")

HOST = os.getenv("STOCK2STOVE_HOST", "0.0.0.0")
PORT = int(os.getenv("STOCK2STOVE_PORT", "8000"))

TARGET_IMG_SIZE = 640
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
LETTERBOX_PAD_COLOR = (114, 114, 114)

DEFAULT_CONF_THRESHOLD = 0.50
NMS_IOU_THRESHOLD = 0.45
TEMPORAL_WINDOW_SIZE = 5
TEMPORAL_MIN_HITS = 3

ORB_NFEATURES = 1000
SIFT_NFEATURES = 1000
LOWE_RATIO_THRESHOLD = 0.75
RANSAC_REPROJ_THRESHOLD = 5.0
MIN_HOMOGRAPHY_INLIERS = 6

LLM_TIMEOUT_SECONDS = 2.5

SUPPORTED_DETECTOR_MODES = [
    "yolov8",
    "orb_baseline",
    "sift_baseline",
    "yolo_world",
]

UNIFIED_CLASSES = [
    {
        "id": 0,
        "class_id": "pasta_noodles",
        "display_name": "Pasta & Noodles",
        "storage_zone": "Pantry",
        "prompts": ["pasta box", "spaghetti pack", "macaroni box", "instant noodles", "pasta"],
    },
    {
        "id": 1,
        "class_id": "canned_tomato",
        "display_name": "Canned Tomatoes",
        "storage_zone": "Pantry",
        "prompts": ["canned tomatoes", "tomato sauce can", "tomato paste", "crushed tomatoes"],
    },
    {
        "id": 2,
        "class_id": "canned_tuna_meat",
        "display_name": "Canned Tuna & Meat",
        "storage_zone": "Pantry",
        "prompts": ["tuna can", "canned tuna", "corned beef can", "sardines can", "canned meat"],
    },
    {
        "id": 3,
        "class_id": "cereal_grains",
        "display_name": "Cereal & Grains",
        "storage_zone": "Pantry",
        "prompts": ["cereal box", "oatmeal carton", "rice pack", "grains box"],
    },
    {
        "id": 4,
        "class_id": "coffee_condiment",
        "display_name": "Coffee & Condiments",
        "storage_zone": "Pantry",
        "prompts": ["coffee bag", "coffee jar", "cooking oil bottle", "soy sauce bottle", "condiment jar"],
    },
    {
        "id": 5,
        "class_id": "milk_dairy",
        "display_name": "Milk & Dairy",
        "storage_zone": "Refrigerator",
        "prompts": ["milk carton", "cheese block", "butter", "yogurt cup", "dairy"],
    },
    {
        "id": 6,
        "class_id": "eggs",
        "display_name": "Eggs",
        "storage_zone": "Refrigerator",
        "prompts": ["eggs", "egg carton", "egg tray"],
    },
    {
        "id": 7,
        "class_id": "fresh_produce",
        "display_name": "Fresh Produce",
        "storage_zone": "Refrigerator",
        "prompts": ["fresh tomato", "onion", "garlic", "bell pepper", "carrot", "vegetables"],
    },
]

CLASS_ALIASES = {
    "pasta_noodles": "pasta_noodles",
    "pasta_box": "pasta_noodles",
    "pasta": "pasta_noodles",
    "noodles": "pasta_noodles",
    "spaghetti": "pasta_noodles",
    "canned_tomato": "canned_tomato",
    "tomato_can": "canned_tomato",
    "canned tomatoes": "canned_tomato",
    "tomato_sauce": "canned_tomato",
    "canned_tuna_meat": "canned_tuna_meat",
    "tuna_can": "canned_tuna_meat",
    "canned_tuna": "canned_tuna_meat",
    "tuna": "canned_tuna_meat",
    "corned_beef": "canned_tuna_meat",
    "cereal_grains": "cereal_grains",
    "cereal_box": "cereal_grains",
    "cereal": "cereal_grains",
    "oats": "cereal_grains",
    "rice": "cereal_grains",
    "coffee_condiment": "coffee_condiment",
    "coffee_bag": "coffee_condiment",
    "coffee": "coffee_condiment",
    "condiment": "coffee_condiment",
    "oil": "coffee_condiment",
    "milk_dairy": "milk_dairy",
    "milk": "milk_dairy",
    "cheese": "milk_dairy",
    "butter": "milk_dairy",
    "yogurt": "milk_dairy",
    "eggs": "eggs",
    "egg": "eggs",
    "fresh_produce": "fresh_produce",
    "vegetables": "fresh_produce",
    "garlic": "fresh_produce",
    "onion": "fresh_produce",
    "tomato": "fresh_produce",
}


def normalize_class_id(raw_name):
    if not raw_name:
        return "pasta_noodles"
    cleaned = str(raw_name).strip().lower()
    if cleaned in CLASS_ALIASES:
        return CLASS_ALIASES[cleaned]
    underscored = cleaned.replace(" ", "_").replace("-", "_")
    if underscored in CLASS_ALIASES:
        return CLASS_ALIASES[underscored]
    for entry in UNIFIED_CLASSES:
        if entry["class_id"] == underscored:
            return entry["class_id"]
        for prompt in entry["prompts"]:
            if prompt == cleaned or prompt in cleaned or cleaned in prompt:
                return entry["class_id"]
    return underscored


def get_display_name(class_id):
    canonical = normalize_class_id(class_id)
    for entry in UNIFIED_CLASSES:
        if entry["class_id"] == canonical:
            return entry["display_name"]
    if class_id == "pasta_box":
        return "Pasta Box"
    if class_id == "tomato_can":
        return "Canned Tomatoes"
    words = str(class_id).replace("_", " ").split()
    capitalized = []
    for word in words:
        capitalized.append(word.capitalize())
    return " ".join(capitalized)


def get_storage_zone(class_id):
    canonical = normalize_class_id(class_id)
    for entry in UNIFIED_CLASSES:
        if entry["class_id"] == canonical:
            return entry["storage_zone"]
    return "Pantry"
