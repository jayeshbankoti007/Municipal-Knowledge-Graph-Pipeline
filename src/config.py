from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "transcripts"
METADATA_CSV = BASE_DIR / "data" / "metadata.csv"
OUTPUT_DIR = BASE_DIR / "output"
EXTRACTIONS_DIR = OUTPUT_DIR / "extractions"
RESOLUTION_FILE = OUTPUT_DIR / "resolved_entities_dict.json"
KG_FILE_PATH = OUTPUT_DIR / "knowledge_graph.pkl"
KG_NEO4J_PATH = OUTPUT_DIR / "knowledge_graph.graphml"

# Create directories
EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

MAX_RETRIES = 3
TEMPERATURE = 0.1
MODEL_NAME = "gpt-4o-mini"

# Processing Configuration
MAX_CHARS_PER_TRANSCRIPT = 2500  # ~15k tokens
PREPROCESSING_MODEL = "gpt-4.1-nano"         
PREPROCESSED_INTERMEDIATE_TOKENS = 30000          
PREPROCESSED_TARGET_TOKENS = 2000
