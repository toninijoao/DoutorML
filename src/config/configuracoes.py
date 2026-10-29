import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-base"

VECTORSTORE_NAME = "doutorml"

DATA_PATH = "data"
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processsed"
DATASETS_PATH = "data/datasets"

VECTORSTORE_PATH = "vectorstore"

MODELS_PATH = "models"
CHECKPOINTS_PATH = "models/checkpoints"
PRODUCTION_MODEL_PATH = "models/production"