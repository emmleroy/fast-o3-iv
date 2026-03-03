from pathlib import Path

#from dotenv import load_dotenv
#load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJ_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
