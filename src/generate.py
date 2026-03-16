
# Load config
def load_config(config_file_path):
    with open(config_file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except:
            logger.info(f"Error reading yaml file")
            return None

# Set Logger
log_dir = config["paths"]["log_dir"]
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "train.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    
def load_tokenizer():
    
def load_model():
    
def tokenize():

def generate():
    
def decode():
    
