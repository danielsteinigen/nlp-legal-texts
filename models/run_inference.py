import logging
import logging.config
import yaml
import sys, getopt
import json

from util.configuration import InferenceConfiguration
from util.data_loader import DataLoader
from spacy_model.model_inference import SpacyInference
from rasa_model.model_inference import RasaInference
from transformers_model.model_inference import TransformersInference

models = ["spacy", "rasa", "transformers"]

if __name__ == "__main__":

    # inizialize logging
    with open("util/logging/logging.yml", 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(config)
    logger = logging.getLogger("ModelInference")

    # get arguments
    input_path = "inference_sample.json"
    output_path = "inference_output.json"
    model = "transformers"
    inference_config = InferenceConfiguration()

    try:
        opts, args = getopt.getopt(sys.argv[1:],"hf:m:p:c:t:o:",["file=","model=","path=","cond=","transformer=","output="])
    except getopt.GetoptError:
        logger.error("run_inference.py -f <paragraphs as JSON> -m <model: spacy/rasa/transformers> -p <path to trained model for key figure classes> -c <path to trained model for condition class> -t <pretrained transformer model>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            logger.info("run_inference.py -f <paragraphs as JSON> -m <model: spacy/rasa/transformers> -p <path to trained model for key figure classes> -c <path to trained model for condition class> -t <pretrained transformer model>")
            sys.exit()
        elif opt in ('-f', '--file'):
            input_paragraph = arg
        elif opt in ('-m', '--model') and arg.lower() in models:
            model = arg.lower()
        elif opt in ("-p", "--path"):
            inference_config.model_path_keyfigure = arg
        elif opt in ("-c", "--cond"):
            inference_config.model_path_condition = arg
        elif opt in ("-t", "--transformer"):
            inference_config.transformer_model = arg
        elif opt in ("-o", "--output"):
            output_path = arg

    # load and tokenize data
    data_loader = DataLoader(input_path, inference_config.spacy_model)
    data = data_loader.get_data_all()

    # define classification and instantiate Model
    logger.info(f"Train model for Key Figure classes")
    if model == "transformers":
        model_inference = TransformersInference(inference_config)
    elif model == "spacy": 
        model_inference = SpacyInference(inference_config)
    elif model == "rasa": 
        model_inference = RasaInference(inference_config)

    # perform model inference
    model_inference.run_inference(data)

    # store inference results in file
    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump({"data": data.dict()["samples"]}, f, ensure_ascii=False, indent=4)

    logger.info(f"Result is stored at {output_path}")

