import logging
import logging.config
import yaml
import sys, getopt

from util.configuration import TrainingConfiguration
from util.data_loader import DataLoader
from spacy_model.model_training import SpacyTraining
from rasa_model.model_training import RasaTraining
from transformers_model.model_training import TransformersTraining

models = ["spacy", "rasa", "transformers"]

if __name__ == "__main__":

    # inizialize logging
    with open("util/logging/logging.yml", 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(config)
    logger = logging.getLogger("ModelTraining")

    # get arguments
    input_path = "../datasets/KeyFiTax/KeyFiTax_data.json"
    model = "transformers"
    training_config = TrainingConfiguration()

    try:
        opts, args = getopt.getopt(sys.argv[1:],"hf:m:n:t:",["file=","model=","folds=","transformer="])
    except getopt.GetoptError:
        logger.error("run_training.py -f <training data as JSON> -m <model: spacy/rasa/transformers> -n <number of folds> -t <pretrained transformer model>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            logger.info("run_training.py -f <training data as JSON> -m <model: spacy/rasa/transformers> -n <number of folds> -t <pretrained transformer model>")
            sys.exit()
        elif opt in ('-f', '--file'):
            input_path = arg
        elif opt in ('-m', '--model') and arg.lower() in models:
            model = arg.lower()
            training_config.model_name += arg.lower()
        elif opt in ("-n", "--folds") and isinstance(arg, int):
            training_config.num_folds = arg
        elif opt in ("-t", "--transformer"):
            training_config.transformer_model = arg

    # load and tokenize data
    data_loader = DataLoader(input_path, training_config.spacy_model)
    data_train, data_test  = data_loader.get_data_split(proportion=0.8)

    # define classification and instantiate Trainer
    logger.info(f"Train model for Key Figure classes")
    if model == "transformers":
        model_training = TransformersTraining(data_train, data_test)
        training_config.classification_file = "../datasets/KeyFiTax/classification.json"
    elif model == "spacy": 
        model_training = SpacyTraining(data_train, data_test)
        training_config.classification_file = "../datasets/KeyFiTax/classification_keyfigure.json"
        training_config.model_name += "_keyfigure"
    elif model == "rasa": 
        model_training = RasaTraining(data_train, data_test)
        training_config.classification_file = "../datasets/KeyFiTax/classification_keyfigure.json"
        training_config.model_name += "_keyfigure"

    # perform training and evaluation
    model_path, evaluation = model_training.run_training(training_config)
    logger.info(f"Model is stored as {model_path}")
    logger.info(f"Evaluation results: {evaluation}")

    # train separate model for 'condition' class
    if model in ["spacy", "rasa"]: 
        logger.info(f"Train model for 'Condition' class")
        training_config.classification_file = "../datasets/KeyFiTax/classification_condition.json"
        training_config.model_name = training_config.model_name[:training_config.model_name.rfind("_")] + "_condition"

        model_path, evaluation = model_training.run_training(training_config)
        logger.info(f"Model is stored as {model_path}")
        logger.info(f"Evaluation results: {evaluation}")
