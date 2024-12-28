import logging
from .classifier import ResNet18, ResNet50
from .classifierWrapper import ClassifierConfig, ClassifierWrapper

def create_classifier(model_name: str, n_classes: int, config: ClassifierConfig) :

    if model_name == "RN18":
        model = ResNet18(n_classes)

    elif model_name == "RN50":
        model = ResNet50(n_classes)
        
    else:
        logging.error(f"The model name '{model_name}' is not supported.")
        raise ValueError()
    
    return ClassifierWrapper(model, config)