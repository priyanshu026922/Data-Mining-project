import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(SAVE_DIR, exist_ok=True)


def save_model(model, filename):

    path = os.path.join(SAVE_DIR, f"{filename}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved: {path}")


def load_model(filename):
  
    path = os.path.join(SAVE_DIR, f"{filename}.pkl")
    if not os.path.exists(path):

        print(f" No saved model found at: {path} ")
        return None
    

    with open(path, 'rb') as f:
        try:
            model = pickle.load(f)
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None
         
    print(f"Model loaded: {path} ")

    return model


def save_all_models(models_dict):
    """
    Saves multiple models at once.
    models_dict: { 'filename': model_object }
    Example: save_all_models({'knn': knn_model, 'rf': rf_model})
    """

    print("\nSaving all models...")
    for filename, model in models_dict.items():
        save_model(model, filename)
    print("All models saved!\n")


def load_all_models(filenames):

    print("\nLoading all models...")
    models = {}
    for filename in filenames:
        model = load_model(filename)
        if model is not None:
            models[filename] = model
    print("  All models loaded!\n")
    return models


def models_exist(filenames):

    for filename in filenames:
        path = os.path.join(SAVE_DIR, f"{filename}.pkl")
        if not os.path.exists(path):
            return False
    return True