from langdetect import detect
import pickle


def pickle_save(obj, path):
    with open(path, "wb") as fOut:
        pickle.dump(obj, fOut)


def pickle_load(path):
    with open(path, "rb") as input_file:
        obj = pickle.load(input_file)
    return obj


def robust_detect(chunk):
    try:
        return detect(chunk)
    except:
        return "en"
