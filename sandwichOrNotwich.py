import ImageTools
import pickle

def main():


    model_file = open('model.pkl', 'rb')
    model = pickle.load(model_file)

    ## TODO:
    hotdog_files = ImageTools.parseImagePaths('./img/hotdog/')
    # Preprocess the hotdog files, just like what was done in trainModel.py
    # results = model.predict(x), where x is an array of all preprocesed images
    # Take mean and standard deviation of results as the data result
