################################ Imports ####################################
from os import listdir
from keras.preprocessing.image import img_to_array,load_img
from numpy import asarray, savez_compressed


################################ Load Dataset ####################################

def load_images(path, size=(256,256)):
    data_list = list()
    # enumerate filenames in directory, assume all are images
    for filename in listdir(path):
        # load and resize the image
        pixels = load_img(path + filename, target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)
        # store
        data_list.append(pixels)
        
    return asarray(data_list)


################################ Main Function ####################################
def main():

   # load images
   path = "<path to images>"
   high = load_images(path + 'ground_truth/')   #directory "path/ground_truth" should be present
   low = load_images(path + 'low/')             #directory "path/low" should be present
   
   savez_compressed('dataset.npz',high,low)

if __name__ == '__main__':
    main()