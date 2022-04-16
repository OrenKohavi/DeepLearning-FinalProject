## Splits up the existing images in Kodak_Full_Images into training data and labels
## Numbers the inputs and outputs, and places them in the ./data folder

import os
from PIL import Image
from numpy import imag #install using 'pip install Pillow'

#Params for the script:
train_test_split = 0.2 #20% of data reserved for testing
include_border_blocks = True #if True, will include the edges of the images, padding with 0
tile_size = 8

def main(source_filepath, dest_filepath):
    lossless_images = get_images(source_filepath) #These are labels
    #Compress these images, write them somewhere, and then re-read them to make inputs
    for img in lossless_images:
        img.save(dest_filepath + "/compressed_whole_images")
    compressed_images
    #Process the image into 8x8 chunks
    img_chunks = []


def get_images(filepath):
    filenames = os.listdir(filepath)
    assert all([f.endswith(".png") for f in filenames]), "Non-PNG found in source data"
    images = []
    #Read all images
    for img in filenames:
        images.append(Image.open(filepath + "/" + img))
    #Sanity-check that the image dimentions are right
    expected_dim = (768,512)
    for img in images:
        if img.size == expected_dim:
            #Good! This is the correct resolution
            pass
        elif img.size == (expected_dim[1],expected_dim[0]):
            #Also fine, just rotate the image 90 degrees so that it fits with the rest
            img = img.rotate(90, expand=True)
            #Now, it should be the right size
            assert img.size == expected_dim
        else:
            #This is a weird size, raise an error!
            raise ValueError(f"Images have bad size! Expected {expected_dim} but got {img.size}")
    return images

if __name__ == "__main__":
    main(source_filepath="./Kodak_Full_Images", dest_filepath="./data")