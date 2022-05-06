## Splits up the existing images in Kodak_Full_Images into training data and labels
## creates .npy files corresponding to train/test inputs and labels in the './data' folder
## This only needs to be run ONCE after cloning the repository

import random
import os
from PIL import Image #install using 'pip install Pillow'
import numpy as np

#Params for the script:
train_test_split = 0.2 #20% of data reserved for testing
#It's much easier to include borders, so it's hardcoded to True for now.
#include_border_blocks = True #if True, will include the edges of the images, padding with 0
jpeg_compression_amount = 25 #Number between 1 and 95, with higher being less compressed
tile_size = 8
expected_dim = (768,512)
output_image_previews = False #Creates a preview_inputs and preview_labels directory to see inputs/labels in image format
num_image_previews = 1000 #If output_image_previews is True, how many previews to make

#Paths within the dest_filepath for different parts of the data
lossless_images_path = "/raw_images"
lossy_images_path = "/compressed_images"

def main(source_filepath : str, dest_filepath : str):
    lossless_images = get_images(source_filepath, ext=".png") #These are labels
    #Compress these images, write them somewhere, and then re-read them to make inputs
    #Make sure correct directories exist (or make them if needed)
    if not os.path.isdir(dest_filepath + lossless_images_path):
        os.mkdir(dest_filepath + lossless_images_path)
    if not os.path.isdir(dest_filepath + lossy_images_path):
        os.mkdir(dest_filepath + lossy_images_path)
    for idx, img in enumerate(lossless_images):
        #Save once as JPEG, another as lossless PNG
        img.save(dest_filepath + lossless_images_path + "/" + f"{idx}".zfill(2) + ".png") #PNG is lossless
        img.save(dest_filepath + lossy_images_path + "/" + f"{idx}".zfill(2) + ".jpeg", quality=jpeg_compression_amount)
    compressed_images = get_images(dest_filepath + lossy_images_path, ext=".jpeg")
    #Process the image into tiles
    img_width, img_height = expected_dim
    assert img_width % tile_size == 0, "Image width not divisible by tile size"
    assert img_height % tile_size == 0, "Image height not divisible by tile size"
    print(f"Reading {len(compressed_images)} images...")
    #Create 3x3 tile-sized inputs
    inputs = make_tiles(compressed_images,step_size=tile_size, make_inputs=True)
    labels = make_tiles(lossless_images, step_size=tile_size, make_inputs=False)
    assert len(inputs) == len(labels), "Number of inputs and labels do not match!"
    #print(f"Generated {len(inputs)} inputs and {len(labels)} labels")
    #Convert it all to a numpy array
    numpy_inputs = []
    numpy_labels = []
    for img in inputs:
        numpy_inputs.append(np.array(img))
    for img in labels:
        numpy_labels.append(np.array(img))
    numpy_inputs = np.array(numpy_inputs)
    numpy_labels = np.array(numpy_labels)
    #Perfect! We now have numpy arrays of images and labels!
    assert numpy_inputs.shape[0] == numpy_labels.shape[0], "Unequal number of Inputs and Labels generated!"
    #Split into train and test sets
    train_test_split_index = int(numpy_inputs.shape[0] * train_test_split)
    test_inputs = numpy_inputs[:train_test_split_index]
    test_labels = numpy_labels[:train_test_split_index]
    train_inputs = numpy_inputs[train_test_split_index:]
    train_labels = numpy_labels[train_test_split_index:]
    #Convert numpy arrays to float32 type
    train_inputs = np.divide(train_inputs, 255.0)
    test_inputs = np.divide(test_inputs, 255.0)
    train_labels = np.divide(train_labels, 255.0)
    test_labels = np.divide(test_labels, 255.0)
    #save to .npy files
    np.save(dest_filepath + "/train_inputs", train_inputs)
    np.save(dest_filepath + "/train_labels", train_labels)
    np.save(dest_filepath + "/test_inputs", test_inputs)
    np.save(dest_filepath + "/test_labels", test_labels)

    #if output_image_previews is True, save inputs/labels as png
    if output_image_previews:
        if not os.path.isdir(dest_filepath + "/preview_labels"):
            os.mkdir(dest_filepath + "/preview_labels")
        if not os.path.isdir(dest_filepath + "/preview_inputs"):
            os.mkdir(dest_filepath + "/preview_inputs")
        num_digits = len(str(num_image_previews - 1))
        for idx, img in enumerate(labels[:num_image_previews]):
            img.save(dest_filepath + "/preview_labels/" + f"{idx}".zfill(num_digits) + ".png")
        for idx, img in enumerate(inputs[:num_image_previews]):
            img.save(dest_filepath + "/preview_inputs/" + f"{idx}".zfill(num_digits) + ".png")
    
def make_tiles(images : list, step_size : int, make_inputs : bool) -> list:
    output_size = step_size*3 if make_inputs else step_size
    results = []
    for img in images:
        width, height = img.size
        start_offset = 0
        if make_inputs:
            start_offset = -step_size
            width = width - step_size
            height = height - step_size
        for w in range(start_offset,width,step_size):
            for h in range(start_offset,height,step_size):
                results.append(img.crop((w,h,w+output_size,h+output_size)))
    return results

def get_images(filepath : str, ext : str) -> list:
    filenames = os.listdir(filepath)
    if not all([f.lower().endswith(ext) for f in filenames]):
        bad_files = [f for f in filenames if not f.lower().endswith(".png")]
        raise ValueError(f"Non {ext} files found in Source Data: Offending files are: {bad_files}")
    images = []
    #Read all images
    for img in filenames:
        images.append(Image.open(filepath + "/" + img))
    #Sanity-check that the image dimentions are right
    new_images = []
    for img in images:
        if img.size == expected_dim:
            #Good! This is the correct resolution
            print("Found image with correct resolution")
            new_images.append(img)
        elif img.size == (expected_dim[1],expected_dim[0]):
            print("Found image with incorrect rotation, but correct dimensions")
            #Also fine, just rotate the image 90 degrees so that it fits with the rest
            img = img.rotate(90, expand=True)
            #Now, it should be the right size
            assert img.size == expected_dim
            new_images.append(img)
        else:
            #This is a weird size, raise an error!
            raise ValueError(f"Images have bad size! Expected {expected_dim} but got {img.size}")
    return new_images

if __name__ == "__main__":
    if not os.path.isdir("./Kodak_Full_Images"):
        raise Exception("Could not find source directory")
    if not os.path.isdir("./data"):
        #That's ok! We can make it
        os.mkdir("./data")
    print("Creating Data...")
    main(source_filepath="./Kodak_Full_Images", dest_filepath="./data")
    print("Done! 🥳🎉")