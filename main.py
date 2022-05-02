#This file trains the neural network with the training data, and then tests the network

from pickletools import uint8
import numpy as np
import random
import os
from PIL import Image #install using 'pip install Pillow'
from network import BlockCNN

def main(train_inputs, train_labels, test_inputs, test_labels, num_epochs, batch_size):
    create_img_output = False
    #^ Set to True if you want to see the images in the output folder
    #^ Currently, image reconstruction is super broken, so this is disabled
    
    #Load the data
    train_inputs = np.load(train_inputs)
    train_labels = np.load(train_labels)
    test_inputs = np.load(test_inputs)
    test_labels = np.load(test_labels)

    #Shuffle the data
    temp = list(zip(train_inputs, train_labels))
    random.shuffle(temp)
    ti, tl = zip(*temp)
    shuffled_train_inputs, shuffled_train_labels = np.array(list(ti)), np.array(list(tl))

    temp = list(zip(test_inputs, test_labels))
    random.shuffle(temp)
    ti, tl = zip(*temp)
    shuffled_test_inputs, shuffled_test_labels = np.array(list(ti)), np.array(list(tl))

    all_inputs = np.concatenate((test_inputs, train_inputs))
    all_labels = np.concatenate((test_labels, train_labels))
    del(train_inputs, train_labels, test_inputs, test_labels) #free up memory, since these are huge arrays

    input_shape=(24,24,3)
    label_shape=(8,8,3)
    #Make sure the expected shape is correct
    assert shuffled_train_inputs.shape[1:] == input_shape
    assert shuffled_train_labels.shape[1:] == label_shape
    assert shuffled_test_inputs.shape[1:] == input_shape
    assert shuffled_test_labels.shape[1:] == label_shape

    model = BlockCNN()
    model.compile(optimizer='adam', loss=model.loss)
    model.fit(shuffled_train_inputs, shuffled_train_labels, epochs=num_epochs, batch_size=batch_size)
    #The model is now trained!
    #Now, run the model on the test data
    loss = model.evaluate(shuffled_test_inputs, shuffled_test_labels)
    print(f"Model loss: {loss}")
    #Create some human-viewable examples with the non-shuffled data
    if create_img_output:
        image_output_size = (768,512)
        tile_size = label_shape[0]
        assert (image_output_size[0]*image_output_size[1]) % (tile_size*tile_size) == 0
        num_tiles_per_image = (image_output_size[0]*image_output_size[1]) // (tile_size*tile_size)
        which_image = 17
        one_image = all_inputs[which_image*num_tiles_per_image:(which_image+1)*num_tiles_per_image]
        one_image_label = all_labels[which_image*num_tiles_per_image:(which_image+1)*num_tiles_per_image]
        label_img = np_to_img(one_image_label, (768,512,3))
        input_img = np_to_img(one_image, (768,512,3), crop_center=True)
        one_image = model.predict(one_image)
        output_img = np_to_img(one_image, (768,512,3))
        if not os.path.isdir("./output"):
            os.mkdir("./output")
        label_img.save("./output/nn_label.png")
        input_img.save("./output/nn_input.png")
        output_img.save("./output/nn_output.png")
        print("Done! 🥳🎉")

def np_to_img(np_array, output_shape, crop_center=False):
    #Converts a np array to PIL image
    #crop_center should be true when the np_array should be cropped to the tile size (hardcoded to 8x8)
    if crop_center:
        new_np_array = np.zeros(shape=(6144,8,8,3)) #Hardcoded for 8x8 tiles and 768x512 image
        for i in range(len(np_array)):
            new_np_array[i] = np_array[i][8:16, 8:16, :]
        np_array = new_np_array
    np_array = np.reshape(np_array, output_shape)
    #Multiply by 255 to convert type back to uint8
    np_array = np_array * 255
    np_array = np_array.astype(np.uint8)
    return Image.fromarray(np.transpose(np_array, axes=(1,0,2)))


if __name__ == "__main__":
    data_path = "./data/"
    num_epochs = 1
    batch_size = 512
    main(data_path+"train_inputs.npy", data_path+"train_labels.npy", data_path+"test_inputs.npy", data_path+"test_labels.npy", num_epochs, batch_size)