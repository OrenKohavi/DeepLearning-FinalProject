#This file trains the neural network with the training data, and then tests the network

from pickletools import uint8
import numpy as np
import random
import os
from PIL import Image #install using 'pip install Pillow'
from network import BlockCNN

def main(train_inputs, train_labels, test_inputs, test_labels, num_epochs, batch_size):
    create_img_output = True
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

def np_to_img(np_array, output_shape, crop_center=False, step=8):
    #Converts a np array to PIL image
    #crop_center should be true when the np_array should be cropped to the tile size (hardcoded to 8x8)
    if crop_center:
        num_tiles = (output_shape[0]*output_shape[1]) // (step*step)
        new_np_array = np.zeros(shape=(num_tiles,step,step,3)) #Hardcoded for 3 channels
        for i in range(len(np_array)):
            new_np_array[i] = np_array[i][step:2*step, step:2*step, :]
        np_array = new_np_array
    img_width = output_shape[0]
    img_height = output_shape[1]
    img_array = np.zeros(shape=output_shape)
    for w in range(0, img_width, step):
        for h in range(0, img_height, step):
            this_tile = np_array[(w//step)*(img_height//step) + (h//step)]
            this_tile = np.transpose(this_tile, axes=(1,0,2))
            img_array[w:w+step, h:h+step, :] = this_tile
    img_array = img_array * 255
    img_array = img_array.astype(np.uint8)
    img_array = np.transpose(img_array, axes=(1,0,2))
    return Image.fromarray(img_array)


if __name__ == "__main__":
    data_path = "./data/"
    num_epochs = 10
    batch_size = 128
    main(data_path+"train_inputs.npy", data_path+"train_labels.npy", data_path+"test_inputs.npy", data_path+"test_labels.npy", num_epochs, batch_size)