#This file trains the neural network with the training data, and then tests the network

import tensorflow as tf
import numpy as np
import random
import os
from PIL import Image #install using 'pip install Pillow'
from network import BlockCNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Suppress TensorFlow warnings

def main(train_inputs, train_labels, test_inputs, test_labels, num_epochs, batch_size):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    create_img_output = True #Enabling this will output the results of the network to the outputs folder
    verbose_img_output = False #Enabling this will also output inputs and labels to the outputs folder
    save_weights = False

    #Shuffle the training data
    temp = list(zip(train_inputs, train_labels))
    random.shuffle(temp)
    ti, tl = zip(*temp)
    shuffled_train_inputs, shuffled_train_labels = np.array(list(ti)), np.array(list(tl))

    #Shuffle the testing data
    temp = list(zip(test_inputs, test_labels))
    random.shuffle(temp)
    ti, tl = zip(*temp)
    shuffled_test_inputs, shuffled_test_labels = np.array(list(ti)), np.array(list(tl))

    if create_img_output:
        #Create the output folder
        if not os.path.isdir("./output"):
            os.mkdir("./output")
        #Create arrays with original (non-shuffled) inputs and labels for reconstruction later
        all_inputs = np.concatenate((test_inputs, train_inputs))
        all_labels = np.concatenate((test_labels, train_labels))

    #free up memory, since these are huge arrays and are no longer needed
    del(train_inputs, train_labels, test_inputs, test_labels)

    input_shape=(24,24,3)
    label_shape=(8,8,3)
    #Make sure the expected shape is correct
    assert shuffled_train_inputs.shape[1:] == input_shape
    assert shuffled_train_labels.shape[1:] == label_shape
    assert shuffled_test_inputs.shape[1:] == input_shape
    assert shuffled_test_labels.shape[1:] == label_shape
    print("Data loaded successfully")

    print(f"Building model and training for {num_epochs} epochs")
    model = BlockCNN() #Create our model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.mse)
    model.build(input_shape=(None,24,24,3))
    model.fit(shuffled_train_inputs, shuffled_train_labels, epochs=num_epochs, batch_size=batch_size)
    print("Training complete")
    #Now, run the model on the test data
    loss = model.evaluate(shuffled_test_inputs, shuffled_test_labels)
    print(f"Model loss: {loss}")

    if save_weights:
        model.save(f"./model_{num_epochs}_epochs")

    if create_img_output:
        print("Creating Image Output... This may take a few seconds")
        #Create some human-viewable examples with the non-shuffled data
        image_output_shape = (768,512,3)
        pixels_per_image = image_output_shape[0]*image_output_shape[1]
        tile_size = label_shape[0]
        #Make sure that the image output size is divisible by the tile size
        assert image_output_shape[0] % tile_size == 0
        assert image_output_shape[1] % tile_size == 0
        num_tiles_per_image = pixels_per_image // (tile_size*tile_size)
        #Make sure that the number of total tiles is divisible by the number of tiles per image
        assert all_inputs.shape[0] % num_tiles_per_image == 0
        assert all_labels.shape[0] % num_tiles_per_image == 0
        images = np.split(all_inputs, all_inputs.shape[0]//num_tiles_per_image)
        if verbose_img_output:
            #Also create images of the input and label
            labels = np.split(all_labels, all_labels.shape[0]//num_tiles_per_image)
            for i in range(len(images)):
                pil_img = np_to_img(images[i], image_output_shape, crop_center=True)
                pil_label = np_to_img(labels[i], image_output_shape)
                pil_img.save(f"./output/{i}_input.png")
                pil_label.save(f"./output/{i}_label.png")
        #Create images of the output after running it through the model
        for i in range(len(images)):
            #To prevent memory overflow issues, split images[i] into batch_size chunks
            batched_img = np.array_split(images[i], images[i].shape[0]//batch_size)
            completed_batches = []
            for batch in batched_img:
                batch_prediction = model.predict_on_batch(batch)
                completed_batches.append(batch_prediction)
            model_prediction = np.concatenate(completed_batches)
            pil_prediction = np_to_img(model_prediction, image_output_shape)
            pil_prediction.save(f"./output/{i}_prediction.png")
    print("Done! 🥳🎉")

def np_to_img(np_array, output_shape, crop_center=False, step=8):
    #Converts a np array to PIL image
    #crop_center should be true when the np_array should be cropped to the tile size (used for the input images)
    if crop_center:
        num_tiles = (output_shape[0]*output_shape[1]) // (step*step)
        new_np_array = np.zeros(shape=(num_tiles,step,step,output_shape[-1]))
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
    #Load the data
    #input is 24x24, label is uncompressed center tile 8x8
    try:
        train_inputs = np.load(data_path+"train_inputs.npy")
        train_labels = np.load(data_path+"train_labels.npy")
        test_inputs = np.load(data_path+"test_inputs.npy")
        test_labels = np.load(data_path+"test_labels.npy")
    except FileNotFoundError as e:
        print("Error: Inputs/Labels could not be found")
        print("Ensure that you have succesfully run 'create_data.py'")
        raise(e)
    main(train_inputs, train_labels, test_inputs, test_labels, num_epochs, batch_size)