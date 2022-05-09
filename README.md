# BlockCNN - Deblocking Compressed JPEG Images
## A final project for CSCI1470
Image compression is necessary for handling high-resolution images in the modern world, but when images are restricted to low-bandwidth or low-storage applications, high levels of compression must be used, leading to loss of information (lossy compression), but also visible loss in quality.

In the case of JPEG Images, the most popular form of lossy compression, images with high compression tend to appear blocky, and smooth curves will often become jagged. This is because JPEG compresses each 8x8-pixel region of an image separately, and borders between tiles of an image often appear jarring and unnatural.

**BlockCNN**, A network architecture detailed in [this paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w50/Maleki_BlockCNN_A_Deep_CVPR_2018_paper.pdf) and re-implemented using tensorflow in this repository, seeks to deblock and recover visual fidelity from highly-compressed JPEG images. (The network can also be used for image compression, but this is outside the scope of our current implementation)

## Example:
This is an image from the testing portion of the dataset, meaning the network was *not* trained using this image.

<img src="readme_images\Sailboat_Compressed.jpeg" alt="Compressed Image of a Sailboat" width="250"/>

Zooming in on fine details or contrast changes, it's clear that the output from the network is far superior to the blocky jpeg.

<img id="zoomed_sail_compressed" src="readme_images\Sailboat_Compressed_Sail.jpeg" alt="Zoomed-In image of the compressed sailboat's sail" width="400"/>

<label for="zoomed_sail_compressed">Compressed Image</label>


<img id="zoomed_sail_deblocked" src="readme_images\Sailboat_Deblocked_Sail.png" alt="Zoomed-In image of the deblocked sailboat's sail" width="400"/>

<label for="zoomed_sail_deblocked">Deblocked Image</label>

Deblocking is best seen in images with curves, such as this portion, containing a parrot's beak:

<img id="parrot_compressed" src="readme_images\Parrot_Compressed.png" alt="Zoomed-In image of the compressed parrot's beak" width="400"/>

<label for="parrot_compressed">Compressed Image</label>


<img id="parrot_deblocked" src="readme_images\Parrot_Deblocked.png" alt="Zoomed-In image of the deblocked parrot's beak" width="400"/>

<label for="parrot_deblocked">Deblocked Image</label>

# How to run:

1) First, create data by running `create_data.py` -- This processes images in the `Kodak_Full_Images` directory, and should create the `data` folder upon completion.
2) Next, run `main.py`, which will train the network, and output all images (training and testing) to a folder called `output`
3) Success! Observe that the images in `output` are much less blocky when compared to those in `data/compressed_images`, which are the inputs to the network. (alternatively, change `verbose_img_output` to `True` in main.py to get inputs/labels created in the `output` directory)