# Art_Generator
![readme_image.jpg](readme_image.jpg)
### Let's get your photos repaint by famouse artists
This notebook takes a content image and a style image and creates an output image which keeps the major characteristics of the content image and adds style features from the style image to it. A pretrained neural net called VGG-19 (`imagenet-vgg-verydeep-19.mat`) is used for modelling which has been published by the Visual Geometry Group at University of Oxford in 2014.

* How to use:
`content_image = imageio.imread("your_picture.jpg")`
`style_image = imageio.imread("image_from_artist.jpg")`


