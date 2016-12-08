# Tips from AlexNet paper at
# https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

# TODO: Downsample training images to 256x256
#
#       Given a rectangular image, we first rescaled the image such that the shorter side was of length 256, and then
#       cropped out the central 256×256 patch from the resulting image. We did not pre-process the images
#       in any other way, except for subtracting the mean activity over the training set from each pixel. So
#       we trained our network on the (centered) raw RGB values of the pixels.

