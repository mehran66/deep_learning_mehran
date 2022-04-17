__version__ = '0.1.0'

'''
#############ISUES/BUGS#############
- In the load data, when reading raw images as inputs, the images are all zero in the generated dataset.
This issue might happen due to the input images (planet data). I need to first test another dataset.
def process_path(image, label):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)


#############IMPROVEMENTS#############
- some codes can be converted to a function in the utils.py so they do not get repeated such as plot batch images in the train and fine-tune


#############Resolved#############


'''