import numpy as np
from matplotlib import pyplot as plt
""" This script implements the functions for data augmentation and preprocessing.
"""
def data_augmentation(image, type):

    if type == "flipY":
        image = np.fliplr(image)
    if type == "flipX":
        image = np.flipud(image)
    elif type == "shearX":
        direction = 1
        strength = 5
        res = np.empty_like(image)
        for i in range(0, 32):
            shift = int((i/32)* strength)
            res[i,:,:] = np.roll(image[i,:,:], direction*shift, 0)
        image = res
    elif type == "shearY":
        direction = 1
        strength = 5
        res = np.empty_like(image)
        for i in range(0, 32):
            shift = int((i / 32) * strength)
            res[:, i, :] = np.roll(image[:, i, :], direction * shift, 0)
        image = res
    elif type == "invert":
        image = 255-image
    elif type == "pad":
        image = np.pad(image, ((4, 4), (4, 4), (0, 0)))
        random_upper_left_x = np.random.randint(9)
        random_upper_left_y = np.random.randint(9)
        image = image[random_upper_left_x:random_upper_left_x + 32, random_upper_left_y:random_upper_left_y + 32]
    elif type == "color reduction":
        image = (image // 64) * 64
    elif type == "contrast":
        delta = np.random.randint(60)
        for x in range(32):
            for y in range(32):
                if np.mean(image[x, y, :3]) > 128:
                    (r, g, b) = image[x, y, :3]
                    image[x, y, :3] = (min(r + delta,255), min(g + delta,255), min(b + delta,255))
                else:
                    (r, g, b) = image[x, y, :3]
                    image[x, y, :3] = (max(0,r - delta), max(0,g - delta), max(0,b - delta))
    else:
        image = image
    return image

def parse_record(record, training, type = "None"):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training, type)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image

def preprocess_image(image, training, type = "None"):
    """ Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    if training:
        types = ["shearX", "shearY", "flipX", "pad", "flipY", "rotate", "None"]
        rand_type = np.random.randint(7)
        image = data_augmentation(image, types[rand_type])
    else:
        image = data_augmentation(image, type)

    ### YOUR CODE HERE
    # Subtract off the mean and divide by the standard deviation of the pixels.
    std = np.std(image)
    mean = np.mean(image)
    image = (image - mean)/std

    ### YOUR CODE HERE

    return image

def visualize(image, save_name='test.png'):
    """Visualize a single test image.
    
    Args:
        image: An array of shape [3072]
        save_name: An file name to save your visualization.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    image = image.reshape((3, 32, 32))
    image = np.transpose(image, [1, 2, 0])
    ### YOUR CODE HERE

    plt.imshow(image)
    plt.savefig(save_name)
    return image