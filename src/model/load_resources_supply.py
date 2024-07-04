from skimage.transform import resize
import numpy as np
import imageio
from PIL import Image
import matplotlib.pyplot as plt

def get_resources_supplies(source, range_y, range_x):
    img_jpg = Image.open(source)
    epithelial_starting_fig = np.asarray(img_jpg)

    editable_starting_config = epithelial_starting_fig.copy()[range_y[0]:range_y[1], range_x[0]:range_x[1], 0]

    # plt.figure()
    # plt.imshow(editable_starting_config)
    # plt.show()

    # down_sized_image = block_reduce(editable_starting_config, block_size=(10,10), func=np.median)
    down_sized_image = resize(editable_starting_config, (100, 100), mode='constant')
    down_sized_image[down_sized_image >= 0.1] = 65355445254542
    down_sized_image[down_sized_image < 0.1] = False
    down_sized_image[down_sized_image == 65355445254542] = True

    return down_sized_image


def get_ducts(source, range_y, range_x):
    img_jpg = Image.open(source)
    epithelial_starting_fig = np.asarray(img_jpg)

    down_sized_image = resize(epithelial_starting_fig, (160, 200), mode='reflect')

    down_sized_image_cut = down_sized_image.copy()[range_y[0]:range_y[1], range_x[0]:range_x[1],0]

    down_sized_image_cut_contrast = down_sized_image_cut.copy()
    down_sized_image_cut_contrast[down_sized_image_cut_contrast < 0.038] = 0
    down_sized_image_cut_contrast[down_sized_image_cut_contrast >= 0.038] = 1
    imageio.imwrite('ducts_thesis.jpeg', down_sized_image_cut_contrast)


    # plt.figure()
    # plt.imshow(down_sized_image_cut_contrast)
    # plt.colorbar()
    # plt.show()
    return down_sized_image_cut_contrast

def get_ducts_epith_guided(source, range_y, range_x, threshold=0.7, down_size=True):
    img_jpg = Image.open(source)
    epithelial_starting_fig = np.asarray(img_jpg)

    if down_size:
        down_sized_image = resize(epithelial_starting_fig, (160, 200), mode='constant')
    else:
        down_sized_image = epithelial_starting_fig / 255

    # imageio.imwrite('down_sized_image.jpeg', down_sized_image)

    down_sized_image_cut = down_sized_image.copy()[range_y[0]:range_y[1], range_x[0]:range_x[1], 0]

    # imageio.imwrite('cropped.jpeg', down_sized_image_cut)

    # down_sized_image = block_reduce(editable_starting_config, block_size=(10,10), func=np.median)

    down_sized_image_cut_contrast = down_sized_image_cut.copy()
    down_sized_image_cut_contrast[down_sized_image_cut_contrast < threshold] = 0
    down_sized_image_cut_contrast[down_sized_image_cut_contrast >= threshold] = 1
    down_sized_image_cut_contrast = 1 - down_sized_image_cut_contrast

    # imageio.imwrite('ducts_thesis.jpeg', down_sized_image_cut_contrast)

    return down_sized_image_cut_contrast

# ducts = get_ducts('starting_configurations/ducts_lining.jpg', (60, 160), (0, 100))
# get_ducts('starting_configurations/ducts_lining_paint.jpg', (60, 160), (0, 100))

# coors = get_ducts_epith_guided('starting_configurations/epithelial_cells.jpg', (60, 160), (0, 100), threshold=0.262)
