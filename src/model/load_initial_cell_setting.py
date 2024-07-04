from skimage.transform import resize
import matplotlib.pyplot as plt
import imageio
import numpy as np
from PIL import Image
import tifffile
import imagecodecs

def get_initial_coordinates(source, range_y, range_x, threshold=0.7):
    img_jpg = Image.open(source)
    epithelial_starting_fig = np.asarray(img_jpg)
    down_sized_image = resize(epithelial_starting_fig, (160, 200), mode='constant')

    # imageio.imwrite('down_sized_image.jpeg', down_sized_image)

    down_sized_image_cut = down_sized_image.copy()[range_y[0]:range_y[1], range_x[0]:range_x[1], 0]

    # imageio.imwrite('cropped.jpeg', down_sized_image_cut)

    # down_sized_image = block_reduce(editable_starting_config, block_size=(10,10), func=np.median)

    print(down_sized_image_cut)
    down_sized_image_cut_contrast = down_sized_image_cut.copy()
    down_sized_image_cut_contrast[down_sized_image_cut_contrast < threshold] = 0
    down_sized_image_cut_contrast[down_sized_image_cut_contrast >= threshold] = 255

    # imageio.imwrite('epithcellselection.jpeg', down_sized_image_cut_contrast)

    # fig, axes = plt.subplots(nrows=2, ncols=2)
    # print(axes)
    # axes[0, 0].imshow(down_sized_image)
    # axes[0, 1].imshow(down_sized_image_cut)
    # axes[1, 0].imshow(down_sized_image_cut_contrast)
    # plt.show()

    return tuple(zip(*np.nonzero(down_sized_image_cut_contrast == 0)))

def get_initial_coordinates_full_tumour(source, range_y, range_x, threshold=0.7):
    '''
    This function creates an epithelial cell landscape from the full tumour image (doesn't need resizing because
    pixel to content ratio is different that the zoomed image)
    :param source: path to tumour image
    :param range_y: pixels to select in y direction
    :param range_x: pixels to select in x direction
    :param threshold: intensity threshold of where to place epithelial cells
    :return:
    '''

    img_jpg = Image.open(source)
    epithelial_starting_fig = np.asarray(img_jpg)


    down_sized_image_cut = epithelial_starting_fig.copy()[range_y[0]:range_y[1], range_x[0]:range_x[1], 0]


    # down_sized_image = block_reduce(editable_starting_config, block_size=(10,10), func=np.median)

    #
    down_sized_image_cut_contrast = down_sized_image_cut.copy() / 255  # divide by 255 to scale to 0, 1
    down_sized_image_cut_contrast[down_sized_image_cut_contrast < threshold] = 0
    down_sized_image_cut_contrast[down_sized_image_cut_contrast >= threshold] = 255

    # imageio.imwrite('epithcellselection.jpeg', down_sized_image_cut_contrast)

    # fig, axes = plt.subplots(nrows=2, ncols=2)
    # print(axes)
    # axes[0, 0].imshow(epithelial_starting_fig)
    # axes[0, 1].imshow(down_sized_image_cut)
    # axes[1, 0].imshow(down_sized_image_cut_contrast)
    # plt.show()

    return tuple(zip(*np.nonzero(down_sized_image_cut_contrast == 0)))

def test_read_tiff(source):
    Image.MAX_IMAGE_PIXELS = None

    ds = tifffile.imread(source)
    epithelial_starting_fig = np.asarray(ds, dtype='uint8')

    tifffile.imsave('test2.jpg', epithelial_starting_fig, compress=True)
    # img_jpg = Image.open(source)
    # img_ = plt.imread(source)
    # print(ds)
    # img_jpg.show()

    # print(epithelial_starting_fig)
    #
    # # editable_starting_config = epithelial_starting_fig.copy()
    #
    plt.figure()
    plt.imshow(epithelial_starting_fig)
    plt.show()

# test_read_tiff('starting_configurations/R1_1-1_1_HE_ojb_HE_20201109_155645.tiff')


# coors = get_initial_coordinates('starting_configurations/epithelial_cells.jpg', (60, 160), (0, 100), threshold=0.262)
# tumour_image_path = 'starting_configurations/tumour_cms4-2.jpg'
# tumour_image_path = 'starting_configurations/epithelial_cells.jpg'
#
# y_range, x_range = (450, 550), (100, 200)
#
# if tumour_image_path == 'starting_configurations/epithelial_cells.jpg':
#     coors = get_initial_coordinates(tumour_image_path, (60, 160), (0, 100))
# elif tumour_image_path == 'starting_configurations/tumour_cms4-2.jpg':
#     coors = get_initial_coordinates_full_tumour(tumour_image_path, y_range, x_range)