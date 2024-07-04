import cv2
import numpy as np
from fipy import dump
import time
from PIL import Image
from skimage.color import deltaE_ciede2000
from scipy import sparse
import random
start = time.time()

def eraser(imagex, parentdirectory, filename, rank):
    filepth = parentdirectory + "/" + filename
    imorg = np.copy(imagex)
    # imlab = cv2.cvtColor(imorg, cv2.COLOR_BGR2Lab)
    # cv2.imwrite(parentdirectory + "/blood_vessels/lab_" + filename, imlab)
    # templab = deltaE_ciede2000(imlab, np.asarray([[0, 128, 128]]).astype(np.float32)) #remove black
    # imorg[templab < 30] = 0

    imlab = cv2.cvtColor(imorg, cv2.COLOR_BGR2Lab)
    simlab = sparse.csr_matrix(imlab[:, :, 0])
    imzero = imorg[:, :, 0] + imorg[:, :, 1] + imorg[:, :, 2]
    imnonzero = np.where(imzero != 0)
    imlab_copy = np.zeros_like(imlab)
    colos = np.unique(imorg[imnonzero].reshape(-1, imorg.shape[2]), axis=0, return_counts=True) # find all possible unqiue colors
    dump.write(colos, parentdirectory + "/blood_vessels/" + filename + ".dat") # write and read for debug
    colos = dump.read(parentdirectory + "/blood_vessels/" + filename + ".dat")
    coloslab = cv2.cvtColor(np.asarray([colos[0]]), cv2.COLOR_BGR2Lab)
    max_colors = 8000
    supacolosind = np.arange(max_colors, dtype=int)
    supacolosrand = np.asarray([random.randint(int(max_colors * 0.1), len(colos[1])) for _ in range(int(
        max_colors*0.9))])
    supacolosind[int(max_colors * 0.1):] = supacolosrand
    random.shuffle(supacolosind)
    for kleurcutoff in [98, 95, 90]:
        for kL in [1.0]:
            for kC in [1.0]:
                for kH in [0.1]:
                    print(kleurcutoff, kL, kC, kH)
                    coloslab_dest = np.zeros_like(coloslab)
                    supaclose = np.zeros((supacolosind.shape[0], supacolosind.shape[0]), dtype=float)
                    supacolos = coloslab[0][supacolosind].astype(np.float32)
                    colodeciders = np.ones_like(supaclose, dtype=int) * 10000
                    for kolox in range(len(supacolosind)): #loop to find similar colors within the unique
                        colodel = deltaE_ciede2000(supacolos, supacolos[kolox], kH=kH)
                        supaclose[kolox] = np.copy(colodel)
                        colodeciders[kolox][np.where(colodel < kleurcutoff)] = kolox
                    uniqcols = np.min(colodeciders, axis=0)
                    truniq = np.unique(uniqcols)
                    # truniq = np.asarray([0, 1, 2])
                    colorpallete = np.zeros((30, len(truniq) * 50, 3), dtype=np.uint8)
                    target_arr = imlab.astype(np.float32)[imnonzero]
                    save_arr = np.zeros((len(truniq), target_arr.shape[0]))
                    for colcount in range(len(truniq)): #loop to find similar colors througout image
                        colorpallete[0:30, colcount * 50:(colcount + 1) * 50] = colos[0][supacolosind[truniq[colcount]]]
                        # print (kleurcutoff, supacolosind[truniq[colcount]])
                        templab = deltaE_ciede2000(target_arr,
                                                   coloslab[0][supacolosind[truniq[colcount]]].astype(np.float32),
                                                   kL=kL, kC=kC, kH=kH)
                        save_arr[colcount] = np.copy(templab)
                    pos_chns = np.argmin(save_arr, axis=0) #find the closest colors at a pixel
                    target_arr = cv2.cvtColor(np.asarray([coloslab[0][supacolosind[truniq[np.argmin(save_arr, axis=0)]]]]), cv2.COLOR_Lab2BGR)
                    imlab_copy[imnonzero] = target_arr[0]
                    target_sum = target_arr[0][:, 0] + 256 * target_arr[0][:, 1] + 256 * 256 * target_arr[0][:, 2]
                    chncount=0
                    for tarsum in np.unique(target_sum):
                        chnarr = np.zeros_like(imlab_copy)
                        kbinary = target_sum==tarsum
                        target_chn = target_arr[0] * np.stack((kbinary,) * 3, axis=-1)
                        chnarr[imnonzero] = target_chn
                        cv2.imwrite(parentdirectory + "/blood_vessels/" + str(kleurcutoff) + str(kL) + str(kC) + str(kH)
                                    + "_"
                                    + str(chncount) + "_" + filename, chnarr)
                        chncount+=1
                    cv2.imwrite(parentdirectory + "/blood_vessels/" + str(kleurcutoff) + str(kL) + str(kC) + str(
                        kH) + "_" + filename, imlab_copy)
                    cv2.imwrite(parentdirectory + "/blood_vessels/" + filename + str(kleurcutoff) + ".tif",
                                colorpallete)


source = 'bloodvessels.jpg'
img_jpg = Image.open(source)
imagex = np.asarray(img_jpg)
parentdirectory = 'starting_configurations'
filename = 'im_seg.jpg'
rank = 'dummy'
# luminocity, chroma, hue (values in list are weights)

eraser(imagex, parentdirectory, filename, rank)