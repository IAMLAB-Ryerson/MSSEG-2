# -*- coding: utf-8 -*-
"""
@author(s): Adam Gibicar, Samir Mitha
"""
import numpy as np


def binarize(img, T):

    '''
    Description
    -----------
    Binarize an image or volume using a user-defined threshold.


    Example
    -------
    >> img = nib.load('pred_unet.nii').get_fdata()
    >> pred = binarize(img, 0.5)


    Parameters
    ----------
    >> img(npy): 2-D image or 3-D volume
    >> T(float): threshold for binarization


    Returns
    -------
    >> img(npy): binarized image or mask
    '''

    img[img > T] = 1
    img[img <= T] = 0

    return img


def normalize_vol(vol, mask=None, normType='min-max'):

    '''
    Description
    -----------
    Performs intensity normalization of an image or volume. If a mask
    is supplied, then the image is first multiplied by the mask before
    normalization.


    Example
    -------
    >> vol = nib.load('vol_01.nii').get_fdata()
    >> vol = normalize_vol(vol, normType='gaussian')


    Parameters
    ----------
    >> vol(float): 2-D image or 3-D volume
    >> mask(int): 2-D or 3-D mask
    >> normType(str): normalization method (default: min-max scaling)


    Returns
    -------
    >> vol(float): normalized image or volume
    '''

    # Cast to float
    vol = vol.astype(np.float32)

    # Normalize to [0, 1]
    if(normType == 'min-max'):
        if(mask is not None):
            I_max = np.max(vol[mask != 0])
            I_min = np.min(vol[mask != 0])
        else:
            I_max = np.max(vol)
            I_min = np.min(vol)

        vol = (vol - I_min) / (I_max - I_min)
        vol = np.asarray(vol)
        alpha = 0
    # Gaussian normalization
    elif(normType == 'gaussian'):

        if(mask is not None):
            mu = np.mean(vol[mask != 0])
            sigma = np.std(vol[mask != 0])
        else:
            mu = np.mean(vol.ravel())
            sigma = np.std(vol.ravel())

        vol -= mu

        try:
            vol /= sigma
        except ZeroDivisionError:
            vol /= 1
        alpha = 0
    # Median normalization
    elif(normType == 'median'):

        if(mask is not None):
            mu = np.median(vol[mask != 0])
            q75, q25 = np.percentile(vol[mask != 0], [75, 25])
            iqr = q75 - q25

        else:
            mu = np.median(vol.ravel())
            q75, q25 = np.percentile(vol.ravel(), [75, 25])
            iqr = q75 - q25

        vol -= mu

        try:
            vol /= iqr
        except ZeroDivisionError:
            vol /= 1

    # 3-sigma normalization
    elif(normType == '3-sigma'):
        if(mask is not None):
            mu = np.mean(vol[mask != 0])
            sigma = np.std(vol[mask != 0])
        else:
            mu = np.mean(vol)
            sigma = np.std(vol)

        try:
            vol /= (mu + 3*sigma)
        except ZeroDivisionError:
            vol /= 1

    # elif(normType == 'z-score')


    # v2 normalization
    if(normType == 'v2'):
        nbins = int(np.max(vol) - np.min(vol) + 1)
        bin_range = [np.min(vol), np.max(vol)]
        bins, freq = plot_volHist(vol, nbins=nbins, bin_range=bin_range, plt_hist=False)
        brain_pk = bins[freq == np.max(freq)]

        if(len(brain_pk) > 1):
            brain_pk = brain_pk[-1]

        alpha = 280.0 / brain_pk
        vol *= alpha

    return vol, alpha


def get_pred_4ch_agg_patches(time1, time2, diffMap1, diffMap2, model1, model2, model3, model4, model5, img_size=(256, 256), patch_size=(64, 64), stride=(1, 1), overlap=0.5, plane = 'ax'):

    '''
    Function that takes a volume and makes predictions on each patch of
    each slice. Each pixel is classified as a 1 if it's a WML and a 0
    otherwise.
    '''

    test_pred1 = np.zeros([diffMap1.shape[0], diffMap1.shape[1], diffMap1.shape[2]])
    test_pred2 = np.zeros([diffMap1.shape[0], diffMap1.shape[1], diffMap1.shape[2]])
    test_pred3 = np.zeros([diffMap1.shape[0], diffMap1.shape[1], diffMap1.shape[2]])
    test_pred4 = np.zeros([diffMap1.shape[0], diffMap1.shape[1], diffMap1.shape[2]])
    test_pred5 = np.zeros([diffMap1.shape[0], diffMap1.shape[1], diffMap1.shape[2]])
    # print(test_pred.shape)
    # Get each 2-D slice
    if plane == 'ax':
        for i in range(time1.shape[2]):
            img_1 = time1[:, :, i]
            img_2 = time2[:, :, i]
            diff_1 = diffMap1[:, :, i]
            diff_2 = diffMap2[:, :, i]
            # img_size = (img.shape[0:1])
            time1_patches, time2_patches, diff_patches1, diff_patches2 = extract_input_patches(img_1, img_2, diff_1, diff_2, patch_size=patch_size, overlap=overlap)
            test_patches1 = np.zeros([*time1_patches.shape])

           # Extract patches
            for j in range(time1_patches.shape[2]):

                for k in range(time1_patches.shape[3]):
                    time1_patch = time1_patches[:, :, j, k]
                    time2_patch = time2_patches[:, :, j, k]
                    diff_patch_1 = diff_patches1[:, :, j, k]
                    diff_patch_2 = diff_patches2[:, :, j, k]
                    patch = np.zeros([*time1_patch.shape])
                    patch[:, :, 0] = time1_patch
                    patch[:, :, 1] = time2_patch
                    patch[:, :, 2] = diff_patch_1
                    patch[:, :, 3] = diff_patch_2
                    # patch = patch.reshape(*patch_size, 1)
                    # patch = np.expand_dims(patch, axis=0)
                    pred = model.predict(patch)
                    test_patches[:, :, j, k] = pred.reshape(*patch_size)

            test_pred[:, :, i] = get_img_from_patches(test_patches, img_size=(test_pred.shape[0], test_pred.shape[1]), overlap=overlap)
    elif plane == 'sag':
        for i in range(time1.shape[0]):
            img_1 = time1[i, :, :]
            img_2 = time2[i, :, :]
            diff_1 = diffMap1[i, :, :]
            diff_2 = diffMap2[i, :, :]
            # img_size = (img.shape[0:1])
            time1_patches, time2_patches, diff_patches1, diff_patches2 = extract_input_patches(img_1, img_2, diff_1, diff_2, patch_size=patch_size, overlap=overlap)
            test_patches1 = np.zeros([*time1_patches.shape])
            test_patches2 = np.zeros([*time1_patches.shape])
            test_patches3 = np.zeros([*time1_patches.shape])
            test_patches4 = np.zeros([*time1_patches.shape])
            test_patches5 = np.zeros([*time1_patches.shape])


            # Extract patches
            for j in range(time1_patches.shape[2]):

                for k in range(time1_patches.shape[3]):
                    time1_patch = time1_patches[:, :, j, k]
                    time2_patch = time2_patches[:, :, j, k]
                    diff_patch_1 = diff_patches1[:, :, j, k]
                    diff_patch_2 = diff_patches2[:, :, j, k]
                    patch = np.zeros([*time1_patch.shape, 4])
                    patch[:, :, 0] = time1_patch
                    patch[:, :, 1] = time2_patch
                    patch[:, :, 2] = diff_patch_1
                    patch[:, :, 3] = diff_patch_2
                    patch = patch.reshape(*patch_size, 4)
                    patch = np.expand_dims(patch, axis=0)
                    pred1 = model1.predict(patch)
                    pred1 = pred1.reshape(*patch_size)
                    test_patches1[:, :, j, k] = pred1

                    pred2 = model2.predict(patch)
                    pred2 = pred2.reshape(*patch_size)
                    test_patches2[:, :, j, k] = pred2

                    pred3 = model3.predict(patch)
                    pred3 = pred3.reshape(*patch_size)
                    test_patches3[:, :, j, k] = pred3

                    pred4 = model4.predict(patch)
                    pred4 = pred4.reshape(*patch_size)
                    test_patches4[:, :, j, k] = pred4

                    pred5 = model5.predict(patch)
                    pred5 = pred5.reshape(*patch_size)
                    test_patches5[:, :, j, k] = pred5

            test_pred1[i, :, :] = get_img_from_patches(test_patches1, img_size=(test_pred1.shape[1], test_pred1.shape[2]),
                                                      overlap=overlap)
            test_pred2[i, :, :] = get_img_from_patches(test_patches2,
                                                       img_size=(test_pred2.shape[1], test_pred2.shape[2]),
                                                       overlap=overlap)
            test_pred3[i, :, :] = get_img_from_patches(test_patches3,
                                                       img_size=(test_pred3.shape[1], test_pred3.shape[2]),
                                                       overlap=overlap)
            test_pred4[i, :, :] = get_img_from_patches(test_patches4,
                                                       img_size=(test_pred4.shape[1], test_pred4.shape[2]),
                                                       overlap=overlap)
            test_pred5[i, :, :] = get_img_from_patches(test_patches5,
                                                       img_size=(test_pred5.shape[1], test_pred5.shape[2]),
                                                       overlap=overlap)
    elif plane == 'cor':
        for i in range(time1.shape[2]):
            img_1 = time1[:, i, :]
            img_2 = time2[:, i, :]
            diff = diffMap[:, i, :]
            # img_size = (img.shape[0:1])
            time1_patches = extract_patches(img_1, patch_size=patch_size, overlap=overlap)
            time2_patches = extract_patches(img_2, patch_size=patch_size, overlap=overlap)
            diff_patches = extract_patches(diff, patch_size=patch_size, overlap=overlap)
            test_patches = np.zeros([*time1_patches.shape])

            # Extract patches
            for j in range(time1_patches.shape[2]):

                for k in range(time1_patches.shape[3]):
                    time1_patch = time1_patches[:, :, j, k]
                    time2_patch = time2_patches[:, :, j, k]
                    diff_patch = diff_patches[:, :, j, k]
                    patch = np.zeros([*time1_patch.shape])
                    patch[:, :, 0] = time1_patch
                    patch[:, :, 1] = time2_patch
                    patch[:, :, 2] = diff_patch
                    # patch = patch.reshape(*patch_size, 1)
                    # patch = np.expand_dims(patch, axis=0)
                    pred = model.predict(patch)
                    test_patches[:, :, j, k] = pred.reshape(*patch_size)

            test_pred[:, i, :] = get_img_from_patches(test_patches, img_size=(test_pred.shape[0], test_pred.shape[1]),
                                                      overlap=overlap)

    return test_pred1, test_pred2, test_pred3, test_pred4, test_pred5


def extract_patches(img, patch_size=(64, 64), stride=(1, 1), overlap=0.5):

    '''
    Description
    -----------
    Extract 2-D patches from a grayscale image and corresponding ground
    truth. The user can specify the degree of overlap (e.g. 0%, 50%).


    Example
    -------
    >> img = imread('img.png')
    >> patch_size = (32, 32)
    >> patches = extract_patches(img, patch_size, overlap=0.5)


    Parameters
    ----------
    >> img(float): 2-D grayscale image
    >> label(int): 2-D ground truth
    >> patch_size(tuple): size of extracted patches
    >> stride(int): number of pixels to move in the sliding window
    >> overlap(float): amount of overlap between extracted patches


    Returns
    -------
    >> patches(float): 4-D numpy array containing all the patches
    >> patches_labels(int): 4-D numpy array containing ground truth patches
    '''

    img_size = img.shape
    I_min = np.min(img)
    
    if(overlap is not None):
        stride_x = int((1 - overlap) * patch_size[0])
        stride_y = int((1 - overlap) * patch_size[1])
    else:
        stride_x = stride[0]
        stride_y = stride[1]

    m = len(range(0, img_size[0], stride_x))
    n = len(range(0, img_size[1], stride_y))

    patches = np.zeros((*patch_size, m, n))

    count_x = 0
    for i in range(0, img_size[0], stride_x):

        if(i + patch_size[0] > img_size[0]):
            img = np.pad(img, ((0, i + patch_size[0] - img_size[0]), (0, 0)), 'constant', constant_values=I_min)

        count_y = 0
        for j in range(0, img_size[1], stride_y):

            if(j + patch_size[1] > img_size[1]):
                img = np.pad(img, ((0, 0), (0, j + patch_size[1] - img_size[1])), 'constant', constant_values=I_min)

            patches[:, :, count_x, count_y] = img[i:i + patch_size[0], j:j + patch_size[1]]
            count_y += 1

        count_x += 1

    return patches


def get_img_from_patches(patches, img_size=(256, 256), stride=(1, 1), overlap=0.5):

    '''
    Description
    -----------
    Reconstructs 2-D images from a 4-D numpy array of patches.


    Example
    -------
    >> img = get_img_from_patches(patches, img_size=(224, 224))


    Parameters
    ----------
    >> patches(npy): 4-D array containing extracted patches
    >> img_size(tuple): target image dimensions
    >> stride(tuple): determines the stride in the x- and y-directions
    >> overlap(float): overlap of the patches specified between 0 - 1


    Returns
    -------
    >> img(float): reconstructed image
    '''

    # Initializations
    img = np.zeros(img_size)
    img_idx = np.zeros(img_size)
    patch_size = patches.shape[0:2]

    # Determine stride
    if(overlap is not None):
        stride_x = int((1 - overlap) * patch_size[0])
        stride_y = int((1 - overlap) * patch_size[1])
    else:
        stride_x = stride[0]
        stride_y = stride[1]

    # Determine number of patches
    m = patches.shape[2]
    n = patches.shape[3]

    count_x = 0
    for i in range(m):
        idx_row = [x for x in range(count_x, patch_size[0] + count_x)]

        if(patch_size[0] + count_x > img_size[0]):
            idx_row = [x for x in range(idx_row[0], img_size[0])]

        count_y = 0
        for j in range(n):
            idx_col = [y for y in range(count_y, patch_size[1] + count_y)]

            if(patch_size[1] + count_y > img_size[1]):
                idx_col = [y for y in range(idx_col[0], img_size[1])]

            img[np.ix_(idx_row, idx_col)] = img[np.ix_(idx_row, idx_col)] + patches[0:len(idx_row), 0:len(idx_col), i, j]
            img_idx[np.ix_(idx_row, idx_col)] += 1
            count_y += stride_y

        count_x += stride_x

    # Compute average pixel value
    img = np.divide(img, img_idx)

    return img


def extract_input_patches(time1, time2, diff1, diff2, patch_size=(64, 64), stride=(1, 1), overlap=0.5):
    '''
    Description
    -----------
    Extract 2-D patches from a grayscale image and corresponding ground
    truth. The user can specify the degree of overlap (e.g. 0%, 50%).


    Example
    -------
    >> img = imread('img.png')
    >> patch_size = (32, 32)
    >> patches = extract_patches(img, patch_size, overlap=0.5)


    Parameters
    ----------
    >> img(float): 2-D grayscale image
    >> label(int): 2-D ground truth
    >> patch_size(tuple): size of extracted patches
    >> stride(int): number of pixels to move in the sliding window
    >> overlap(float): amount of overlap between extracted patches


    Returns
    -------
    >> patches(float): 4-D numpy array containing all the patches
    >> patches_labels(int): 4-D numpy array containing ground truth patches
    '''

    img_size = time1.shape
    I_min = 0

    if (overlap is not None):
        stride_x = int((1 - overlap) * patch_size[0])
        stride_y = int((1 - overlap) * patch_size[1])
    else:
        stride_x = stride[0]
        stride_y = stride[1]

    m = len(range(0, img_size[0], stride_x))
    n = len(range(0, img_size[1], stride_y))

    time1_patches = np.zeros((*patch_size, m, n))
    time2_patches = np.zeros((*patch_size, m, n))
    diff1_patches = np.zeros((*patch_size, m, n))
    diff2_patches = np.zeros((*patch_size, m, n))
    count_x = 0
    for i in range(0, img_size[0], stride_x):

        if (i + patch_size[0] > img_size[0]):
            time1 = np.pad(time1, ((0, i + patch_size[0] - img_size[0]), (0, 0)), 'constant', constant_values=I_min)
            time2 = np.pad(time2, ((0, i + patch_size[0] - img_size[0]), (0, 0)), 'constant', constant_values=I_min)
            diff1 = np.pad(diff1, ((0, i + patch_size[0] - img_size[0]), (0, 0)), 'constant', constant_values=I_min)
            diff2 = np.pad(diff2, ((0, i + patch_size[0] - img_size[0]), (0, 0)), 'constant', constant_values=I_min)

        count_y = 0
        for j in range(0, img_size[1], stride_y):

            if (j + patch_size[1] > img_size[1]):
                time1 = np.pad(time1, ((0, 0), (0, j + patch_size[1] - img_size[1])), 'constant', constant_values=I_min)
                time2 = np.pad(time2, ((0, 0), (0, j + patch_size[1] - img_size[1])), 'constant', constant_values=I_min)
                diff1 = np.pad(diff1, ((0, 0), (0, j + patch_size[1] - img_size[1])), 'constant', constant_values=I_min)
                diff2 = np.pad(diff2, ((0, 0), (0, j + patch_size[1] - img_size[1])), 'constant', constant_values=I_min)

            time1_patches[:, :, count_x, count_y] = time1[i:i + patch_size[0], j:j + patch_size[1]]
            time2_patches[:, :, count_x, count_y] = time2[i:i + patch_size[0], j:j + patch_size[1]]
            diff1_patches[:, :, count_x, count_y] = diff1[i:i + patch_size[0], j:j + patch_size[1]]
            diff2_patches[:, :, count_x, count_y] = diff2[i:i + patch_size[0], j:j + patch_size[1]]
            count_y += 1

        count_x += 1

    return time1_patches, time2_patches, diff1_patches, diff2_patches