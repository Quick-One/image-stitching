from math import floor, ceil
import numpy as np
import cv2


def compute_homography(pt1, pt2):
    '''
    Computes the homography matrix that maps points from pt2 to pt1.
    pt1 : numpy.ndarray of shape (n, 2) (xy format)
    pt2 : numpy.ndarray of shape (n, 2) (xy format)
    returns : homography matrix of shape (3, 3)
    '''
    assert pt1.shape == pt2.shape

    A = []
    B = []
    for i in range(pt1.shape[0]):
        x_, y_ = pt1[i]
        x, y = pt2[i]

        a = np.array([
            [x, y, 1, 0, 0, 0, -x*x_, -y*x_],
            [0, 0, 0, x, y, 1, -x*y_, -y*y_]
        ])
        b = np.array([x_, y_])
        A.append(a)
        B.append(b)

    A = np.vstack(A)
    B = np.hstack(B)

    H = np.linalg.lstsq(A, B, rcond=None)[0]
    H = np.append(H, 1)
    H.resize((3, 3))

    return H


def get_bounding_box(image, H):
    '''
    Computes the bounding box of the warped image.
    image : unwarped image (the unaltered image not on canvas)
    H : homography matrix
    returns : bounding box of the warped image (xmin, xmax, ymin, ymax)
    '''
    corners = np.array([
        [0, 0, 1],
        [image.shape[1], 0, 1],
        [0, image.shape[0], 1],
        [image.shape[1], image.shape[0], 1],
    ]).T

    corners = H @ corners
    corners = corners / corners[2]  # normalize

    min_x = floor(np.min(corners[0]))
    max_x = ceil(np.max(corners[0]))
    min_y = floor(np.min(corners[1]))
    max_y = ceil(np.max(corners[1]))

    return (min_x, max_x, min_y, max_y)


def get_translation_matrix(deltax, deltay):
    return np.array([
        [1, 0, deltax],
        [0, 1, deltay],
        [0, 0, 1]
    ], dtype=np.float64)


def warp_images(image_homographies_pairs):
    '''
    Warp all the images to a common canvas.
    image_homographies_pairs : list of tuples (image, H)
    H is with respect to the origin of the first image
    returns : list of warped images on the common canvas
    '''
    X, Y = float('inf'), float('inf')
    Xprime, Yprime = float('-inf'), float('-inf')
    for image, H in image_homographies_pairs:
        min_x, max_x, min_y, max_y = get_bounding_box(image, H)
        X = min(X, min_x)
        Y = min(Y, min_y)
        Xprime = max(Xprime, max_x)
        Yprime = max(Yprime, max_y)

    canvas_size = (Xprime - X, Yprime - Y)
    images = []
    for image, H in image_homographies_pairs:
        deltax = -X
        deltay = -Y
        translation_matrix = get_translation_matrix(deltax, deltay)
        warped = cv2.warpPerspective(image, translation_matrix @ H, canvas_size)
        images.append(warped)
    return images


def blend_image(im1, im2, method = 'mean'):
    '''
    return a blended image of im1 and im2
    im1 and im2 are both belong to the same canvas
    returns : blended image
    
    mean: takes a mean of the common of the two images
    linear: blends along the X axis linearly
    '''
    if method == 'mean':
        im1_mask = np.any(im1 != [0, 0, 0], axis=2)
        im2_mask = np.any(im2 != [0, 0, 0], axis=2)
        common = np.logical_and(im1_mask, im2_mask)
        
        common_image = cv2.addWeighted(im1, 0.5, im2, 0.5, 0)
        final_image = im1 + im2
        final_image[common] = common_image[common]
        return final_image

    elif method == 'linear':
        im1_mask = np.any(im1 != [0, 0, 0], axis=2)
        im2_mask = np.any(im2 != [0, 0, 0], axis=2)
        common = np.logical_and(im1_mask, im2_mask)
        
        # leftmost in the common map
        leftmost = np.argmax(common, axis=1)
        leftmost = min(leftmost[np.max(common, axis=1).astype(bool)]) # remove those rows in which there is no common pixel

        rightmost = (np.argmax(common[:, ::-1], axis=1))
        rightmost = min(rightmost[np.max(common, axis=1).astype(bool)]) # remove those rows in which there is no common pixel
        rightmost = im1.shape[1] - rightmost - 1

        gradient = np.arange(im1.shape[1], dtype=np.float32)
        gradient = (gradient - leftmost) / (rightmost - leftmost)
        gradient = np.clip(gradient, 0, 1)

        mask_gradient_right = common * gradient
        mask_gradient_left = common - mask_gradient_right

        # make the mask gradient R X C to R X C X 3
        mask_gradient_right = np.repeat(mask_gradient_right[:, :, np.newaxis], 3, axis=2)
        mask_gradient_left = np.repeat(mask_gradient_left[:, :, np.newaxis], 3, axis=2)
        
        final_image = im1 + im2
        final_image[common] = (im1 * mask_gradient_left + im2 * mask_gradient_right)[common]
        return final_image
    
    else: raise NotImplementedError
    
