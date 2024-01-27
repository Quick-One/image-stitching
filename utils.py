from math import floor, ceil
import numpy as np
import cv2
from skimage.feature import corner_harris, peak_local_max
from scipy.spatial.distance import cdist
import heapq


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
    

def merge_2images(im1, im2, H, method = 'mean'):
    im1, im2 = warp_images([(im1, np.eye(3)), (im2, H)])
    return blend_image(im1, im2, method)


def harris_points(im, min_dist = 10, threshold_rel = 0.001, exclude_border = 30):
    '''
    returns the harris points in the image (yx format)
    '''
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h = corner_harris(im_gray)
    hpoints = peak_local_max(h, min_distance=min_dist, 
                             threshold_rel=threshold_rel, 
                             exclude_border=exclude_border)
    return hpoints, h


def ANMS(hpoints, h, n = 1000, c_robust = 0.9):
    '''
    Adaptive Non-Maximal Suppression on haris points
    hpoints : numpy.ndarray of shape (n, 2) (yx format)
    h : haris response of (image.shape)
    returns : numpy.ndarray of shape (m, 2) (yx format)
    '''
    assert hpoints.shape[0] < 7500  # otherwise cdist will run out of memory
    dist = cdist(hpoints, hpoints)
    strength = h[hpoints[:,0], hpoints[:,1]]
    
    ver = np.tile(strength, (strength.shape[0], 1))
    hor = ver.T
    mask = (ver < c_robust*hor)
    dist[mask] = 1e9
    
    minimum_distance = np.min(dist, axis=1)
    indices_greatest = np.argsort(minimum_distance)[::-1][:n]
    return hpoints[indices_greatest]


def get_descriptors(im, pts, patch_size = 40):
    '''
    takes list of points (n x 2) (yx format)
    returns descriptors of shape (n x 64*3)
    '''
    patches = []
    off = patch_size // 2
    for pt in pts:
        y, x = pt
        patch = im[y-off:y+off, x-off:x+off]
        patch = (patch - np.mean(patch)) / np.std(patch)
        patch = cv2.resize(patch, (8, 8))
        patches.append(patch.flatten())
    return np.vstack(patches)


def match_descriptors(desc1, desc2, threshold = 0.5):
    '''
    desc1 : descriptors of shape (n1 x 64*3)
    desc2 : descriptors of shape (n2 x 64*3)
    returns : matches of shape (m x 2) (index of desc1, index of desc2)
    '''
    dist = cdist(desc1, desc2, metric='sqeuclidean')
    smallest_ind = np.argmin(dist, axis=1)
    smallest_val = np.min(dist, axis=1)
    dist[:, smallest_ind] = 1e9
    second_smallest_val = np.min(dist, axis=1)
    
    mask = (smallest_val / second_smallest_val) < threshold
    matches = np.vstack((np.where(mask)[0], smallest_ind[mask]))
    unique_ind = np.unique(matches[1], return_index=True)[1]
    matches = matches[:, unique_ind]
    return matches.T


def homogenize(pts):
    '''
    takes points of shape (n x 2) (yx format)
    returns points of shape (n x 3) (xy1 format)
    '''
    return np.hstack((yx_to_xy(pts), np.ones((pts.shape[0], 1))))


def yx_to_xy(pts):
    '''
    takes points of shape (n x 2) (yx format)
    returns points of shape (n x 2) (xy format)
    '''
    return np.flip(pts, axis=1)
     

def RANSAC(pts1, pts2, n = 5000, threshold = 20):
    '''
    pts1 : numpy.ndarray of shape (n, 2) (yx format)
    pts2 : numpy.ndarray of shape (n, 2) (yx format)
    n : number of trialss
    threshold : threshold for inliers
    returns : best homography matrix, best number of inliers
    '''
    best = 0
    bestH = None
    pts1_xy1 = homogenize(pts1)
    pts2_xy1 = homogenize(pts2)

    for _ in range(n):
        indices = np.random.choice(np.arange(len(pts1)), 4, replace=False)
        p1 = yx_to_xy(pts1[indices])
        p2 = yx_to_xy(pts2[indices])
        H = compute_homography(p1, p2)
        
        transformed_all = H @ pts2_xy1.T
        transformed_all /= transformed_all[2]
        
        # compute inliers
        dist = np.linalg.norm(pts1_xy1.T - transformed_all, axis=0)
        inliers = np.sum(dist < threshold)
        
        if inliers > best:
            best = inliers
            bestH = H

    return bestH, best


def MST(weight):
    '''
    given a weight matrix, return the maxiumum spanning tree
    weight[i, j] = the weight if i < j
    if W[i, j] = 0, because i > j
    '''
    N = weight.shape[0]
    graph = [[] for _ in range(N)]
    
    for i in range(N):
        for j in range(i+1, N):
            graph[i].append((-weight[i, j], j, i))
            graph[j].append((-weight[i, j], i, j))
    
    visited = [False] * N
    visited[0] = True
    
    queue = graph[0].copy()
    heapq.heapify(queue)
    mst = []
    while queue:
        w, v, x = heapq.heappop(queue)
        if visited[v]: continue
        visited[v] = True
        mst.append((v, x))
        for w, u, _ in graph[v]:
            if not visited[u]:
                heapq.heappush(queue, (w, u, v))

    return mst


def adjacency_list_from_edges(edges, N):
    '''
    edges : list of tuples (weight, vertex)
    N : number of vertices
    returns : adjacency list
    '''
    graph = [[] for _ in range(N)]
    for x, y in edges:
        graph[x].append(y)
        graph[y].append(x)
    return graph