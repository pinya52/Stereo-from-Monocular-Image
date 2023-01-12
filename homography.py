import numpy as np
import cv2

def get_sift_correspondences(img1, img2, good_rate):
    '''
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image
    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    '''
    #sift = cv.xfeatures2d.SIFT_create()# opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree
    sift = cv2.SIFT_create()             # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < good_rate * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)

    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

    return points1, points2

def main():
    img1_path = './kitti/data_scene_flow/training/image_2'
    img2_path = './kitti/data_scene_flow/training/image_3'

    img1 = cv2.imread('%s/000000_10.png'%(img1_path))
    img2 = cv2.imread('%s/000000_10.png'%(img2_path))

    points1, points2 = get_sift_correspondences(img1, img2, 0.75)

    h, _ = cv2.findHomography(points1, points2)

    idx = '132'
    src_img = cv2.imread('%s/000%s_10.png'%(img1_path, idx))
    (y, x) = src_img.shape[:2]

    warped_image = cv2.warpPerspective(src_img, h, (x, y))

    cv2.imwrite('warp.png', warped_image)
    cv2.imshow('a', warped_image)
    cv2.waitKey(0)

main()