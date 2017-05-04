import cv2
from matplotlib import pyplot as plt

"""
Finds edges in an image using canny86 algorithm.
More details can be found at: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html

"""

threshold1 = 100
threshold2 = 200


def detect_edges(img):
    canny_edges = cv2.Canny(img, threshold1, threshold2)  # better than sobel on test set
    # sobel_edges = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5) #features are being lost
    return canny_edges


if __name__ == "__main__":
    img = cv2.imread('Strada_Provinciale_BS_510_Sebina_Orientale.jpg', 0)

    edges = detect_edges(img)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Canny Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()

