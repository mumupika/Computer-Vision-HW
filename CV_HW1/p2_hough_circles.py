#!/usr/bin/env python3
import cv2
import numpy as np


def detect_edges(image):
    """Find edge points in a grayscale image.

    Args:
    - image (2D uint8 array): A grayscale image.

    Return:
    - edge_image (2D float array): A heat map where the intensity at each point
        is proportional to the edge magnitude.
    """
    
    # define the sobel opt. Using (3,3) sobel.
    sobel_x = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ])
    sobel_y = np.array([
        [1,2,1],
        [0,0,0],
        [-1,-2,-1]
    ])
    
    # Implement the convolution part.
    def conv(image:np.ndarray, kernel:np.array, model:str) -> np.ndarray:
        if model == '8U':
            curType = np.uint8
        if model == '32F':
            curType = np.float32
        h, w = image.shape
        k = kernel.shape[0]
        p = (k-1) // 2  # The padding size.
        
        def padding(image: np.array, k: int, curType: np.dtype) -> np.ndarray:
            # TODO: To implement it to fix sobel(5,5).
            padded_image = np.zeros((h+2*p,w+2*p),dtype=curType)
            # border reflect pad.
            padded_image[0,1:w+p] = np.array(image[1,:],dtype=curType)
            padded_image[1:h+p,0] = np.array(image[:,1],dtype=curType)
            padded_image[-1,1:w+p] = np.array(image[-2,:],dtype=curType)
            padded_image[1:h+p,-1] = np.array(image[:,-2],dtype=curType)
            # second four corner points.
            padded_image[0,0] = np.array(image[1,1],dtype=curType)
            padded_image[0,-1] = np.array(image[1,-2],dtype=curType)
            padded_image[-1,0] = np.array(image[-2,1],dtype=curType)
            padded_image[-1,-1] = np.array(image[-2,-2],dtype=curType)
            # third execute filling.
            padded_image[p:h+p,p:w+p] = np.array(image,dtype=curType)
            return padded_image
        
        padded_image = padding(image, k, np.float32)
        conv_image = np.zeros((h,w),dtype=curType)
      
        for i in range(h):
            for j in range(w):
                conv_image[i,j] = np.sum(np.array(padded_image[i:i+2*p+1, j:j+2*p+1] * kernel,dtype=curType)).astype(curType)
        
        return conv_image
    
    # Execute the sobel convolution.
    conv_x = conv(image=image, kernel=sobel_x, model='32F')
    conv_y = conv(image=image, kernel=sobel_y, model='32F')
    
    # Make the final gradient with square sum and root.
    edge_image = (np.abs(conv_x)+np.abs(conv_y))
    cv2.imwrite('./CV_HW1/mySobel.png',edge_image)
    
    # official sobel convolution.
    grad_x = cv2.Sobel(image,cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image,cv2.CV_32F, 0, 1, ksize=3)
    G = (np.abs(grad_x)+np.abs(grad_y))
    cv2.imwrite('./CV_HW1/officialSobel.png',G)
    print("Equal to official implementation?",np.array_equal(G,edge_image))
    print("Dtype is float32?",edge_image.dtype==np.float32)
    return edge_image


def hough_circles(edge_image:np.ndarray, edge_thresh:float, radius_values:list) -> np.ndarray:
    """Threshold edge image and calculate the Hough transform accumulator array.

    Args:
    - edge_image (2D float array): An H x W heat map where the intensity at each
        point is proportional to the edge magnitude.
    - edge_thresh (float): A threshold on the edge magnitude values.
    - radius_values (1D int array): An array of R possible radius values.

    Return:
    - thresh_edge_image (2D bool array): Thresholded edge image indicating
        whether each pixel is an edge point or not.
    - accum_array (3D int array): Hough transform accumulator array. Should have
        shape R x H x W.
    """
    thresh_edge_image = edge_image
    thresh_edge_image[edge_image < edge_thresh] = 0
    thresh_edge_image[edge_image >= edge_thresh] = 255
    cv2.imwrite('./CV_HW1/output/coins_edge.png',thresh_edge_image)
    accum_array = []
    H,W = thresh_edge_image.shape
    R_MAX = radius_values[-1]
    for r in radius_values:
        accum_array_R = np.zeros((H+2*R_MAX,W+2*R_MAX),dtype=np.int32)  # Expand the border of the pictures.
        theta = np.arange(0,(2*np.pi+0.1),0.1)
        for i in range(H):
            for j in range(W):
                if thresh_edge_image[i,j] == 0:
                    continue
                x:np.ndarray = (j + r * np.cos(theta)).astype(np.int16)
                y:np.ndarray = (i + r * np.sin(theta)).astype(np.int16)
                
                accum_array_R[y+r,x+r] += 1
        accum_array.append(accum_array_R)
    
    accum_array:np.ndarray = np.array(accum_array,dtype = np.int32)
    
    for i in range(accum_array.shape[0]):
        cv2.imwrite(f'./CV_HW1/hough_transform/{i}.png',np.array(20*accum_array[i,:,:],dtype=np.float32))
    
    return thresh_edge_image,accum_array
    


def find_circles(image:np.ndarray, accum_array:np.ndarray, radius_values:list, hough_thresh:int):
    """Find circles in an image using output from Hough transform.

    Args:
    - image (3D uint8 array): An H x W x 3 BGR color image. Here we use the
        original color image instead of its grayscale version so the circles
        can be drawn in color.
    - accum_array (3D int array): Hough transform accumulator array having shape
        R x H x W.
    - radius_values (1D int array): An array of R radius values.
    - hough_thresh (int): A threshold of votes in the accumulator array.

    Return:
    - circles (list of 3-tuples): A list of circle parameters. Each element
        (r, y, x) represents the radius and the center coordinates of a circle
        found by the program.
    - circle_image (3D uint8 array): A copy of the original image with detected
        circles drawn in color.
    """
    R,H,W = accum_array.shape
    orig_H, orig_W, _ = image.shape        # This is the original 3 RGB color image.
    
    # The first pass: To filter out the possible radius.
    circles = list()
    for r in range(R):
        accum_array_R = accum_array[r,:,:]
        current_y,current_x = np.where(accum_array_R > hough_thresh)
        for i in range(current_x.shape[0]):
            circles.append((r,int(current_x[i]),int(current_y[i])))
    
    # second pass: To merge the value that was very close. sort by dictionary order, then choose the 
    # brightest center and radius.
    real_circles = list()
    count = 0
    circles.sort(key=lambda x: x[1])
    while count < len(circles):
        r,x,y = circles[count]
        lightness_chosen = accum_array[r,x,y]
        true_r, true_x, true_y = r,x,y
        count += 1
        while  count < len(circles) and abs(circles[count][1]-x) + abs(circles[count][2]-y) < 9:
            cur_r,cur_x,cur_y = circles[count]
            lightness = accum_array[cur_r,cur_x,cur_y]
            count += 1
            if lightness >= lightness_chosen:
                true_r, true_x, true_y = cur_r, cur_x, cur_y
        real_circles.append((true_r,true_x-true_r,true_y-true_r))
    
    real_circles.sort(key = lambda x: x[0])
    
    # third step: Get the circles draw in color. Using cv2's drawing.
    for circle in real_circles:
        cv2.circle(image, (circle[1],circle[2]),circle[0],(0,255,0),2)
    cv2.imwrite('./CV_HW1/output/coins_circle.png',image)
            

if __name__ == '__main__':
    input_dir = './CV_HW1/data/coins.png'
    image: np.ndarray = cv2.imread(input_dir,cv2.IMREAD_COLOR)
    gray_image: np.ndarray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_image: np.ndarray = cv2.GaussianBlur(gray_image,(5,5),1,sigmaY=1)
    edge_image:np.ndarray = detect_edges(gray_image)
    radius_value = [ i for i in range(1,41)]
    thresh_edge_image, accum_array = hough_circles(edge_image=edge_image,edge_thresh=200,radius_values=radius_value)
    find_circles(image=image, accum_array=accum_array, radius_values=radius_value, hough_thresh=42)
