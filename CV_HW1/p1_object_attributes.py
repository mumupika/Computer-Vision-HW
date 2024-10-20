#!/usr/bin/env python3
import cv2
import numpy as np
import sys

def binarize(gray_image:np.ndarray, thresh_val:int) -> np.ndarray:
    # histogram is to be done.
    binary_image = gray_image
    binary_image[binary_image >= thresh_val] = 255
    binary_image[binary_image <= thresh_val] = 0
    return binary_image

def label(binary_image:np.ndarray) -> np.ndarray:
    # deployed by the https://courses.cs.washington.edu/courses/cse373/00au/chcon.pdf
    
    def initialize() -> list:
        # Initialize the union set.
        maxLabel = 10
        parent_list = [0 for i in range(maxLabel+1)]
        return parent_list
    
    def prior_neighbors(i:int, j:int, labeled_image: np.ndarray) -> list:
        # The first neighbors.
        neighbors = list()
        if i > 0 and labeled_image[i-1,j] > 0:
            neighbors.append(labeled_image[i-1,j])
        if j > 0 and labeled_image[i,j-1] > 0:
            neighbors.append(labeled_image[i,j-1])
        return neighbors
    
    def union(cur:int ,label:int, parent:list):
        # The union procedure.
        if cur >= len(parent) or label >= len(parent):
            # deal with the size.
            ranges=max(cur,label) * 2
            new_parent=[0 for i in range(max(cur,label)*2)]
            new_parent[:len(parent)]=parent
            parent=new_parent
        
        while parent[cur] != 0:
            cur = parent[cur]
        while parent[label] != 0:
            label = parent[label]
        
        if cur != label:
            parent[label]=cur
        return parent
    
    def find(cur:int, parent:list) -> int:
        # Finding the union set.
        while parent[cur]!=0:
            cur=parent[cur]
        return cur
        
    MaxRow, MaxCol=binary_image.shape
    parent = initialize()
    # Initialize all labels to 0.
    labeled_image = np.full(shape=(MaxRow,MaxCol),fill_value=0,dtype=int)
    label_counter=0
    for i in range(MaxRow):
        for j in range(MaxCol):
            # process Line L.
            if binary_image[i,j] != 0:
                neighbors = prior_neighbors(i,j,labeled_image)
                if len(neighbors) == 0:
                    label_counter += 1
                    labeled_image[i,j] = label_counter
                else:
                    labeled_image[i,j] = min(neighbors)
                    for label in neighbors:
                        if label != labeled_image[i,j]:
                            parent=union(labeled_image[i,j],label,parent)
    
    # The second pass.
    for i in range(MaxRow):
        for j in range(MaxCol):
            if binary_image[i,j]!=0:
                labeled_image[i,j]=find(labeled_image[i,j],parent)
    
    return labeled_image
    
    
def get_attribute(labeled_image):
    # reference: https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/OWENS/LECT2/node3.html
    # We find the label level.
    height,width = labeled_image.shape
    label_level=[]
    for i in range(height):
        for j in range(width):
            if labeled_image[i,j]!=0 and label_level.count(labeled_image[i,j])==0:
                label_level.append(labeled_image[i,j])
    
    # We process with the object one by one.
    # To derive all the pictures into the different pictures.
    picture_list=[]
    for i in range(len(label_level)):
        one_label_img = np.array(labeled_image)
        one_label_img[one_label_img != label_level[i]] = 0
        picture_list.append(one_label_img)
        
    # Remember the origin is at the bottom left of the picture.
    attribute_list=list()   # label: x,y,orientation.
    for i in range(len(picture_list)):
        object_pattern = np.array(picture_list[i],dtype=np.float32)/label_level[i]  # normalize.
        # Calculate Area,x_bar,y_bar,a,b,c.
        area:np.float64 = np.sum(object_pattern)
        height_array:np.ndarray = np.arange(height,0,-1)    # Be aware of reverse order.
        width_array:np.ndarray = np.arange(1,width+1,1) 
        x_bar:np.float64 = np.sum(width_array @ object_pattern.T) / area
        y_bar:np.float64 = np.sum(height_array @ object_pattern) / area
        a_org:np.float64 = np.sum(object_pattern @ (width_array**2).reshape((-1,1)))
        b_org:np.float64 = 2*np.sum(height_array.reshape((1,-1)) @ object_pattern @ width_array.reshape((-1,1)))
        c_org:np.float64 = np.sum((height_array**2).reshape((1,-1)) @ object_pattern)
        a:np.float64 = (a_org - x_bar ** 2 * area)
        b:np.float64 = (b_org - 2 * x_bar * y_bar * area)
        c:np.float64 = (c_org - y_bar ** 2 * area)
        theta_1=np.arctan2(b,(a-c))/2
        theta_2=theta_1 + np.pi/2
        Energy = lambda a,b,c,theta: a*(np.sin(theta)**2) - b*np.sin(theta)*np.cos(theta) + c*(np.cos(theta)**2)
        roundedness = Energy(a,b,c,theta_1) / Energy(a,b,c,theta_2)
        positions={'x':x_bar,'y':y_bar}
        attribute={
            'label': label_level[i],
            'position':positions,
            'orientation': theta_1,
            'roundedness': roundedness
        }
        attribute_list.append(attribute)
    return attribute_list

def my_print(attribute_list:list[dict]) -> None:
    # This is print part.
    num=0
    for attribute in attribute_list:
        num += 1
        print(f"object {num}")
        for key,value in attribute.items():
            if key=='position':
                print('position:',f"({np.round(value['x'],6)}, {np.round(value['y'],6)})")
            else:
                print(f'{key}: {np.round(value,6)}')
        print('----------------------')
    print('\n\n\n')
    
def main(argv):
    img_name = argv[0]
    thresh_val = int(argv[1])
    img = cv2.imread('./CV_HW1/data/' + img_name + '.png', cv2.IMREAD_COLOR)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_image = binarize(gray_image, thresh_val=thresh_val)
    labeled_image = label(binary_image)
    attribute_list = get_attribute(labeled_image)
    
    cv2.imwrite('./CV_HW1/output/' + img_name + "_gray.png", gray_image)
    cv2.imwrite('./CV_HW1/output/' + img_name + "_binary.png", binary_image)
    cv2.imwrite('./CV_HW1/output/' + img_name + "_labeled.png", labeled_image)
    my_print(attribute_list)

if __name__ == '__main__':
  	main(sys.argv[1:])