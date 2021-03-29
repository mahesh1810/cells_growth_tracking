#Program to track human body cells growth in a petry dish
#input: image of cells in a petri dish 
#output: image of cells identified with contours drawn around and area of contours
import cv2
import numpy as np
import argparse

def get_contour_areas(contours):
    # returns the areas of all contours as list
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas

def display_image(caption, image):
    cv2.imshow(caption, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    ## Load images from the file location
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--image", default='./images/cell3.jpeg', help="path for the cell image ")
    args = parser.parse_args()
    # Load our image
    image = cv2.imread(args.image)
    display_image('0 - Original Image', image) 
    area=find_cells_area(image)
    return(area)
    
def find_cells_area(image):
    # Create a copy of our original image
    orginal_image = image.copy()
    # Grayscale image
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Find Canny edges
    edged = cv2.Canny(gray, 50, 60)
    kernel = np.ones((1,1), np.uint8)
    edged = cv2.erode(edged, kernel, iterations = 1)
    kernel_sharpening = np.array([[-1,-1,-1], 
                                  [-1,9,-1], 
                                  [-1,-1,-1]])
    # applying different kernels to the input image
    edged = cv2.filter2D(edged, -1, kernel_sharpening)
    # Find contours and print how many were found
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print ("Number of contours found = ", len(contours))
    # Sort contours large to small
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    area=sum(get_contour_areas(sorted_contours))
    print(area)
    display_image('original', orginal_image)
    # Iterate over our contours and draw one at a time
    for c in sorted_contours:
        cv2.drawContours(image, [c], -1, (255,0,0), 1)
    text = "Area: " + str(area)+" Pixel"
    cv2.putText(image, text, (250, 30), cv2.FONT_HERSHEY_SIMPLEX,1 , (0, 100, 255), 2)
    display_image('Contours by area', image)
    # Save the ouput image
    cv2.imwrite('./images/cell3_output.jpeg',image)
    return(area)

if __name__ == "__main__":
    main()
    
    





