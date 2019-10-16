import argparse
import cv2
from imutils import paths
import os
import csv

def open_csv_file(file_name):
    '''
        This function is used to open a csv file
    '''
    try:
        with open(file_name, 'r') as read_file:
            reader = csv.reader(read_file)
            lines = list(reader)
            read_file.close()
            return lines
    except FileNotFoundError:
        open(file_name, 'a').close()
        return []

def write_csv_file(file_name, rows):
    '''
        This function is used to write into csv file
    '''
    with open(file_name, 'w') as write_file:
        writer = csv.writer(write_file)
        writer.writerows(rows)
        write_file.close()

def get_number_unique_images(rows):
    unique_images = []
    for row in rows:
        if row[0] not in unique_images:
            unique_images.append(row[0])

    return len(unique_images)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-iP", "--images", required=True, 
            help="Path to the directory that contains images")
    ap.add_argument("-csv", "--csv_file", required=True,
            help="Name of the CSV where to save the dataset")
    args = vars(ap.parse_args())

    images = paths.list_images(args["images"])

    # Instead of start in the same image every time, discard the ones that have 
    # already been seen
    lines = open_csv_file(args['csv_file'])
    number_unique_images = get_number_unique_images(lines)
    for i in range(0, number_unique_images):
        next(images)

    def click_and_draw(event, x, y, flags, param):
        '''
            This function is used to draw a rectangle around a specified area.
            This rectangle will be used as a bounding box for training

        '''
        global rect

        if event == cv2.EVENT_LBUTTONDOWN:
            rect = [(x, y)]
    
        elif event == cv2.EVENT_LBUTTONUP:  
            rect.append((x, y))

            # draw a rectangle around the region of interest
            cv2.rectangle(current_image, rect[0], rect[1], (0, 255, 0), 2)
            cv2.imshow("image", current_image)

    for image in images:

        # read the data that is already stored in the csv file
        rows = open_csv_file(args['csv_file'])
        print(len(rows))
        
        current_image = cv2.imread(image)
        clone = current_image.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click_and_draw)
        
        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", current_image)
            key = cv2.waitKey(33)
            # if the 'n' key is pressed, next image
            if key == ord("n"):
                print('next')
                break

            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                current_image = clone.copy()
                print('Reset image')
        
            # if the 'b' key is pressed, save bib configuration
            elif key == ord("b"):
                if len(rect) == 2:
                    image_path = os.path.abspath(image)

                    # Save into csv: image_path, x1, y1, x2, y2, class 
                    rows.append([image_path, rect[0][0], rect[0][1], rect[1][0], rect[1][1], "bib"])
                    print('bib configuration appended')
            
            # if the 'p' key is pressed, save the person configuration
            elif key == ord("p"):
                if len(rect) == 2:
                    image_path = os.path.abspath(image)

                    # Save into csv: image_path, x1, y1, x2, y2, class 
                    rows.append([image_path, rect[0][0], rect[0][1], rect[1][0], rect[1][1], "person"])
                    print('person configuration appended')

            elif key == ord("s"):
                write_csv_file(args['csv_file'], rows)
                print("Saved into csv")
                
            # if the 'e' key is pressed, exit
            elif key == ord("e"):
                print('exit')
                exit(0)

    # close all open windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 