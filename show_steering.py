import cv2
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image file")


def display_steering_wheel(image_name, angle):

    WIDTH = 1920
    HEIGHT = 1208
    WHEEL_SIZE = 300
    #name of the steering wheel
    wheel = cv2.imread("volant.png")
    image = cv2.imread(image_name)
    wheel = cv2.resize(wheel, (WHEEL_SIZE,WHEEL_SIZE))
    #wheel position
    x_offset = int(1920/2 - WHEEL_SIZE/2)
    y_offset = int(HEIGHT - WHEEL_SIZE - 50)
    #define rotaton center
    rot_x = int(WHEEL_SIZE/2 )
    rot_y = int(WHEEL_SIZE/2 )
    #ratation matrix a rotation itself
    rotation_matix = cv2.getRotationMatrix2D((rot_x,rot_y),45,1)
    rot_wheel = cv2.warpAffine(wheel,rotation_matix, (WHEEL_SIZE,WHEEL_SIZE))
    
    #if steering wheel overlays default image display sw instead
    for i in range (y_offset, y_offset+wheel.shape[0]):
        for j in range(x_offset, x_offset+wheel.shape[1]):
            pixel = rot_wheel[i-y_offset][j-x_offset]
            #sometimes there are black pixels because of rataton and white because of crop
            if not(sum(pixel) >= 660 or sum(pixel) == 0 ):
                image[i][j] = rot_wheel[i-y_offset][j-x_offset]
    #just to fit the screen, comment out if needed

    cv2.putText(image, ("Angle is " + str(angle)),(x_offset +40, y_offset + WHEEL_SIZE + 30),cv2.FONT_HERSHEY_TRIPLEX,1.5,(0,0,255))
    image = cv2.resize(image, (int(WIDTH*0.7), int(HEIGHT*0.7)))

    cv2.imshow("Image", image)
    cv2.waitKey(0)

def main():
    args = vars(ap.parse_args())
    image = args["image"]
    display_steering_wheel(image, 45);


if __name__ == "__main__":
    main()