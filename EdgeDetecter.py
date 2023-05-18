import numpy as np
import cv2
import matplotlib.pyplot as plt

jjangu_image = cv2.imread("mangu.png", cv2.IMREAD_GRAYSCALE)

cv2.imshow("original_image_gray_scale", jjangu_image)

def sobel_convolve(array,direction):
    padding = np.pad(array, ((1, 1), (1, 1)), 'constant', constant_values=0)

    result_image = np.ones((len(array), len(array[0])))
    result_image = result_image.astype('float32')

    filter = np.zeros((3, 3), dtype=np.float32)
    Sy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Sx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    if direction == 'y' :
        filter = Sy
    elif direction == 'x' :
        filter = Sx

    for i in range(len(result_image)):
        for j in range(len(result_image[0])):
            result_image[i][j] = np.sum(padding[i:i + len(filter), j:j + len(filter)] * filter)
    return result_image


def find_gradient_magnitude(image) :
    Sobel_res_image_y = sobel_convolve(image, 'y')
    Sobel_res_image_x = sobel_convolve(image, 'x')

    # cv2.imshow('Sobel_y', Sobel_res_image_y)
    # cv2.imshow('Sobel_x', Sobel_res_image_x)

    magnit = np.hypot(Sobel_res_image_x, Sobel_res_image_y)
    magnit = magnit / magnit.max() * 255
    gradient = np.arctan2(Sobel_res_image_y, Sobel_res_image_x)

    return (magnit, gradient, Sobel_res_image_x, Sobel_res_image_y)

def non_maxima_suppression(magnit, gradient) :
    img_height, img_width = magnit.shape
    direction = gradient * 180. / np.pi
    direction[direction < 0] += 180

    res = np.zeros((img_height,img_width),dtype=np.int8)

    for i in range(1,img_height-1):
        for j in range(1,img_width-1):

            if (0 <= direction[i, j] < 22.5) or (157.5 <= direction[i, j] <= 180):
                left = magnit[i, j - 1]
                right = magnit[i, j + 1]


            elif 22.5 <= direction[i, j] < 67.5:
                left = magnit[i - 1, j + 1]
                right = magnit[i + 1, j - 1]


            elif 67.5 <= direction[i, j] < 112.5:
                left = magnit[i - 1, j]
                right = magnit[i + 1, j]


            elif 112.5 <= direction[i, j] < 157.5:
                left = magnit[i - 1, j - 1]
                right = magnit[i + 1, j + 1]

            if (magnit[i, j] > left) and (magnit[i, j] > right):
                res[i, j] = magnit[i, j]
    return res

def double_thresholding(image) :
    high_threshold = image.max() * 0.2
    low_threshold = image.max() * 0.07

    res = np.zeros(image.shape,dtype=np.int32)

    strong_x, strong_y = np.where(image > high_threshold)
    weak_x, weak_y = np.where((image<=high_threshold) & (image >= low_threshold))
    
    res[strong_x, strong_y] = 255
    res[weak_x, weak_y] = 120

    return res

def hysteresis(img):
    H,W = img.shape

    for i in range(1,H-1):
        for j in range(1,W-1):
            if(img[i,j] == 120):
                if ((img[i + 1, j - 1] == 255) or (img[i + 1, j] == 255) or (img[i + 1, j + 1] == 255)
                        or (img[i, j - 1] == 255) or (img[i, j + 1] == 255)
                        or (img[i - 1, j - 1] == 255) or (img[i - 1, j] == 255) or (img[i - 1, j + 1] == 255)):
                    img[i, j] = 255
                else:
                    img[i, j] = 0
    return img


def mooreNeighborTracing(image):
    inside = False
    paddedImage = np.pad(image, ((1, 1), (1, 1)), 'constant', constant_values=0)
    img_height,img_width = image.shape
    
    borderImage = np.zeros((img_height+2, img_width+2),dtype=np.int32)


    for y in range (1, img_height) :
        for x in range (1, img_width) :

            pos = [y, x]

            if borderImage[pos[0]][pos[1]] == 255 and inside==False :
                inside = True

            elif paddedImage[pos[0]][pos[1]] == 255 and inside==True :
                continue

            elif paddedImage[pos[0]][pos[1]] == 0 and inside==True : 
                inside = False

            elif paddedImage[pos[0]][pos[1]] == 255 and inside==False : 
                borderImage[pos[0]][pos[1]] = 255
                checkLocationNr = 0
                startPos = pos
                counter = 0
                counter2 = 0

                neighborhood = [[0, -1, 6],[-1, -1, 6],[-1, 0, 0],[-1, 1, 0],[0, 1, 2],[1, 1, 2],[1, 0, 4],[1, -1, 4]]
                
                while True :
                    checkPosition = [pos[0] + neighborhood[checkLocationNr][0], pos[1] + neighborhood[checkLocationNr][1]]
                    newCheckLocationNr = neighborhood[checkLocationNr][2]

                    if paddedImage[checkPosition[0]][checkPosition[1]] == 255 :

                        if checkPosition == startPos :
                            counter = counter+1
                            if newCheckLocationNr == 0 or counter >= 3 : 
                                inside = True
                                break
                        checkLocationNr = newCheckLocationNr
                        pos = checkPosition
                        counter2 = 0             
                        borderImage[checkPosition[0]][checkPosition[1]] = 255
                    else :
                        checkLocationNr = (checkLocationNr+1) % 8
                        if counter2 > 8 : 
                            counter2 = 0
                            break
                        else : 
                            counter2 = counter2 + 1
    return borderImage

fig = plt.figure()
rows = 2
cols = 4

image1 = fig.add_subplot(rows, cols, 1)
image1.imshow(cv2.cvtColor(jjangu_image, cv2.COLOR_BGR2RGB))
image1.set_title('original_image')
image1.axis("off")

magnit, gradient, S_arr_x, S_arr_y = find_gradient_magnitude(jjangu_image)
image2 = fig.add_subplot(rows, cols, 2)
image2.imshow(cv2.cvtColor(S_arr_y, cv2.COLOR_BGR2RGB))
image2.set_title('Sobel_filter_y')
image2.axis("off")

image3 = fig.add_subplot(rows, cols, 3)
image3.imshow(cv2.cvtColor(S_arr_x, cv2.COLOR_BGR2RGB))
image3.set_title('Sobel_filter_x')
image3.axis("off")

image4 = fig.add_subplot(rows, cols, 4)
magnit = np.array(magnit, dtype=np.uint8)
image4.imshow(cv2.cvtColor(magnit, cv2.COLOR_BGR2RGB))
image4.set_title('magnitude_image')
image4.axis("off")

image5 = fig.add_subplot(rows, cols, 5)
nms_result = np.array(non_maxima_suppression(magnit, gradient), dtype = np.uint8)
image5.imshow(cv2.cvtColor(nms_result, cv2.COLOR_BGR2RGB))
image5.set_title('non_maxima_suppression_image')
image5.axis("off")

image6 = fig.add_subplot(rows, cols, 6)
d_arr = np.array(double_thresholding(magnit), dtype = np.uint8)
image6.imshow(cv2.cvtColor(d_arr, cv2.COLOR_BGR2RGB))
image6.set_title('Double_thresholding')
image6.axis("off")

image7 = fig.add_subplot(rows, cols, 7)
h_arr = np.array(hysteresis(d_arr), dtype = np.uint8)
image7.imshow(cv2.cvtColor(h_arr, cv2.COLOR_BGR2RGB))
image7.set_title('hysteresis')
image7.axis("off")

image8 = fig.add_subplot(rows, cols, 8)
Moore_res = np.array(mooreNeighborTracing(h_arr), dtype = np.uint8)
image8.imshow(cv2.cvtColor(Moore_res, cv2.COLOR_BGR2RGB))
image8.set_title('Moore_Boundary_tracing')
image8.axis("off")

plt.show()
