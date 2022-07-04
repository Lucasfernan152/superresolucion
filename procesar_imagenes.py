import cv2
import matplotlib.image as img
import numpy as np

def registrarImagenes(frames):
    orb_detector = cv2.ORB_create(5000)
    set_red = []
    set_green = []
    set_blue = []
    path = "Recortes\imagen0.jpg"
    image_base = img.imread(path)
    
    img2 = cv2.cvtColor(image_base, cv2.COLOR_BGR2GRAY)
    kp1, d1 = orb_detector.detectAndCompute(img2, None) 

    for j in range(0, frames):
        path1 = "Recortes\imagen"+str(j)+".jpg"
        image_registrar = img.imread(path1)

        img1 = cv2.cvtColor(image_registrar, cv2.COLOR_BGR2GRAY)
        height, width = img2.shape
        
        kp1, d1 = orb_detector.detectAndCompute(img1, None) 
        kp2, d2 = orb_detector.detectAndCompute(img2, None) 
        
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 
        
        matches = matcher.match(d1, d2) 
        
        matches = sorted(matches, key = lambda x: x.distance)
        matches = tuple(matches)
        
        matches = matches[:int(len(matches)*90)]
        no_of_matches = len(matches) 
        
        p1 = np.zeros((no_of_matches, 2)) 
        p2 = np.zeros((no_of_matches, 2)) 
        
        for i in range(len(matches)): 
            p1[i, :] = kp1[matches[i].queryIdx].pt 
            p2[i, :] = kp2[matches[i].trainIdx].pt 
        
        homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 
        
        transformed_img = (cv2.warpPerspective(image_registrar, homography,(width, height)))
        cv2.imwrite("Registradas\imagen"+str(j)+".jpg",transformed_img)

        set_red.append(transformed_img[::,::,0])
        set_green.append(transformed_img[::,::,1])
        set_blue.append(transformed_img[::,::,2])
    return stackearImagenes(set_red, set_green, set_blue)

def stackearImagenes(set_red, set_green, set_blue):
    stack_red = np.stack(set_red)
    stack_green = np.stack(set_green)
    stack_blue = np.stack(set_blue)
    return stack_red, stack_green, stack_blue

def promediarImagenes(list_red, list_green, list_blue):
    array_red = np.array(list_red)
    array_green = np.array(list_green)
    array_blue = np.array(list_blue)

    img_red_prom = np.mean(array_red, 0)
    img_green_prom = np.mean(array_green, 0)
    img_blue_prom = np.mean(array_blue, 0)

    array_prom = np.zeros((img_red_prom.shape[0], img_red_prom.shape[1], 3))
    array_prom[::,::,0] = img_red_prom
    array_prom[::,::,1] = img_green_prom
    array_prom[::,::,2] = img_blue_prom
    array_prom = array_prom.astype(np.uint8)
    return array_prom

def ecualizarImagenes(stack_red, stack_green, stack_blue):
    list_red = []
    list_green = []
    list_blue = []
    frames = stack_red.shape[0]
    for i in range(frames):
        equ_red = cv2.equalizeHist(stack_red[i,::,::])
        equ_green = cv2.equalizeHist(stack_green[i,::,::])
        equ_blue = cv2.equalizeHist(stack_blue[i,::,::])
        list_red.append(equ_red)
        list_green.append(equ_green)
        list_blue.append(equ_blue)
    return promediarImagenes(list_red, list_green, list_blue)

def aplicarMediaImagenes(stack_red, stack_green, stack_blue):
    list_red = []
    list_green = []
    list_blue = []
    frames = stack_red.shape[0]
    for i in range(frames):
        kernel = np.ones((3,3),np.float32)/9
        equ_red = cv2.filter2D(stack_red[i,::,::],-1,kernel)
        equ_green = cv2.filter2D(stack_green[i,::,::],-1,kernel)
        equ_blue = cv2.filter2D(stack_blue[i,::,::],-1,kernel)
        list_red.append(equ_red)
        list_green.append(equ_green)
        list_blue.append(equ_blue)
    return promediarImagenes(list_red, list_green, list_blue)

def aplicarHighBoostImagenes(stack_red, stack_green, stack_blue):
    imgMedia = aplicarMediaImagenes(stack_red, stack_green, stack_blue)
    imgOriginal = cv2.imread("Recortes\imagen0.jpg")
    imgHighBoost = 2*imgOriginal-imgMedia
    return imgHighBoost

def ecualizarImagenes_Local(image, kernel=3):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    delta = (kernel-1)//2
    pcentral = (kernel+1)//2
    matrizDistancias = obtenerMatrizDistancias(kernel-1)
    shape = img_gray.shape
    nueva = np.zeros(shape, dtype="uint8")
    for i in range(delta, shape[0]-delta):
        for j in range(delta, shape[1]-delta):
            ecualize = cv2.equalizeHist(img_gray[i-delta:i+delta+1,j-delta:j+delta+1])
            #nueva[i,j] = np.mean(ecualize*matrizDistancias)
            nueva[i,j] = ecualize[pcentral, pcentral]
    return nueva

def ecualizarImagenes_Local_pre(stack_red, stack_green, stack_blue, kernel=3):
    list_red = []
    list_green = []
    list_blue = []
    frames = stack_red.shape[0]
    for i in range(frames):
        equ_red = ecualizarImagenes_Local(stack_red[i,::,::], kernel)
        equ_green = ecualizarImagenes_Local(stack_green[i,::,::], kernel)
        equ_blue = ecualizarImagenes_Local(stack_blue[i,::,::], kernel)
        list_red.append(equ_red)
        list_green.append(equ_green)
        list_blue.append(equ_blue)
    return promediarImagenes(list_red, list_green, list_blue)

def obtenerMatrizDistancias(kernel):
    matriz = np.ones((kernel, kernel))
    centro = [(kernel+1)//2-1, (kernel+1)//2-1]
    for i in range(kernel):
        for j in range(kernel):
            c1 = centro[0]-i
            c2 = centro[1]-j
            distancia = np.sqrt((c1*c1)+(c2*c2))
            matriz[i, j] = kernel-distancia
    return matriz

def imadjust(x,a,b,c,d,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y