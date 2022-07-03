import cv2
from matplotlib import pyplot as plt
from procesar_imagenes import *
import skimage

frames = 30
kernel = 49
num_video = 1
w = h = f1 = f2 = ew = eh = 0

def cut(event, x, y, flags, param):
    global w, ew, h, eh, f1 ,f2
    img = 0
    # La función de esto es registrar la posición inicial cuando se presiona el botón izquierdo del mouse
    if event == cv2.EVENT_LBUTTONDOWN:
        f1 = 1
        h = x
        w = y
    # La función de este paso es dibujar un marco cuando se desliza el mouse y cuando se presiona el botón izquierdo, y registrar la posición de la parte posterior izquierda
    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        img = frame.copy()
        cv2.rectangle(img, (h,w), (x,y), (255,0,0),5)
        cv2.imshow("frame", img)
        f2 = 1
        eh = x
        ew = y
    # Cuando se ejecutan los dos pasos anteriores y se suelta el botón izquierdo, puede interceptar una imagen.
    if f1 == 1 and f2 == 1 and event == cv2.EVENT_LBUTTONUP:
        # Los siguientes dos ifs son consideraciones para tomar capturas de pantalla en todas las direcciones
        if ew < w:
            w,ew = ew,w
        if eh < h:
            eh,h = h, eh
        f1 = f2 = 0
    w1 = w
    w2 = ew
    h1 = h
    h2 = eh

#PROGRAMA PRINCIPAL
while True:
    print("1) Elegir cantidad de frames")
    print("2) Elegir tamaño del kernel")
    print("3) Elegir video")
    print("4) Comenzar")
    print("5) Salir")
    opcion = int(input("Seleccione una opcion: "))
    if(opcion == 1):
        frame = int(input("Ingrese la cantidad de frames a trabajar: "))
    if(opcion == 2):
        kernel = int(input("Ingrese el tamaño del kernel a trabajar: "))
    if(opcion == 3):
        print("1) Video 1")
        print("2) Video 6")
        print("3) Video 7")
        print("4) Video 10")
        opcion_video = int(input("Seleccione un video: "))
        if(opcion_video == 1):
            num_video = 1
        elif(opcion_video == 2):
            num_video = 6
        elif(opcion_video == 3):
            num_video = 7
        elif(opcion_video == 4):
            num_video = 10
        else:
            print("**VIDEO INVALIDO**")
            print("**VIDEO 1 DE FORMA PREDETERMINADA**")
    if(opcion == 4):
        path = "Videos\Video"+str(num_video)+".mp4"
        cap = cv2.VideoCapture(path)
        cv2.namedWindow("frame",cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("frame",cut)
        count = 0
        while True:
            ret, frame = cap.read()
            if ret == False:
                break
            cv2.imshow("frame", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            if cv2.waitKey(10) & 0xFF == ord('k'):
                print("Video pausado")
                while True:
                    if cv2.waitKey(10) == ord("c"):
                        print("Recortando Fotos")
                        for cant in range(frames):
                            img = frame[w:ew, h:eh]
                            path = "Recortes\imagen" + "%d"%cant+".jpg"
                            cv2.imwrite(path,img)
                            ret, frame = cap.read()
                        stack_red, stack_green, stack_blue = registrarImagenes(frames)
                        shape = (stack_red.shape[1], stack_red.shape[2])

                        img_original = cv2.imread("Recortes\imagen0.jpg")

                        img_high_boost = aplicarHighBoostImagenes(stack_red, stack_green, stack_blue)

                        img_ecualizada_pos = ecualizarImagenes_Local(img_high_boost, kernel)
                        img_ecualizada_ad = skimage.exposure.equalize_adapthist(img_high_boost, kernel)

                        fig = plt.figure()
                        fig.suptitle("Imagenes Registradas")
                        h1 = plt.subplot(1,3,1), plt.imshow(img_original), plt.title('Imagen Original')
                        h2 = plt.subplot(1,3,2), plt.imshow(img_ecualizada_pos, cmap='Greys'), plt.title('Ecualizacion Local')
                        h3 = plt.subplot(1,3,3), plt.imshow(img_ecualizada_ad), plt.title('Ecualizacion Adaptativa')
                        plt.show()
                    if cv2.waitKey(10) & 0xFF == ord('k'):
                        print("Video reanudado")
                        break
        cap.release()
        cv2.destroyAllWindows()
    if(opcion == 5):
        break
    else:
        print("**OPCION INVALIDA**")