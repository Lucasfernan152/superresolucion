import cv2
from cv2 import cvtColor
from matplotlib import pyplot as plt
from procesar_imagenes import *
import skimage
import os

frames = 30
kernel = 15
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

def grabarVideo(image_folder, video_name):
    video_name = 'Videos\\'+video_name+'.mp4'
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width,height))
    for image in images:
        for i in range(10):
            video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()

def reproducirVideos(path1, path2, tipoColor):
    cap1 = cv2.VideoCapture(path1)
    cv2.namedWindow("frame1", cv2.WINDOW_NORMAL)
    cap2 = cv2.VideoCapture(path2)
    cv2.namedWindow("frame2", cv2.WINDOW_NORMAL)
    frame_counter = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        frame_counter += 1
        if ret1 == False or ret2 == False:
            break
        if frame_counter == cap1.get(cv2.CAP_PROP_FRAME_COUNT) or frame_counter == cap2.get(cv2.CAP_PROP_FRAME_COUNT):
            frame_counter = 0
            cap1 = cv2.VideoCapture(path1)
            cap2 = cv2.VideoCapture(path2)
        cv2.imshow("frame1", frame1)
        if(tipoColor == 2):
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_HSV2RGB)
        cv2.imshow("frame2", frame2)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

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
    elif(opcion == 2):
        kernel = int(input("Ingrese el tamaño del kernel a trabajar: "))
    elif(opcion == 3):
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
    elif(opcion == 4):
        path = "Videos\Video"+str(num_video)+".mp4"
        cap = cv2.VideoCapture(path)
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("frame", cut)
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
                        print('1) Ecualizar mediante RGB')
                        print('2) Ecualizar mediante HSV')
                        tipoColor = int(input('Seleccione una opcion: '))
                        stack_red, stack_green, stack_blue = registrarImagenes(frames, tipoColor)
                        shape = (stack_red.shape[1], stack_red.shape[2])

                        #Mostrar imagenes recortadas y registradas
                        grabarVideo('Recortes', 'Recortadas')
                        grabarVideo('Registradas', 'Registradas')
                        reproducirVideos('Videos\Recortadas.mp4', 'Videos\Registradas.mp4', tipoColor)

                        img_original = cv2.cvtColor(cv2.imread("Recortes\imagen0.jpg"), cv2.COLOR_BGR2RGB)
                        img_high_boost = cv2.cvtColor(aplicarHighBoostImagenes(stack_red, stack_green, stack_blue), cv2.COLOR_BGR2RGB)
                        img_ecualizada_pos = ecualizarImagenes_Local(img_high_boost, kernel)
                        img_ecualizada_ad = skimage.exposure.equalize_adapthist(cv2.cvtColor(img_high_boost, cv2.COLOR_BGR2GRAY), kernel)
                        fig = plt.figure()
                        fig.suptitle("Imagenes Registradas")
                        h1 = plt.subplot(2,2,1), plt.imshow(img_original), plt.title('Imagen Original')
                        h1 = plt.subplot(2,2,2), plt.imshow(img_high_boost), plt.title('Imagen HighBoost')
                        h2 = plt.subplot(2,2,3), plt.imshow(img_ecualizada_pos, cmap='Greys'), plt.title('Ecualizacion Local')
                        h3 = plt.subplot(2,2,4), plt.imshow(img_ecualizada_ad, cmap='Greys'), plt.title('Ecualizacion Adaptativa')
                        plt.show()
                        break
                    if cv2.waitKey(10) & 0xFF == ord('k'):
                        print("Video reanudado")
                        break
        cap.release()
        cv2.destroyAllWindows()
    elif(opcion == 5):
        break
    else:
        print("**OPCION "+opcion+" INVALIDA**")