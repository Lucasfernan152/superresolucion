import cv2
from matplotlib import pyplot as plt
from procesar_imagenes import *


i = w = h = f1 = f2 = ew = eh = 0
frames = 60

def crearRecorte(cant):
    img = frame[w:ew, h:eh]
    path = "PDI\Recortes\imagen" + "%d"%i+".jpg"
    cv2.imwrite(path,img)

def cut(event, x, y, flags, param):
    global i,w,h, f1,f2,ew,eh
    img = 0
    # La función de esto es registrar la posición inicial cuando se presiona el botón izquierdo del mouse
    if event == cv2.EVENT_LBUTTONDOWN:
        f1 = 1
        h = x
        w = y
        print("w = ", w, "h = ", h)
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
        i += 1
        f1 = f2 = 0
        crearRecorte(i)

path = "PDI\Videos\Video6.mp4"
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
                    path = "PDI\Recortes\imagen" + "%d"%cant+".jpg"
                    cv2.imwrite(path,img)
                    ret, frame = cap.read()
                stack_red, stack_green, stack_blue = registrarImagenes(frames)
                shape = (stack_red.shape[1], stack_red.shape[2])

                img_original = cv2.imread("PDI\Recortes\imagen0.jpg")

                img_ecualizada = ecualizarImagenes(stack_red, stack_green, stack_blue)
                img_media = aplicarMediaImagenes(stack_red, stack_green, stack_blue)
                img_mediana = aplicarMedianaImagenes(stack_red, stack_green, stack_blue)
                img_canny = aplicarBordesImagenes(stack_red, stack_green, stack_blue)
                img_gauss = aplicarGuasianoImagenes(stack_red, stack_green, stack_blue)
                img_high_boost = aplicarHighBoostImagenes(stack_red, stack_green, stack_blue)

                img_ecualizada_pos = ecualizarImagenes_Local(img_high_boost, 50)
                #img_ecualizada_pre = ecualizarImagenes_Local_pre(stack_red, stack_green, stack_blue, 50)

                fig = plt.figure()
                fig.suptitle("Imagenes Registradas")
                h1 = plt.subplot(1,3,1), plt.imshow(img_original), plt.title('Imagen Original')
                h2 = plt.subplot(1,3,2), plt.imshow(img_ecualizada_pos, cmap='Greys'), plt.title('Ecualizacion Local Post')
                #h3 = plt.subplot(1,3,3), plt.imshow(img_ecualizada_pre), plt.title('Ecualizacion Local Pre')
                plt.show()
            if cv2.waitKey(10) & 0xFF == ord('k'):
                print("Video reanudado")
                break
cap.release()
cv2.destroyAllWindows()