import cv2
import numpy as np

def calcul_angle(contour):
    vx, vy, x, y = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = np.arctan2(vy, vx) * 180 / np.pi
    return np.mean(angle)

image = cv2.imread('exemplebande.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # On modifie l'image en une echelle de gris

lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])
mask_blue = cv2.inRange(image, lower_blue, upper_blue)  # Filtre bleu

lower_black = np.array([0, 0, 0])
upper_black = np.array([70, 70, 70])
mask_black = cv2.inRange(image, lower_black, upper_black)   # Filtre noir

contours,_ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # On affiche le contour
    angle = calcul_angle(contour)
    print("Angle de rotation nécessaire : {:.2f} degrés".format(angle))

rows, cols, _ = image.shape
rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

cv2.imshow('Contours et rotation', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()