import cv2
import numpy as np

def get_largest_contour(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea) if contours else None
    return largest_contour

def calculate_average_hsv(frame, contour):
    mask = np.zeros_like(frame)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_values = hsv_frame[mask[:,:,1] > 0]
    average_hsv = np.mean(hsv_values, axis=0) if hsv_values.size else None
    return average_hsv

def match_color(average_hsv, stored_hsv_values):
    differences = np.linalg.norm(stored_hsv_values - average_hsv, axis=1)
    match_index = np.argmin(differences)
    return match_index

def sort_order(arr):
    indexed_arr = [(val, idx) for idx, val in enumerate(arr)]
    sorted_arr = sorted(indexed_arr)
    sorted_indices = [pair[1] + 1 for pair in sorted_arr]
    return sorted_indices

stored_hsv_values = [[0 for _ in range(3)] for _ in range(5)]

print("Début de la capture des références. \n Appuyer sur Entrée pour capturer la première nuance.\n")

cap = cv2.VideoCapture(0)

for i in range(5):
    print("Couleur n°" + str(i+1) + "\n")
    for j in range(3):
        print("Nuance n°" + str(j+1))
        while True:
            ret, frame = cap.read()
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1)
            if key == 13:
                largest_contour = get_largest_contour(frame)
                if largest_contour is not None:
                    average_hsv = calculate_average_hsv(frame, largest_contour)
                    if average_hsv is not None:
                        stored_hsv_values[i][j] = average_hsv
                        print("Capture OK. Appuyer à nouveau sur Entrée pour capturer la nuance suivante.")
                        break
                    else:
                        print("Erreur : Aucune couleur détectée, veuillez réessayer.")
                else:
                    print("Erreur : Aucun contour détecté, veuillez réessayer.")
            elif key == ord('q'):
                break
            else:
                continue
        if key == ord('q'):
            exit()
    else:
        continue
    break

matched_indices = []

print("\nRéférences capturées. \nAppuyez sur Entrée pour capturer une couleur. \n")

i = 0
while i < 15:
    print("Couleur n°" + str(i+1))
    while True:
        ret, frame = cap.read()
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 13:
            largest_contour = get_largest_contour(frame)
            if largest_contour is not None:
                average_hsv = calculate_average_hsv(frame, largest_contour)
                if average_hsv is not None:
                    match_index = match_color(average_hsv, stored_hsv_values)
                    print("AVG", average_hsv, "STORED", stored_hsv_values)
                    print("\nINDEX : ",match_index, matched_indices)
                    if match_index not in matched_indices:
                        matched_indices.append(match_index)
                        print("Capture OK. Appuyer à nouveau sur Entrée pour capturer la couleur suivante.\n")
                        i += 1
                        break
                    else:
                        print("Erreur : Couleur déjà capturée, veuillez réessayer.")
                else:
                    print("Erreur : Aucune couleur détectée, veuillez réessayer.")
            else:
                print("Erreur : Aucun contour détecté, veuillez réessayer.")
        elif key == ord('r'):
            if matched_indices:
                matched_indices.pop()
                print("Dernière capture annulée, veuillez reprendre la capture.")
                i -= 1
                break
            else:
                print("Aucune capture à annuler.")
        elif key == ord('q'):
            exit()
        else:
            continue

cap.release()
cv2.destroyAllWindows()

print(matched_indices)
print("Ordre des couleurs observées à l'écran :")
print(sort_order(matched_indices))