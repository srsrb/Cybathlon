import sys
import time
import cv2
import numpy as np
import pyttsx3

def preprocess(frame):
    blurred = cv2.GaussianBlur(frame, (21,21), 0)
    return blurred

def normalize_intensity(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = ((image - min_val) / (max_val - min_val)) * 255
    return normalized_image.astype(np.uint8)

def gamma_correction(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def getBlue(frame):

    # Convertir l'image en espace de couleur HSV
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Sélectionner une couleur sur l'image (par exemple, en cliquant dessus)
    # Remplacer les coordonnées (x, y) par les coordonnées du pixel sélectionné
    y, x, _ = frame.shape 
    x=int(x/2)
    y=int(y/2)
    print("x =", x, "y =", y)
    selected_color = hsv_image[y, x]
    print(selected_color)

    # Afficher les valeurs de couleur sélectionnées
    # Définir les bornes pour créer le masque
    tolerance = 15  # Tolérance pour la détection
    lower_bound = np.array([selected_color[0] - tolerance, selected_color[1] - tolerance, selected_color[2] - tolerance])
    upper_bound = np.array([selected_color[0] + tolerance, selected_color[1] + tolerance, selected_color[2] + tolerance])
    return lower_bound, upper_bound

def detect_mat(frame):
    preprocessed_frame = preprocess(frame)

    lower_blue = np.array([80, 0, 0])
    upper_blue = np.array([255, 100, 100])

    mask = cv2.inRange(preprocessed_frame, lower_blue, upper_blue)
    
    kernel = np.ones((9, 9), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_mask = np.zeros_like(mask)

    area_size = 10000
    max_contour = None
    max_contour_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > area_size:

            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            if max_contour is None:
                cv2.drawContours(contour_mask, [contour], -1, (255), -1)
                max_contour = contour
                max_contour_area = area
            else:
                max_cX = int(cv2.moments(max_contour)["m10"] / cv2.moments(max_contour)["m00"])
                max_cY = int(cv2.moments(max_contour)["m01"] / cv2.moments(max_contour)["m00"])
                distance_to_max = np.sqrt((cX - max_cX) ** 2 + (cY - max_cY) ** 2)

                if distance_to_max < 1000:
                    max_contour_area += area
                    cv2.drawContours(contour_mask, [contour], -1, (255), -1)

    if max_contour is None:
        return None
    
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    poly = cv2.approxPolyDP(max_contour, epsilon, True)
    x, y, w, h = cv2.boundingRect(poly)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 6)
    cv2.polylines(frame, [poly], isClosed=True, color=(255, 0, 0), thickness=8)

    cv2.imshow("Contours detected", contour_mask)
    
    return x, y, w, h

def merge_close_rectangles(rectangles, threshold):
    merged_rects = []

    for rect in rectangles:
        x, y, w, h = rect
        rect_center = (x + w // 2, y + h // 2)

        merged = False
        for merged_rect in merged_rects:
            merged_x, merged_y, merged_w, merged_h = merged_rect
            merged_center = (merged_x + merged_w // 2, merged_y + merged_h // 2)

            distance = np.sqrt((rect_center[0] - merged_center[0])**2 + (rect_center[1] - merged_center[1])**2)

            if distance < threshold:
                merged_x = min(x, merged_x)
                merged_y = min(y, merged_y)
                merged_w = max(x + w, merged_x + merged_w) - merged_x
                merged_h = max(y + h, merged_y + merged_h) - merged_y
                merged_rect = (merged_x, merged_y, merged_w, merged_h)
                merged = True
                break

        if not merged:
            merged_rects.append(rect)

    return merged_rects

def calculate_roi_from_lines(lines, width, height):
    min_x, min_y = width, height
    max_x, max_y = 0, 0

    for line in lines:
        (x1, y1), (x2, y2) = line
        min_x = min(min_x, x1, x2)
        min_y = min(min_y, y1, y2)
        max_x = max(max_x, x1, x2)
        max_y = max(max_y, y1, y2)

    roi_margin = 5
    roi_x = max(0, min_x - roi_margin)
    roi_y = max(0, min_y - roi_margin)
    roi_w = min(width, max_x + roi_margin) - roi_x
    roi_h = min(height, max_y + roi_margin) - roi_y

    return roi_x, roi_y, roi_w, roi_h


def calculate_weighted_centroid(contours):
    if contours:
        x, y, w, h = contours
        cX = x + (w // 2)
        cY = y + (h // 2)
        return (cX, cY)
    else:
        return None

def adjust_position(frame, weighted_centroid):
    global engine
    height, width, _ = frame.shape
    center_x = width // 2

    if engine is not None:
        if weighted_centroid:
            centroid_x, _ = weighted_centroid
            deviation = centroid_x - center_x
            if deviation > 10:
                engine.say("Shift right")
                engine.runAndWait()
                print("Décalage vers la droite")
            elif deviation < -10:
                engine.say("Shift left")
                engine.runAndWait()
                print("Décalage vers la gauche")
            else:
                engine.say("In the center")
                engine.runAndWait()
                print("Au centre")
        else:
            print("Pas de matelas détecté")
            engine.say("No mattress detected")
            engine.runAndWait()
    else:
        if weighted_centroid:
            centroid_x, _ = weighted_centroid
            deviation = centroid_x - center_x
            if deviation > 10:
                print("Décalage vers la droite")
            elif deviation < -10:
                print("Décalage vers la gauche")
            else:
                print("Au centre")
        else:
            print("Pas de matelas détecté")

def live_capture():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width, _ = frame.shape
        roi = frame[int(height * 0.7):, :]
        
        roi_contours = detect_mat(roi)
        
        weighted_centroid = calculate_weighted_centroid(roi_contours)
        
        adjust_position(frame, weighted_centroid)
            
        cv2.imshow('Live Capture', frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def video(filename):
    cap = cv2.VideoCapture(filename)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width, _ = frame.shape
        roi = frame[int(height * 0.7):, :]
        
        roi_contours = detect_mat(roi)
        
        weighted_centroid = calculate_weighted_centroid(roi_contours)
        
        adjust_position(frame, weighted_centroid)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def image(filename):
    frame = cv2.imread(filename)
    
    height, width, _ = frame.shape
    roi = frame[int(height * 0.7):, :]
    
    roi_contours = detect_mat(roi)
        
    weighted_centroid = calculate_weighted_centroid(roi_contours)
    
    adjust_position(frame, weighted_centroid)

    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', 800, 600)
    cv2.imshow('Frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    global engine

    engine = None

    if len(sys.argv) == 1:
        # engine = pyttsx3.init()
        live_capture()
        exit()
    if len(sys.argv) != 3:
        print("Usage: python replacement.py <empty> (for live), 1 (for image) or 2 (for video) <empty>, <image_file_name> or <video_file_name>")
        exit()

    filename = sys.argv[2]

    if sys.argv[1] == "1":
        image(filename)
    elif sys.argv[1] == "2":
        video(filename)
    else:
        print("Usage: python replacement.py <empty> (for live), 1 (for image) or 2 (for video) <empty>, <image_file_name> or <video_file_name>")
        exit()

if __name__ == "__main__":
    main()
