import cv2
import numpy as np

mask_path = 'grayscale_mask.png'
image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
h, w = image.shape
line_width = int((h + w) * 0.5 * 0.0025)

contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
contours_approx = []
polygons = []
for contour in contours:
    epsilon = 0.001 * cv2.arcLength(contour, True)
    contour_approx = cv2.approxPolyDP(contour, epsilon, True)
    contours_approx.append(contour_approx)
    polygon = contour_approx.flatten().tolist()
    polygons.append(polygon)
cv2.drawContours(image, contours_approx, -1, 128, line_width)
for polygon in polygons:
    i = 0
    while(i < len(polygon)):
        cv2.circle(image, [polygon[i], polygon[i + 1]], line_width, 255, -1)
        i += 2
cv2.imwrite('polygons.png', image)





