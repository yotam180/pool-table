import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("image.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)

plt.imshow(gray, cmap="gray")
plt.title("Original Image")
plt.show()

v = np.median(gray)
sigma = 0.33
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))

print("Canny thresholds: ", lower, upper)

# What does this give us?
edges = cv2.Canny(gray, lower, upper)

contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

for contour in contours:
    # Approximate the contour
    perimeter = cv2.arcLength(contour, True)
    approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    # if len(approximation) == 4:
    cv2.drawContours(image, [approximation], -1, (0, 255, 0), 2)
    # break

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Detected Playing Surface")
plt.show()
