import cv2
import numpy as np
from matplotlib import pyplot as plt


def create_convex_hull(contours):
    """
    Creates a convex hull that covers all the provided contours.

    :param contours: A list of contours to cover with the convex hull.
    :return: The points of the convex hull covering all the contours.
    """
    # Combine all the contours into one array
    all_points = np.vstack(contours)

    # Compute the convex hull for the combined points
    convex_hull = cv2.convexHull(all_points)

    return convex_hull


quad = None


def process_frame(image):
    global quad

    original_image = image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for blue color and create mask
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Use mask to isolate blue table
    blue_only = cv2.bitwise_and(image, image, mask=mask)

    # plt.imshow(cv2.cvtColor(blue_only, cv2.COLOR_BGR2RGB))
    # plt.title("Blue Table")
    # plt.show()

    # Convert to gray scale for edge detection
    gray = cv2.cvtColor(blue_only, cv2.COLOR_BGR2GRAY)

    image = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    # plt.imshow()
    # plt.title("Gray Scale Image")
    # plt.show()

    # # Edge detection
    # v = np.median(gray)
    # sigma = 0.33
    # lower = int(max(0, (1.0 - sigma) * v))
    # upper = int(min(255, (1.0 + sigma) * v))
    # edges = cv2.Canny(gray, lower, upper)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # One approach
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[
        :1
    ]  # Adjust number as needed

    # # Draw contours
    # for contour in contours:
    #     # perimeter = cv2.arcLength(contour, True)
    #     # approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    #     cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    # Another approach
    hull = create_convex_hull(contours)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.001 * peri, True)

    hull_center_of_mass = np.mean(hull, axis=0)

    def get_vertical_lines(hull):
        # Returns line that are (generally) vertical, computed by their angle and not absolute difference
        vertical_lines = []
        # for i in range(len(hull)):
        #     ang = np.arctan2(
        #         hull[i][0][1] - hull[(i + 1) % len(hull)][0][1],
        #         hull[i][0][0] - hull[(i + 1) % len(hull)][0][0],
        #     )
        #     if abs(ang) < np.pi / 4:
        #         vertical_lines.append((hull[i], hull[(i + 1) % len(hull)]))

        # return vertical_lines

        # Return all lines that are not horizontal:
        for i in range(len(hull)):
            if abs(hull[i][0][0] - hull[(i + 1) % len(hull)][0][0]) > 10:
                vertical_lines.append((hull[i], hull[(i + 1) % len(hull)]))
        return vertical_lines

    def get_horizontal_lines(hull):
        # Returns line that are (generally) horizontal
        horizontal_lines = []
        for i in range(len(hull)):
            line_height = abs(hull[i][0][1] - hull[(i + 1) % len(hull)][0][1])
            line_width = abs(hull[i][0][0] - hull[(i + 1) % len(hull)][0][0])
            ratio = line_height / line_width
            max_ratio = 0.1
            min_width = 50
            if ratio < max_ratio and line_width > min_width:
                horizontal_lines.append((hull[i], hull[(i + 1) % len(hull)]))
        return horizontal_lines

    vertical_lines = get_vertical_lines(hull)
    horizontal_lines = get_horizontal_lines(hull)

    for line in vertical_lines:
        cv2.line(image, tuple(line[0][0]), tuple(line[1][0]), (0, 255, 0), 4)

    for line in horizontal_lines:
        cv2.line(image, tuple(line[0][0]), tuple(line[1][0]), (0, 0, 255), 4)

    vertical_lines_right_to_the_center_of_mass = [
        line
        for line in vertical_lines
        if line[0][0][0] > hull_center_of_mass[0][0]
        and line[1][0][0] > hull_center_of_mass[0][0]
    ]

    vertical_lines_left_to_the_center_of_mass = [
        line
        for line in vertical_lines
        if line[0][0][0] < hull_center_of_mass[0][0]
        and line[1][0][0] < hull_center_of_mass[0][0]
    ]

    horizontal_lines_above_the_center_of_mass = [
        line
        for line in horizontal_lines
        if line[0][0][1] < hull_center_of_mass[0][1]
        and line[1][0][1] < hull_center_of_mass[0][1]
    ]

    horizontal_lines_below_the_center_of_mass = [
        line
        for line in horizontal_lines
        if line[0][0][1] > hull_center_of_mass[0][1]
        and line[1][0][1] > hull_center_of_mass[0][1]
    ]

    def get_longest_line(group_of_lines):
        return max(group_of_lines, key=lambda line: np.linalg.norm(line[0] - line[1]))

    # # Draw each group of lines in a different color
    # for line in vertical_lines_right_to_the_center_of_mass:
    #     cv2.line(image, tuple(line[0][0]), tuple(line[1][0]), (0, 255, 0), 2)

    # for line in vertical_lines_left_to_the_center_of_mass:
    #     cv2.line(image, tuple(line[0][0]), tuple(line[1][0]), (0, 0, 255), 2)

    # for line in horizontal_lines_above_the_center_of_mass:
    #     cv2.line(image, tuple(line[0][0]), tuple(line[1][0]), (255, 0, 0), 2)

    # for line in horizontal_lines_below_the_center_of_mass:
    #     cv2.line(image, tuple(line[0][0]), tuple(line[1][0]), (255, 255, 0), 2)

    top_line = get_longest_line(horizontal_lines_above_the_center_of_mass)
    bottom_line = get_longest_line(horizontal_lines_below_the_center_of_mass)
    left_line = get_longest_line(vertical_lines_left_to_the_center_of_mass)
    right_line = get_longest_line(vertical_lines_right_to_the_center_of_mass)

    # # Draw each line in a different color
    cv2.line(image, tuple(top_line[0][0]), tuple(top_line[1][0]), (0, 255, 0), 5)
    cv2.line(image, tuple(bottom_line[0][0]), tuple(bottom_line[1][0]), (0, 0, 255), 5)
    cv2.line(image, tuple(left_line[0][0]), tuple(left_line[1][0]), (255, 0, 0), 5)
    cv2.line(image, tuple(right_line[0][0]), tuple(right_line[1][0]), (255, 255, 0), 5)

    def find_intersection(line_a, line_b):
        # Each line is provided as ((x1, y1), (x2, y2))
        ((x1, y1),), ((x2, y2),) = line_a
        ((x3, y3),), ((x4, y4),) = line_b

        # Compute the coefficients of the line equations
        a1 = y2 - y1
        b1 = x1 - x2
        c1 = a1 * x1 + b1 * y1

        a2 = y4 - y3
        b2 = x3 - x4
        c2 = a2 * x3 + b2 * y3

        # Assemble the coefficient matrix and the constant vector
        A = np.array([[a1, b1], [a2, b2]])
        C = np.array([c1, c2])

        # Solve the system of equations
        try:
            intersection = np.linalg.solve(A, C)
            return intersection
        except np.linalg.LinAlgError:
            # If the lines are parallel or identical, there might be no solution or infinitely many solutions
            return None

    top_left = find_intersection(top_line, left_line)
    top_right = find_intersection(top_line, right_line)
    bottom_left = find_intersection(bottom_line, left_line)
    bottom_right = find_intersection(bottom_line, right_line)

    # Draw the 4 points:
    cv2.circle(image, tuple(top_left.astype(int)), 5, (255, 255, 255), -1)
    cv2.circle(image, tuple(top_right.astype(int)), 5, (255, 255, 255), -1)
    cv2.circle(image, tuple(bottom_left.astype(int)), 5, (255, 255, 255), -1)
    cv2.circle(image, tuple(bottom_right.astype(int)), 5, (255, 255, 255), -1)

    if quad is None:
        quad = np.array(
            [top_left, top_right, bottom_right, bottom_left], dtype=np.int32
        )
        print("QUAD", quad)

    cv2.polylines(image, [quad], True, (0, 255, 255), 2)

    # Cut the quad from the original image and show only it:
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [quad], 255)
    quad_mask = mask
    cutout = cv2.bitwise_and(original_image, original_image, mask=mask)
    # plt.imshow(cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB))
    # plt.title("Detected Playing Surface")
    # plt.show()

    def warpImage(image, corners, height, width):
        # Define the target coordinates for the perspective transform
        target = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
        )

        # Convert corners to a NumPy float32 array
        corners = np.array(corners, dtype=np.float32)

        # Calculate the perspective transformation matrix
        mat = cv2.getPerspectiveTransform(corners, target)

        # Apply the perspective warp transformation
        out = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_CUBIC)

        return out

    # board = warpImage(original_image, quad, 1400, 800)
    # plt.imshow(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))
    # plt.title("Board from top")
    # plt.show()

    # Apply laplacian filter to detect edges of balls:
    # gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    # laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # laplacian = np.uint8(np.absolute(laplacian))

    # # Contours of the laplacian image
    # contours, _ = cv2.findContours(laplacian, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # # Draw them on the original image
    # for contour in contours:
    #     cv2.drawContours(board, [contour], -1, (0, 255, 0), 2)

    # plt.imshow(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))
    # plt.title("Detected Balls")
    # plt.show()

    def get_median_color(image, hull):
        # Create an empty black mask with the same dimensions as the image
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Fill the convex hull polygon with white (255)
        cv2.drawContours(mask, [hull], 0, 255, -1)

        # Mask the image: only get pixels inside the convex hull
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Extract pixels inside the mask
        points = np.where(mask == 255)
        pixels = image[points[0], points[1]]

        # Compute the median color
        median_color = np.median(pixels, axis=0)

        return median_color

    board_height = 1400
    board_width = 800

    warped = warpImage(original_image, quad, board_height, board_width)
    warped_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    # Approach A: Remove the blue median color...
    median_color = get_median_color(warped_hsv, hull)

    # Perform gradients on warped to detect ball edges?
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))

    # Erode and Dilate the laplacian image to make the edges more visible
    # kernel = np.ones((5, 5), np.uint8)
    # big_kernel = np.ones((20, 20), np.uint8)
    # laplacian = cv2.erode(laplacian, kernel, iterations=1)
    # laplacian = cv2.dilate(laplacian, big_kernel, iterations=1)

    plt.imshow(laplacian, cmap="gray")
    plt.title("Laplacian")
    plt.show()

    tolerance = 10
    lower_bound = (median_color[0] - tolerance, 100, 150)
    upper_bound = (median_color[0] + tolerance, 255, 255)

    mask = cv2.inRange(warped_hsv, lower_bound, upper_bound)
    mask = cv2.bitwise_not(mask)
    # mask = cv2.bitwise_and(mask, quad_mask)
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    warped_masked = cv2.bitwise_and(warped, warped, mask=mask)

    # Increase saturation on warped:
    # (s, h, v) = cv2.split(warped_hsv)
    # s = s * 5// 3
    # s = np.clip(s,0,255)
    # warped_hsv = cv2.merge([s, h,v])
    # warped = cv2.cvtColor(warped_hsv, cv2.COLOR_HSV2BGR)

    top_left = (10, 10)
    top_right = (board_width - 10, 10)
    bottom_left = (10, board_height - 10)
    bottom_right = (board_width - 10, board_height - 10)
    mid_left = (10, board_height // 2 + 10)
    mid_right = (board_width - 10, board_height // 2 + 10)

    hole_positions = [
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        mid_left,
        mid_right,
    ]

    # Draw all circles for top/bottom and left/right
    cv2.circle(warped_masked, top_left, 10, (0, 255, 255), -1)
    cv2.circle(warped_masked, top_right, 10, (0, 255, 255), -1)
    cv2.circle(warped_masked, bottom_left, 10, (0, 255, 255), -1)
    cv2.circle(warped_masked, bottom_right, 10, (0, 255, 255), -1)
    cv2.circle(warped_masked, mid_left, 10, (0, 255, 255), -1)
    cv2.circle(warped_masked, mid_right, 10, (0, 255, 255), -1)

    def is_contour_a_pot(contour, holes):
        # Check if the contour is close to any of the holes
        for hole in holes:
            if cv2.pointPolygonTest(contour, hole, False) > 0:
                return True

        return False

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(warped_masked, contours, -1, (255, 255, 255), -1)
    return warped_masked

    relevant_rects = []
    for contour in contours:
        if is_contour_a_pot(contour, hole_positions):
            continue

        for hole_position in hole_positions:
            cv2.circle(warped, hole_position, 50, (20, 20, 20), -1)

        # Check for the contour if it's around a circle shape
        # That means height to width ratio should be close to 1:
        x, y, w, h = cv2.boundingRect(contour)
        if h / w > 2.5 or w / h > 2.5:
            continue

        # It should have some area (remove small contours):
        if cv2.contourArea(contour) < 1000:
            continue

        relevant_rects.append((x, y, w, h))

    arr = np.array([])
    lenghts = []
    for rect in relevant_rects:
        x, y, w, h = rect
        pixels = warped[y : y + h, x : x + w]
        pixels = pixels.reshape((-1, 3))
        arr = np.append(arr, pixels)
        lenghts.append(len(pixels))

    arr = arr.reshape((-1, 3))

    # do KNN on the array
    img_reshape = np.float32(arr)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        img_reshape, 20, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]

    def find_closest_color_label(color_bgr, color_labels):
        """
        Find the closest color label for a given BGR color.

        Parameters:
        - color_bgr: A tuple representing the BGR color (e.g., (255, 0, 0) for blue).
        - color_labels: A dictionary where the keys are color names and the values are BGR tuples.

        Returns:
        - An rgb value for the closest color label.
        """

        def euclidean_distance(color1, color2):
            # Calculate the Euclidean distance between two colors
            return np.sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2)))

        # Initialize minimum distance and closest color
        min_distance = float("inf")
        closest_color = None

        # Check each defined color label
        for label, label_color in color_labels.items():
            distance = euclidean_distance(color_bgr, label_color)
            if distance < min_distance:
                min_distance = distance
                closest_color = label

        return color_labels[closest_color]

    # Define the color labels with their respective BGR values
    # Note: BGR values may need to be calibrated to your specific case
    color_labels = {
        "yellow": (0, 255, 255),
        "dark blue": (255, 0, 0),
        "red": (0, 0, 255),
        "purple": (255, 0, 255),
        "orange": (0, 165, 255),
        "green": (0, 128, 0),
        "dark red": (0, 0, 139),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "light blue": (255, 229, 204),
    }

    # for color in centers:
    #     print(color)
    #     closest_color = find_closest_color_label(color, color_labels)
    #     # Replace the color in segmented_image with closest_color
    #     mask = (segmented_image == color).all(axis=1)
    #     segmented_image[mask] = closest_color

    # Cool magic!
    arr = segmented_image

    colors = {}
    # Loop over each color and print how many pixels in arr are of that color:
    for color in centers:
        c = tuple(int(u) for u in color)
        c_in_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
        norm = np.linalg.norm(c_in_hsv - median_color)
        if norm < 50:
            continue

        # Check if c_in_hsv is either black or white:
        if c_in_hsv[2] < 50 or c_in_hsv[1] < 50:
            continue

        c = cv2.cvtColor(np.uint8([[c_in_hsv]]), cv2.COLOR_HSV2BGR)[0][0]
        c = tuple(int(u) for u in c)

        mask = (arr == color).all(axis=1)
        print(f"Color {color} has {len(arr[mask])} pixels")
        colors[c] = len(arr[mask])

        z = np.zeros_like(warped)
        z[:] = color
        # plt.imshow(cv2.cvtColor(z, cv2.COLOR_BGR2RGB))
        # plt.title(f"Color {color}, pixels {len(arr[mask])}")
        # plt.show()

    print(colors)
    sum = np.array([0, 0, 0])
    count = 0
    for color, pixels in colors.items():
        count += pixels
        sum += np.array(color) * pixels
    sum = sum / count

    z = np.zeros_like(warped)
    z[:] = sum
    return z

    final_rect_contents = []
    for length, rect in zip(lenghts, relevant_rects):
        pixels_1d = arr[:length]
        x, y, w, h = rect
        pixels_reshaped_back = pixels_1d.reshape((h, w, 3))
        final_rect_contents.append(pixels_reshaped_back)
        arr = arr[length:]

    for rect, content in zip(relevant_rects, final_rect_contents):
        x, y, w, h = rect
        warped[y : y + h, x : x + w] = content

    threshold = 0

    def create_mask_for_color(image, color, threshold):
        lower_bound = np.array([max(0, c - threshold) for c in color])
        upper_bound = np.array([min(255, c + threshold) for c in color])
        mask = cv2.inRange(image, lower_bound, upper_bound)
        return mask

    # Loop over each color, detect contours, and draw them
    img = np.zeros_like(warped)
    # img = warped.copy()
    for color in centers:
        mask = create_mask_for_color(warped, color, threshold)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = tuple(int(u) for u in color)
        c_in_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
        print(c_in_hsv)

        # Check if c_in_hsv is close to the median color
        norm = np.linalg.norm(c_in_hsv - median_color)
        print("Norm", norm)
        if norm < 80:
            continue

        # Check if c_in_hsv is either black or white:
        if c_in_hsv[2] < 50 or c_in_hsv[1] < 50:
            continue

        c = cv2.cvtColor(np.uint8([[c_in_hsv]]), cv2.COLOR_HSV2BGR)[0][0]
        c = tuple(int(u) for u in c)

        for contour in contours:
            cv2.drawContours(img, [contour], -1, c, -1)

    return img

    # # Morphological closing and distance transform
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # kernel = np.ones((20, 20), np.uint8)
    # gray = cv2.erode(gray, kernel, iterations=1)
    # # gray = cv2.dilate(gray, kernel, iterations=1)
    # gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # gray = cv2.distanceTransform(gray, cv2.DIST_L2, 5)
    # gray = cv2.normalize(gray, gray, 0, 1.0, cv2.NORM_MINMAX)
    # img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return img


VIDEO = False

if VIDEO:
    print("start")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    print("cap")
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        try:
            endgame = process_frame(frame)
        except:
            import traceback

            traceback.print_exc()

            continue

        # Resize endgame to 50%
        endgame = cv2.resize(
            endgame, (int(endgame.shape[1] / 2), int(endgame.shape[0] / 2))
        )

        # Display the resulting frame
        cv2.imshow("frame", endgame)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

else:

    image = cv2.imread("calibration/daylight/blue/1.jpg")
    warped = process_frame(image)
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
    plt.title("Detected Playing Surface")
    plt.show()
