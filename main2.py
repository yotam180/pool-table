import cv2
import colour
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


def create_blue_mask(image_hsv):
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
    image_masked_hsv = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)
    return (mask, image_masked_hsv)


def display(image_bgr, title):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.show()


def display_color(color_hsv, title):
    z = np.zeros((100, 100, 3), dtype=np.uint8)
    # Paint it with the median color
    z[:] = color_hsv
    display_hsv(z, title)


def display_gray(image_gray, title):
    # Used to display grayscale or laplacian / gradient images
    plt.imshow(image_gray, cmap="gray")
    plt.title(title)
    plt.show()


def display_hsv(image_hsv, title):
    image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    display(image_rgb, title)


def find_largest_convex_hull(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    hull = create_convex_hull(contours)
    return hull


def paint_hull_on_image(image_bgr, hull):
    image_with_hull = image_bgr.copy()
    cv2.drawContours(image_with_hull, [hull], -1, (0, 255, 255), 5)
    return image_with_hull


def paint_lines_on_image(image_bgr, lines, color):
    image_with_lines = image_bgr.copy()
    for line in lines:
        cv2.line(image_with_lines, line[0], line[1], color, 5)
    return image_with_lines


def is_line_horizontal(line):
    line_height = abs(line[0][0][1] - line[1][0][1])
    line_width = abs(line[0][0][0] - line[1][0][0])
    if line_width < 50:
        return False

    ratio = line_height / line_width
    max_ratio = 0.1
    return ratio < max_ratio


def is_line_vertical(line):
    min_height = 50
    line_height = abs(line[0][0][1] - line[1][0][1])
    if line_height < min_height:
        return False

    return not is_line_horizontal(line)


def filter_lines(hull, pred):
    for i in range(len(hull)):
        line = hull[i], hull[(i + 1) % len(hull)]
        if pred(line):
            yield ((hull[i], hull[(i + 1) % len(hull)]))


def get_horizontal_lines(hull):
    return filter_lines(hull, is_line_horizontal)


def get_vertical_lines(hull):
    return filter_lines(hull, is_line_vertical)


def get_hull_directional_lines(hull):
    """
    Returns: [
        [vertical lines right to the center of mass],
        [vertical lines left to the center of mass],
        [horizontal_lines_above_the_center_of_mass],
        [horizontal_lines_below_the_center_of_mass],
    ]
    """

    hull_center_of_mass = np.mean(hull, axis=0)
    horizontal_lines = list(get_horizontal_lines(hull))
    vertical_lines = list(get_vertical_lines(hull))

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

    return [
        vertical_lines_right_to_the_center_of_mass,
        vertical_lines_left_to_the_center_of_mass,
        horizontal_lines_above_the_center_of_mass,
        horizontal_lines_below_the_center_of_mass,
    ]


def get_longest_line(group_of_lines):
    return max(group_of_lines, key=lambda line: np.linalg.norm(line[0] - line[1]))


def get_hull_ewns_edges(hull):
    """
    Returns one the east, west, north, and south lines of the quadrilateral hull
    """
    es, ws, ns, ss = get_hull_directional_lines(hull)
    return [
        get_longest_line(es),
        get_longest_line(ws),
        get_longest_line(ns),
        get_longest_line(ss),
    ]


def nptt(np_pt):
    """
    np point to tuple
    """
    return int(np_pt[0]), int(np_pt[1])


def np_line_to_tuples(np_line):
    return ((np_line[0][0][0], np_line[0][0][1]), (np_line[1][0][0], np_line[1][0][1]))


def np_lines_to_tuples(np_lines):
    return [np_line_to_tuples(line) for line in np_lines]


def find_intersection(line_a, line_b):
    # Each line is provided as ((x1, y1), (x2, y2))
    (x1, y1), (x2, y2) = line_a
    (x3, y3), (x4, y4) = line_b

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


def find_table_top_edges(e, w, n, s):
    """
    Returns 4 points [nw, ne, sw, se]
    """
    return [
        find_intersection(n, w),
        find_intersection(n, e),
        find_intersection(s, w),
        find_intersection(s, e),
    ]


def find_table_top_quadrilateral(hull):
    hull_e, hull_w, hull_n, hull_s = get_hull_ewns_edges(hull)
    hull_e, hull_w, hull_n, hull_s = np_lines_to_tuples(
        [hull_e, hull_w, hull_n, hull_s]
    )
    nw, ne, sw, se = find_table_top_edges(hull_e, hull_w, hull_n, hull_s)

    return np.array([nw, ne, se, sw], dtype=np.int32)


def mask_quad(image_bgr, quad):
    # Create mask which is the same shape as bgr but with 1 channel (grayscale) instead of 3
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [quad], 255)
    image_masked_bgr = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

    return mask, image_masked_bgr


def warp_image_internal(image, corners, height, width):
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


def get_tabletop_image(image_bgr, quad, height=1400, width=800):
    return warp_image_internal(image_bgr, quad, height, width)


def get_median_color(image, hull=None):
    if hull:
        # Create an empty black mask with the same dimensions as the image
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Fill the convex hull polygon with white (255)
        cv2.drawContours(mask, hull, 0, 255, -1)

        # Mask the image: only get pixels inside the convex hull
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Extract pixels inside the mask
        points = np.where(mask == 255)
        pixels = image[points[0], points[1]]

    else:
        # Just use all image pixels:
        pixels = image.reshape(-1, 3)

    # Filter out white and black pixels / pixels that are too light / too dark to be a distinct color:
    pixels = pixels[(pixels[:, 0] > 10) & (pixels[:, 1] > 10) & (pixels[:, 2] > 10)]

    # Compute the median color
    median_color = np.median(pixels, axis=0)

    return median_color


def get_k_significant_colors_in_contour(image_hsv, contour, k):
    """
    Performs k-means clustering on the colors of the pixels inside the contour
    and returns the k most significant colors, together with the % of pixels for each color
    """

    # Extract the pixels inside the contour
    mask = np.zeros(image_hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    points = np.where(mask == 255)
    pixels = image_hsv[points[0], points[1]]

    # Perform k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        np.float32(pixels), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)

    # Count the number of pixels for each cluster
    unique, counts = np.unique(labels, return_counts=True)

    # Compute the % of pixels for each cluster
    percentages = counts / len(pixels)

    # Sort the clusters by the number of pixels
    sorted_indices = np.argsort(counts)[::-1]
    sorted_centers = centers[sorted_indices]
    sorted_percentages = percentages[sorted_indices]

    return sorted_centers, sorted_percentages


def display_colors_hsv(colors):
    """
    For N given colors, returns an image N*100 pixels width, 100 pixels height
    where each color is displayed in a 100x100 square
    """

    # Create an empty image
    image = np.zeros((100, 100 * len(colors), 3), dtype=np.uint8)

    # Paint each color in a 100x100 square
    for i, color in enumerate(colors):
        image[:, i * 100 : (i + 1) * 100] = color

    # Display the image
    display_hsv(image, "Colors")


def cut_out_contour(image, contour):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image


def find_initial_ball_mask_by_color_negation(tabletop_hsv):
    median_color = get_median_color(tabletop_hsv)

    tolerance = 10
    lower_bound = (median_color[0] - tolerance, 100, 100)
    upper_bound = (median_color[0] + tolerance, 255, 255)

    tabletop_hsv_blurred = cv2.GaussianBlur(tabletop_hsv, (15, 15), 0)

    mask = cv2.inRange(tabletop_hsv_blurred, lower_bound, upper_bound)
    mask = cv2.bitwise_not(mask)
    kernel = np.ones((15, 15), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=1)
    # Commented out since it's not very helpful
    # mask = cv2.dilate(mask, kernel, iterations=1)

    tabletop_masked_bgr = cv2.bitwise_and(tabletop_hsv, tabletop_hsv, mask=mask)
    return mask, tabletop_masked_bgr


def find_initial_ball_mask_by_laplace(tabletop_hsv):
    # Apply Laplacian filter to the image
    laplacian = cv2.Laplacian(tabletop_hsv, cv2.CV_64F)

    # Convert the result to 8-bit grayscale
    laplacian = cv2.convertScaleAbs(laplacian)

    gray = cv2.cvtColor(laplacian, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to the image
    # _, mask = cv2.threshold(gray, 50, 255, 50) # What is the last 50???
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    # Blur mask
    # mask = cv2.GaussianBlur(mask, (15, 15), 0)

    # Dilate
    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations=1)

    # # Threshold again
    # _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    # print("A")

    # # Find external contours in image:
    # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     area_percent = area / (mask.shape[0] * mask.shape[1])
    #     if 0.0001 < area_percent < 0.1:
    #         print("Drawing", area_percent)
    #         cv2.drawContours(mask, [contour], -1, 255, -1)

    display_gray(mask, "Laplacian mask")

    return mask, None


def get_pot_positions(tabletop_bgr):
    board_height, board_width = tabletop_bgr.shape[:2]
    top_left = (10, 10)
    top_right = (board_width - 10, 10)
    bottom_left = (10, board_height - 10)
    bottom_right = (board_width - 10, board_height - 10)
    mid_left = (10, board_height // 2 + 10)
    mid_right = (board_width - 10, board_height // 2 + 10)

    return [
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        mid_left,
        mid_right,
    ]


def get_ball_contours_without_pot_contours(initial_ball_mask):
    pot_positions = get_pot_positions(initial_ball_mask)

    def is_contour_a_pot(contour, pots):
        # Check if the contour is close to any of the pots
        for pot in pots:
            if cv2.pointPolygonTest(contour, pot, False) > 0:
                return True

        return False

    image_area = initial_ball_mask.shape[0] * initial_ball_mask.shape[1]
    min_ball_area = image_area * 0.001

    contours, _ = cv2.findContours(
        initial_ball_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        if is_contour_a_pot(contour, pot_positions):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h / w > 2.5 or w / h > 2.5:
            continue

        area = cv2.contourArea(contour)
        if area < min_ball_area:
            continue

        yield contour


def mask_ball_contours(tabletop_image, ball_contours):
    mask = np.zeros(tabletop_image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, ball_contours, -1, 255, -1)
    masked_image = cv2.bitwise_and(tabletop_image, tabletop_image, mask=mask)
    return mask, masked_image


class TableTop(object):
    def __init__(self, original_bgr):
        self.original_bgr = original_bgr
        self.original_hsv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)

        self.blue_mask, self.blue_masked_hsv = create_blue_mask(self.original_hsv)
        self.blue_masked_bgr = cv2.cvtColor(self.blue_masked_hsv, cv2.COLOR_HSV2BGR)

        self.tabletop_hull = find_largest_convex_hull(self.blue_masked_bgr)
        self.tabletop_quad = find_table_top_quadrilateral(self.tabletop_hull)

        self.tabletop_bgr = get_tabletop_image(self.original_bgr, self.tabletop_quad)
        self.tabletop_hsv = cv2.cvtColor(self.tabletop_bgr, cv2.COLOR_BGR2HSV)

        self.tabletop_median_hsv = get_median_color(self.tabletop_hsv)

        self.ball_contours = list(self.find_ball_contours_by_color_negation())

        _, self.masked_balls_tabletop_hsv = mask_ball_contours(
            self.tabletop_hsv, self.ball_contours
        )

    def find_ball_contours_by_color_negation(self):
        initial_mask, tabletop_masked_hsv = find_initial_ball_mask_by_color_negation(
            self.tabletop_hsv
        )
        ball_contours = list(get_ball_contours_without_pot_contours(initial_mask))

        self.tabletop_masked_hsv = tabletop_masked_hsv

        return ball_contours

    def get_single_ball_contour(self):
        # We use this for color calibration
        # Currently just return largest ball contour:

        try:
            return max(self.ball_contours, key=cv2.contourArea)
        except:
            print("[-] No ball found in tabletop image")
            return None

    def get_single_ball_color(self):
        # Return median color of the single ball contour (for calibration)
        ball_contour = self.get_single_ball_contour()
        return get_median_color(self.tabletop_hsv, [ball_contour])


def color_distance(color_a, color_b):
    # First convert HSV to lab:
    color_a = cv2.cvtColor(np.float32([[color_a]]), cv2.COLOR_HSV2RGB)
    color_a = cv2.cvtColor(color_a, cv2.COLOR_RGB2Lab)

    color_b = cv2.cvtColor(np.float32([[color_b]]), cv2.COLOR_HSV2RGB)
    color_b = cv2.cvtColor(color_b, cv2.COLOR_RGB2Lab)

    return colour.delta_E(color_a, color_b)


def get_colors_range_for_ball_calibration(image_bgr):
    num_colors = 5
    tt = TableTop(image_bgr)

    ball_contour = tt.get_single_ball_contour()
    if ball_contour is None:
        return []

    colors, percentages = get_k_significant_colors_in_contour(
        tt.tabletop_hsv, ball_contour, num_colors
    )

    results = []

    for color, percentage in zip(colors, percentages):
        # Get distance of color from tt.tabletop_median_hsv:
        distance = color_distance(color, tt.tabletop_median_hsv)
        strength = color[1] / 255
        color_score = strength * distance

        # "Weak colors"?
        if color[1] < 80 or (color[1] < 100 and color[2] < 100):
            continue

        if percentage > 0.2:
            results.append((color, color_score))

    return results


def get_colors_for_calibration(images):
    colors = []
    for image in images:
        colors += get_colors_range_for_ball_calibration(image)

    colors = [x[0] for x in sorted(colors, key=lambda x: x[1], reverse=True)]
    return colors  # [: min(len(colors) // 3, 4)]


def create_color_mask(colors, image_hsv):
    # For each color, create upper and lower bounds
    # Then, create separate masks for each color for the image
    # and combine them with bitwise or

    masks = []
    for color in colors:
        lower_bound = np.array([color[0] - 10, color[1] - 40, color[2] - 40])
        upper_bound = np.array([color[0] + 10, color[1] + 40, color[2] + 40])
        mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
        masks.append(mask)

    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    masked_image_hsv = cv2.bitwise_and(image_hsv, image_hsv, mask=combined_mask)

    return combined_mask, masked_image_hsv


if __name__ == "__main__":
    # image_bgr = cv2.imread("calibration/daylight/red/1.jpg")
    # process_frame(image_bgr)

    images = []
    for i in range(1, 6):
        images.append(cv2.imread(f"calibration/daylight/purple/{i}.jpg"))

    purples = get_colors_for_calibration(images)

    image_bgr = cv2.imread("data/1.jpg")
    tt = TableTop(image_bgr)
    # display_hsv(tt.tabletop_masked_hsv, "Tabletop Masked HSV")

    mask, masked = find_initial_ball_mask_by_laplace(tt.tabletop_hsv)
    display_hsv(masked)

    # _, masked_image_hsv = create_color_mask(purples, tt.tabletop_masked_hsv)
    # display_hsv(masked_image_hsv, "Masked tabletop HSV")
