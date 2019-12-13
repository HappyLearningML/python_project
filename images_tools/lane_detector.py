#-*- coding:utf-8-*-
'''
本代码是基于霍夫变换来做车道线检测的，个人感觉只针对直行道路有效。对于弯道，效果不好，其原因在于没有考虑到坐标转换下的效果。但canny概念倒是值得参考，具体可参见canny.py。
在笛卡尔坐标系中，我们可以通过绘制y对x来表示y = mx + b的直线。但是，我们也可以通过绘制b对m来将此线表示为霍夫空间中的单个点。
通常，在霍夫空间中相交的曲线越多意味着由该交点表示的线对应于更多的点。对于我们的实现，我们将在霍夫空间中定义最小阈值交叉点数以检测线。
因此，霍夫变换基本上跟踪帧中每个点的霍夫空间交叉点。如果交叉点的数量超过定义的阈值，我们将识别具有相应θ和r参数的线。
'''
import cv2
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

def do_canny(frame, kernel_size=5, low_threshold=50, high_threshold=150):
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Applies a 5x5 gaussian blur with deviation of 0 to frame - not mandatory since Canny will do this for us
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    # Applies Canny edge detector with minVal of 50 and maxVal of 150
    canny = cv2.Canny(blur, low_threshold, high_threshold)
    return canny

def do_segment(frame, trap_bottom_width, trap_top_width, trap_height):
    # Since an image is a multi-directional array containing the relative intensities of each pixel in the image, we can use frame.shape to return a tuple: [number of rows, number of columns, number of channels] of the dimensions of the frame
    # frame.shape[0] give us the number of rows of pixels the frame has. Since height begins from 0 at the top, the y-coordinate of the bottom of the frame is its height
#    height = frame.shape[0]
    # Creates a triangular polygon for the mask defined by three (x, y) coordinates
#    polygons = np.array([[(0, height), (800, height), (380, 290)]])
    polygons = np.array([[\
        ((frame.shape[1] * (1 - trap_bottom_width)) // 2, frame.shape[0]), \
        ((frame.shape[1] * (1 - trap_top_width)) // 2, frame.shape[0] - frame.shape[0] * trap_height), \
        (frame.shape[1] - (frame.shape[1] * (1 - trap_top_width)) // 2, frame.shape[0] - frame.shape[0] * trap_height), \
        (frame.shape[1] - (frame.shape[1] * (1 - trap_bottom_width)) // 2, frame.shape[0])]], dtype=np.int32)
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(frame.shape) > 2:
        channel_count = frame.shape[2]
        ignore_mask_color = (255, ) * channel_count
    else:
        ignore_mask_color = 255
        

    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv2.fillPoly(mask, polygons, ignore_mask_color)
    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    segment = cv2.bitwise_and(frame, mask)
    return segment

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
	"""
	`img` is the output of the hough_lines(), An image with lines drawn on it.
	Should be a blank image (all black) with lines drawn on it.
	
	`initial_img` should be the image before any processing.
	
	The result image is computed as follows:
	
	initial_img * α + img * β + λ
	NOTE: initial_img and img must be the same shape!
	"""
	return cv2.addWeighted(initial_img, α, img, β, λ)

def filter_colors(image):
	"""
	Filter the image to include only yellow and white pixels
	"""
	# Filter white pixels
	white_threshold = 200 #130
	lower_white = np.array([white_threshold, white_threshold, white_threshold])
	upper_white = np.array([255, 255, 255])
	white_mask = cv2.inRange(image, lower_white, upper_white)
	white_image = cv2.bitwise_and(image, image, mask=white_mask)

	# Filter yellow pixels
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_yellow = np.array([90,100,100])
	upper_yellow = np.array([110,255,255])
	yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
	yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)

	# Combine the two above images
	image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)

	return image2

def calculate_lines(frame, lines):
    # Empty arrays to store the coordinates of the left and right lines
    left = []
    right = []
    # Loops through every detected line
    for line in lines:
        # Reshapes line from 2D array to 1D array
        x1, y1, x2, y2 = line.reshape(4)
        # Fits a linear polynomial to the x and y coordinates and returns a vector of coefficients which describe the slope and y-intercept
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the lane
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
    # Averages out all the values for left and right into a single slope and y-intercept value for each line
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    # Calculates the x1, y1, x2, y2 coordinates for the left and right lines
    left_line = calculate_coordinates(frame, left_avg)
    right_line = calculate_coordinates(frame, right_avg)
    return np.array([left_line, right_line])

def calculate_coordinates(frame, parameters):
    slope, intercept = parameters
    # Sets initial y-coordinate as height from top down (bottom of the frame)
    y1 = frame.shape[0]
    # Sets final y-coordinate as 150 above the bottom of the frame
    y2 = int(y1 - 150)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def visualize_lines(frame, lines):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    lines_visualize = np.zeros_like(frame)
    # Checks if any lines are detected
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # Draws lines between two coordinates with green color and 5 thickness
            cv2.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return lines_visualize

#trap_height height of the trapezoid expressed as percentage of image height
def draw_lines(img, lines, color=[255, 0, 0], thickness=10, trap_height=0.4):
	"""
	NOTE: this is the function you might want to use as a starting point once you want to 
	average/extrapolate the line segments you detect to map out the full
	extent of the lane (going from the result shown in raw-lines-example.mp4
	to that shown in P1_example.mp4).  
	
	Think about things like separating line segments by their 
	slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
	line vs. the right line.  Then, you can average the position of each of 
	the lines and extrapolate to the top and bottom of the lane.
	
	This function draws `lines` with `color` and `thickness`.	
	Lines are drawn on the image inplace (mutates the image).
	If you want to make the lines semi-transparent, think about combining
	this function with the weighted_img() function below
	"""
	# In case of error, don't draw the line(s)
	if lines is None:
		return
	if len(lines) == 0:
		return
	draw_right = True
	draw_left = True
	
	# Find slopes of all lines
	# But only care about lines where abs(slope) > slope_threshold
	slope_threshold = 0.5
	slopes = []
	new_lines = []
	for line in lines:
		x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]
		
		# Calculate slope
		if x2 - x1 == 0.:  # corner case, avoiding division by 0
			slope = 999.  # practically infinite slope
		else:
			slope = (y2 - y1) / (x2 - x1)
			
		# Filter lines based on slope
		if abs(slope) > slope_threshold:
			slopes.append(slope)
			new_lines.append(line)
		
	lines = new_lines
	
	# Split lines into right_lines and left_lines, representing the right and left lane lines
	# Right/left lane lines must have positive/negative slope, and be on the right/left half of the image
	right_lines = []
	left_lines = []
	for i, line in enumerate(lines):
		x1, y1, x2, y2 = line[0]
		img_x_center = img.shape[1] / 2  # x coordinate of center of image
		if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
			right_lines.append(line)
		elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
			left_lines.append(line)
			
	# Run linear regression to find best fit line for right and left lane lines
	# Right lane lines
	right_lines_x = []
	right_lines_y = []
	
	for line in right_lines:
		x1, y1, x2, y2 = line[0]
		
		right_lines_x.append(x1)
		right_lines_x.append(x2)
		
		right_lines_y.append(y1)
		right_lines_y.append(y2)
		
	if len(right_lines_x) > 0:
		right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)  # y = m*x + b
	else:
		right_m, right_b = 1, 1
		draw_right = False
		
	# Left lane lines
	left_lines_x = []
	left_lines_y = []
	
	for line in left_lines:
		x1, y1, x2, y2 = line[0]
		
		left_lines_x.append(x1)
		left_lines_x.append(x2)
		
		left_lines_y.append(y1)
		left_lines_y.append(y2)
		
	if len(left_lines_x) > 0:
		left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)  # y = m*x + b
	else:
		left_m, left_b = 1, 1
		draw_left = False
	
	# Find 2 end points for right and left lines, used for drawing the line
	# y = m*x + b --> x = (y - b)/m
	y1 = img.shape[0]
	y2 = img.shape[0] * (1 - trap_height)
	
	right_x1 = (y1 - right_b) / right_m
	right_x2 = (y2 - right_b) / right_m
	
	left_x1 = (y1 - left_b) / left_m
	left_x2 = (y2 - left_b) / left_m
	
	# Convert calculated end points from float to int
	y1 = int(y1)
	y2 = int(y2)
	right_x1 = int(right_x1)
	right_x2 = int(right_x2)
	left_x1 = int(left_x1)
	left_x2 = int(left_x2)
	
	# Draw the right and left lines on image
	if draw_right:
		cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
	if draw_left:
		cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)



def annotate_image_array(image_in):
    frame = filter_colors(image_in)
    
    canny = do_canny(frame, kernel_size=5, low_threshold=50, high_threshold=150)
    cv2.imshow("canny", canny)

    # Region-of-interest vertices
    # We want a trapezoid shape, with bottom edge at the bottom of the image
    # trap_bottom_width = 0.85  # width of bottom edge of trapezoid, expressed as percentage of image width
    # trap_top_width = 0.07  # ditto for top edge of trapezoid
    # trap_height = 0.4  # height of the trapezoid expressed as percentage of image height
    segment = do_segment(canny, 0.85, 0.07, 0.4)


    # Hough Transform
    #rho = 2 # distance resolution in pixels of the Hough grid
    #theta = 1 * np.pi/180 # angular resolution in radians of the Hough grid
    #threshold = 15	 # minimum number of votes (intersections in Hough grid cell)
    #min_line_length = 10 #minimum number of pixels making up a line
    #max_line_gap = 20	# maximum gap in pixels between connectable line segments
    #lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=10, maxLineGap=20)
    #lines = cv2.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)
    lines = cv2.HoughLinesP(segment, 2, 1 * np.pi/180, 15, np.array([]), minLineLength=10, maxLineGap=20)
    if 1:
        line_img = np.zeros((*segment.shape, 3), dtype=np.uint8)  # 3-channel RGB image
        draw_lines(line_img, lines)

        # Draw lane lines on the original image
        initial_image = image_in.astype('uint8')
        annotated_image = weighted_img(line_img, initial_image)
    else:

        # Averages multiple detected lines from hough into one line for left border of lane and one line for right border of lane
        lines_img = calculate_lines(image_in, lines)
        # Visualizes the lines
        lines_visualize = visualize_lines(image_in, lines_img)
        cv2.imshow("hough", lines_visualize)

        # Overlays lines on frame by taking their weighted sums and adding an arbitrary scalar value of 1 as the gamma argument
        output = cv2.addWeighted(image_in, 0.9, lines_visualize, 1, 1)

    return annotated_image


if __name__ == "__main__":
    cap = cv2.VideoCapture("F:\\developing\\this_week\\lane-detection\\advanced_lane_detection-master\\challenge_video.mp4")
    while (cap.isOpened()):
        ret, frame = cap.read()
        output = annotate_image_array(frame)
        # Opens a new window and displays the output frame
        cv2.imshow("output", output)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()