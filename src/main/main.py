
import cv2 
import numpy as np

def detect_lanes(image):
  # Pre-process the image
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (5, 5), 0)
  
  # Apply edge detection
  edges = cv2.Canny(blur, 50, 150)
  
  # Use Hough transform to detect lines
  lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
  
  # Filter out lines that are not likely to be lane lines
  lanes = []
  for line in lines:
    x1, y1, x2, y2 = line[0]
    slope = (y2 - y1) / (x2 - x1)
    if abs(slope) < 0.5:
      continue
    lanes.append(line)
  
  # Overlay the detected lanes on the original image
  output = image.copy()
  for lane in lanes:
    x1, y1, x2, y2 = lane[0]
    cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
  
  return output

# Test the lane detection function on sample images
image1 = cv2.imread("../dataset/images/f00124.png")
#image2 = cv2.imread("../dataset/images/dest3.png")
#image3= cv2.VideoCapture("D:\Proje1\lane-detection-ai\src\dataset\mideos\challenge.mp4")


lane_image1 = detect_lanes(image1)
#lane_image2 = detect_lanes(image2)


cv2.imwrite("../output/lane_image1.jpg", lane_image1)
#cv2.imwrite("../output/lane_image2.jpg", lane_image2)


cv2.imshow("result", lane_image1)
cv2.waitKey(0)
#cv2.imshow("result", lane_image2)
#cv2.waitKey(0)