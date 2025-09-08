import cv2
import numpy as np
reference = cv2.imread('train.png')
test = cv2.imread('test.png')
test = cv2.resize(test, (reference.shape[1], reference.shape[0]))
diff = cv2.absdiff(reference, test)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, missing = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(missing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = test.copy()
cv2.drawContours(output, contours, -1, (0, 0, 255), 2)
cv2.imshow('Missing Parts Highlighted', output)
cv2.waitKey(0)
cv2.destroyAllWindows()