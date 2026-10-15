import cv2, numpy as np, sys
img = cv2.imread(sys.argv[1])
img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
img = cv2.bilateralFilter(img, 9, 40, 9)
# per-channel line extraction: dark relative to local neighborhood
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L = lab[:,:,0].astype(np.float32)
dil = cv2.dilate(L, np.ones((5,5),np.uint8))
diff = dil - L   # darker than surroundings => line
# also color edges
edges = cv2.Canny(cv2.GaussianBlur(img,(5,5),0), 40, 110)
line = (diff > 22).astype(np.uint8)*255
line = cv2.bitwise_or(line, cv2.dilate(edges, np.ones((2,2),np.uint8)))
# clean small specks
n, lab2, stats, _ = cv2.connectedComponentsWithStats(line, 8)
keep = np.zeros_like(line)
for i in range(1,n):
    if stats[i, cv2.CC_STAT_AREA] >= 40:
        keep[lab2==i]=255
out = 255-keep
cv2.imwrite('lines.png', out)
cv2.imwrite('lines.pbm', out)
