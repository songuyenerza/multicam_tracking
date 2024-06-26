import cv2

img_path = '/data/datasets/thainq/sonnt373/dev/Multi_cam_tracking/multicam_tracking/IMAGE_OVERVIEW.jpg'
image = cv2.imread(img_path)
print("image: ", image.shape)
W = image.shape[1]
H = image.shape[0]

image = cv2.resize(image, (W//2, H//2))
name = img_path.split('/')[-1].split('.')[0]
list_point = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at ({x}, {y})")
        cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)
        cv2.imshow('Image', image)
        list_point.append(str(round(x/image.shape[1], 3)))
        list_point.append(str(round(y/image.shape[0], 3)))

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

cv2.imshow('Image', image)
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        full_roi = str(';'.join(list_point))
        print(full_roi)
        print(name)
        f = open(name + '.txt', 'w')
        f.write(full_roi)
        f.close()
        break

cv2.destroyAllWindows()