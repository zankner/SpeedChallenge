import cv2

dashcam_feed = cv2.VideoCapture('../data/raw/train.mp4')
frame_count = 0
success = True

while success:
    if frame_count % 500 == 0:
        print(frame_count)
    success, image = dashcam_feed.read()
    cv2.imwrite(f'../data/raw/frames/frame-{frame_count}.jpg', image)
    frame_count += 1
