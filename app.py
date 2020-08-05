import cv2
import datetime
from datetime import date
import numpy as np
from imutils.video import VideoStream  # from requirements.txt

from utils import detector_utils, cv_frame


def object_detection():
    detection_graph, sess = detector_utils.load_inference_graph()

    # Detection confidence threshold to draw bounding box
    score_thresh = 0.80

    # max number of objects we want to detect/ track in a single frame
    num_objects_detect = 50

    im_height, im_width = (None, None)

    # vs = cv2.VideoCapture('rtsp://192.168.1.64')
    # vs = VideoStream(r'\Users\om\Downloads\Video\YoungerPeople.mp4').start()
    vs = VideoStream(0).start()  # start capturing the video and start giving us frame
    cv2.namedWindow('Detection Window', cv2.WINDOW_NORMAL)

    # Used to calculate fps
    start_time = datetime.datetime.now()
    num_frames = 0

    try:
        while True:
            frame = vs.read()
            frame = np.array(frame)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if im_height == None:
                print('frame.shape: ', frame.shape)
                im_height, im_width = frame.shape[:2]
                print('since, im_height, im_width = frame.shape[:2],\n, im_height is frame.shape[0] = ', im_height,
                      '\nim_width is frame.shape[1]= ', im_width)

            # Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")
            # cv2.line(img=frame, pt1=(0, Line_Position1), pt2=(frame.shape[1], Line_Position1), color=(255, 0, 0), thickness=2, lineType=8, shift=0)

            # cv2.line(img=frame, pt1=(0, Line_Position2), pt2=(frame.shape[1], Line_Position2), color=(255, 0, 0), thickness=2, lineType=8, shift=0)

            # Run image through tensorflow graph
            boxes, scores, classes = detector_utils.detect_objects(frame, detection_graph, sess)
            print(f'boxes: {boxes},\n,boxes[0][1]: {boxes[0][1]},\n, scores: {scores},\n, classes: {classes}')

            # Draw bounding boxees and text
            cv_frame.draw_box_on_image(
                num_objects_detect, score_thresh, scores, boxes, classes, im_width, im_height, frame)

            # Calculate Frames per second (FPS)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() -
                            start_time).total_seconds()
            fps = num_frames / elapsed_time

            # Display FPS on frame
            cv_frame.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)
            cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                vs.stop()
                break

    except :
        print('error in app.py')
