import cv2


def draw_box_on_image(num_objects_detect, score_thresh, scores, boxes, classes, im_width, im_height, image_np):
    # Determined using a piece of paper of known length, code can be found in distance to camera
    focalLength = 875

    # The average width of a human hand (inches) http://www.theaveragebody.com/average_hand_size.php
    # added an inch since thumb is not included
    avg_width = 4.0
    # To more easily differetiate distances and detected bboxes

    color = None
    color0 = (255, 0, 0)
    color1 = (0, 50, 255)

    for i in range(num_objects_detect): # objects in a frame

        if (scores[i] > score_thresh):

            if classes[i] == 1:
                id = 'person'
                # b=1
            elif classes[i] == 47:
                id = 'cup'
                avg_width = 3.0  # To compensate bbox size change
            elif classes[i] == 62:
                id = 'chair'
                avg_width = 3.0
            elif classes[i] == 74:
                id = 'mouse'
                avg_width = 3.0
            elif classes[i] == 77:
                id = 'cell phone'
                avg_width = 3.0
            elif classes[i] == 84:
                id = 'book'
            elif classes[i] == 77:
                id = 'clock'
            else:
                id = 'other'


            if i == 0:
                color = color0
            elif i == 77:
                color = (255, 24, 150)
            else:
                color = color1

            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            print(f'left:{left}, right:{right}, top:{top}, bottom:{bottom}')

            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))

            dist = distance_to_camera(avg_width, focalLength, int(right - left))

            cv2.rectangle(image_np, p1, p2, color, 3, 1)

            cv2.putText(image_np, 'Object_' + str(i) + ': ' + id, (int(left), int(top) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(image_np, 'confidence: ' + str("{0:.2f}".format(scores[i])),
                        (int(left), int(top) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.putText(image_np, 'distance from camera: ' + str("{0:.2f}".format(dist) + ' inches'),
                        (int(im_width * 0.65), int(im_height * 0.9 + 30 * i)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)


# Show fps value on image.
def draw_text_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


# compute and return the distance from the hand to the camera using triangle similarity
def distance_to_camera(knownWidth, focalLength, pixelWidth):
    return (knownWidth * focalLength) / pixelWidth
