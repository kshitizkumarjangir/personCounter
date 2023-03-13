import torch
import cv2
import numpy as np
import time
import matplotlib

# import deep sort libraries
from deep_sort_realtime.deepsort_tracker import DeepSort

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

exit_area = np.array([
    (888, 720),  # bottom-left corner
    (1096, 540),  # top-left corner
    (1203, 563),  # top-right corner
    (1071, 720)],  # bottom-right corner
    np.int32)

entry_line_len = np.sqrt((exit_area[0][0] - exit_area[1][0]) ** 2 + (exit_area[0][1] - exit_area[1][1]) ** 2)


def is_pt_crossed_entry_line(pt):
    """
    (888, 720),  # bottom-left corner
    (1087, 452)  # top-left corner
    """
    len_pt_ep1 = np.sqrt((exit_area[0][0] - pt[0]) ** 2 + (exit_area[0][1] - pt[1]) ** 2)
    len_pt_ep2 = np.sqrt((exit_area[1][0] - pt[0]) ** 2 + (exit_area[1][1] - pt[1]) ** 2)

    fraction = (len_pt_ep1 + len_pt_ep2) / entry_line_len

    if 1.05 >= fraction > 0.95:
        return True
    else:
        return False


def is_pt_inside_entry_area(pt):
    """
    (888, 720),  # bottom-left corner
    (1096, 540),  # top-left corner
    (1203, 563),  # top-right corner
    (1071, 720)  # bottom-right corner
    """
    path = matplotlib.path.Path(exit_area)

    is_inside = path.contains_point(pt)

    return is_inside


def detectPerson(video_feed):
    # Generate random color
    rand_color = list(np.random.choice(range(255), size=(80, 3), replace=False))

    # Load Yolov5 model
    model = torch.hub.load('yolov5', 'yolov5m', source='local')

    class_list = model.names  # class name dict

    # Load Deep Sort model to track the objects
    object_tracker = DeepSort(max_age=10,
                              n_init=2,
                              nms_max_overlap=0.8,
                              max_cosine_distance=0.3,
                              nn_budget=None,
                              override_track_class=None,
                              embedder="mobilenet",
                              half=True,
                              bgr=True,
                              embedder_gpu=True,
                              embedder_model_name=None,
                              embedder_wts=None,
                              polygon=False,
                              today=None)

    personData = {}  # Key: ID, Value: is_pt_inside_entry_area
    enteredCount = 0
    exitedCount = 0
    enteredPeople = []  # list of entered people ID
    exitedPeople = []  # list of exited people ID

    # read video feed
    cap = cv2.VideoCapture(video_feed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        _, img = cap.read()
        if img is None:
            break

        t1 = time.time()

        # Detect frame image - Yolo
        result = model(img)

        # Read detected objects frame data
        df = result.pandas().xyxy[0]

        detections = []
        for idx in df.index:
            class_name = df['name'][idx]
            confidence = (df['confidence'][idx]).round(2)

            if confidence > min_confidence:  # check for certainty
                if class_name in object_type_to_tracked or len(object_type_to_tracked) == 0:
                    x1, y1 = int(df['xmin'][idx]), int(df['ymin'][idx])
                    x2, y2 = int(df['xmax'][idx]), int(df['ymax'][idx])

                    detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], confidence, class_name))

        # Tracking Objects - Using Deep Sort
        tracks = object_tracker.update_tracks(detections, frame=img)

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            track_id = track.track_id
            bbox = track.to_ltrb()
            class_name = track.get_det_class()

            obj_color = rand_color[list(class_list.values()).index(class_name)]
            obj_color = tuple(np.ndarray.tolist(obj_color))

            # Add marker and class name + track ID
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), obj_color, 1)
            if show_annotation:
                cv2.putText(img, str(track_id), (int(bbox[0]), int(bbox[1] - 10)),
                            font, 0.9, obj_color, 2)

            # mark entry/exit line and space
            # pts = exit_area.reshape((-1, 1, 2))
            # cv2.polylines(img, [pts], True, (123, 255, 255), 1)
            cv2.line(img, (exit_area[0][0], exit_area[0][1]), (exit_area[1][0], exit_area[1][1]), (255, 0, 0), 3)

            # BBox center point
            c_pt = (int(bbox[2]), int(bbox[3]))

            # cv2.circle(img, c_pt, 3, (255, 123, 255), 2)

            if track_id in personData.keys():

                # check for last state of person
                if is_pt_crossed_entry_line(c_pt):
                    if personData[track_id]:  #
                        if track_id not in exitedPeople:
                            exitedCount += 1
                            exitedPeople.append(track_id)
                    else:
                        if track_id not in enteredPeople:
                            enteredCount += 1
                            enteredPeople.append(track_id)
                else:
                    if is_pt_inside_entry_area(c_pt):  # current state of person
                        personData[track_id] = True
                    else:
                        personData[track_id] = False

            else:
                personData[track_id] = False

            # Update exit/entry status
            cv2.putText(img, "Entered People Count: " + str(enteredCount), (5, 50), font, 0.9, (255, 0, 255), 1)
            cv2.putText(img, "Exited People Count: " + str(exitedCount), (5, 80), font, 0.9, (255, 0, 255), 1)
            # cv2.putText(img, "Max. People Allowed Inside: " + str(max_people_allowed), (5, 110), font, 0.9,
            #             (255, 128, 0), 1)
            cv2.putText(img, "Total People Inside: " + str(enteredCount - exitedCount), (5, 110), font, 1,
                        (89, 255, 84), 1)
            if (enteredCount - exitedCount) > max_people_allowed:
                cv2.putText(img, "Warning! The space is overcrowded." + '\n' +
                            "Total people inside are more than maximum allowable safety limit of " +
                            str(max_people_allowed), (310, 30), font, 1, (0, 0, 255), 1)

        # Fetch FPS
        fps = 1. / (time.time() - t1)
        cv2.putText(img, "FPS: {:.0f}".format(fps), org=(5, 30), fontFace=font, fontScale=2,
                    color=(255, 255, 0), thickness=2)

        img = cv2.resize(img, (1080, 720))
        cv2.imshow(root_window, img)

        c = cv2.waitKey(1)
        if c == 27:  # Stop when Esc is pressed
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # ----------------------------------Control Params-----------------------------------------------------------
    '''
        Entry/Exit line-
        bottom-left = (888, 720)*
        top-left    = (1087, 452)*
        top-right   = (1203, 563)
        bottom-right= (1071, 720)
    '''

    root_window = 'personCounter'
    show_annotation = True
    max_people_allowed = 5

    object_type_to_tracked = ['person']
    min_confidence = 0.49

    cv2.namedWindow(root_window)

    videoPath = 'testData/station.mp4'

    detectPerson(videoPath)
