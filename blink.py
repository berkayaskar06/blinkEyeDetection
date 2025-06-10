import argparse
import time
from imutils import face_utils
from imutils.video import FileVideoStream, VideoStream
import dlib
import cv2
from scipy.spatial import distance as dist
import imutils
import numpy as np


def eye_aspect_ratio(eye):
    """Compute the eye aspect ratio for blink detection."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def main():
    parser = argparse.ArgumentParser(description="Blink detection via EAR")
    parser.add_argument(
        "-p",
        "--shape-predictor",
        default="shape_predictor_68_face_landmarks.dat",
        help="path to dlib's facial landmark predictor",
    )
    parser.add_argument(
        "-v", "--video", help="path to optional video file"
    )
    args = parser.parse_args()

    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)

    print("[INFO] starting video stream...")
    vs = (
        FileVideoStream(args.video).start()
        if args.video
        else VideoStream(src=0).start()
    )
    time.sleep(1.0)

    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 3
    counter = 0
    total = 0

    (l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    while True:
        frame = vs.read()
        if frame is None:
            break
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            left_eye = shape[l_start:l_end]
            right_eye = shape[r_start:r_end]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            left_hull = cv2.convexHull(left_eye)
            right_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_hull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                counter += 1
            else:
                if counter >= EYE_AR_CONSEC_FRAMES:
                    total += 1
                counter = 0

            cv2.putText(
                frame,
                f"Blinks: {total}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    vs.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
