import argparse


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Blink and fatigue detection")
    parser.add_argument(
        "-p",
        "--shape-predictor",
        default="shape_predictor_68_face_landmarks.dat",
        help="path to dlib's facial landmark predictor",
    )
    parser.add_argument("-v", "--video", help="path to optional video file")
    parser.add_argument(
        "--eye-ar-thresh", type=float, default=0.25, help="eye aspect ratio threshold"
    )
    parser.add_argument(
        "--eye-ar-consec",
        type=int,
        default=3,
        help="frames below threshold required for a blink",
    )
    parser.add_argument(
        "--drowsy-frames",
        type=int,
        default=48,
        help="consecutive frames for fatigue alert",
    )
    parser.add_argument(
        "--mar-thresh", type=float, default=0.75, help="mouth aspect ratio threshold"
    )
    parser.add_argument(
        "--yawn-consec",
        type=int,
        default=15,
        help="frames above MAR threshold required to count a yawn",
    )
    return parser.parse_args()


def eye_aspect_ratio(eye):
    """Compute the eye aspect ratio for blink detection."""
    from scipy.spatial import distance as dist

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    """Compute mouth aspect ratio for yawn detection."""
    from scipy.spatial import distance as dist

    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    return (A + B + C) / (2.0 * D)


def main():
    args = parse_args()

    import time
    import dlib
    import cv2
    from imutils import face_utils
    from imutils.video import FileVideoStream, VideoStream
    import imutils

    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)

    print("[INFO] starting video stream...")
    vs = FileVideoStream(args.video).start() if args.video else VideoStream(src=0).start()
    time.sleep(1.0)

    blink_counter = 0
    total_blinks = 0
    closed_frames = 0
    yawn_counter = 0

    (l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (m_start, m_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

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
            mouth = shape[m_start:m_end]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            mar = mouth_aspect_ratio(mouth)

            left_hull = cv2.convexHull(left_eye)
            right_hull = cv2.convexHull(right_eye)
            mouth_hull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [left_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouth_hull], -1, (0, 0, 255), 1)

            if ear < args.eye_ar_thresh:
                blink_counter += 1
                closed_frames += 1
            else:
                if blink_counter >= args.eye_ar_consec:
                    total_blinks += 1
                blink_counter = 0
                closed_frames = 0

            if mar > args.mar_thresh:
                yawn_counter += 1
            else:
                yawn_counter = 0

            fatigue = closed_frames >= args.drowsy_frames or yawn_counter >= args.yawn_consec

            cv2.putText(frame, f"Blinks: {total_blinks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if fatigue:
                cv2.putText(frame, "FATIGUE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    vs.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
