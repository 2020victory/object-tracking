import argparse
import cv2
import json

parser = argparse.ArgumentParser(description='Opencv')

parser.add_argument('--track', '-t', action='store_true',
                       help='To prompt tracking')

parser.add_argument("--type", type=str, default="kcf", required=False,
	help="OpenCV object tracker type")

args = parser.parse_args()

OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		# "boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		# "tld": cv2.TrackerTLD_create,
		# "medianflow": cv2.TrackerMedianFlow_create,
		# "mosse": cv2.TrackerMOSSE_create
	}

print(args.type)

if not args.type in list(OPENCV_OBJECT_TRACKERS.keys()):
    print("Invalid class")
    exit(-1)


tracker = OPENCV_OBJECT_TRACKERS[args.type]()
video = cv2.VideoCapture(0)

if args.track:
    while True:
        k, frame = video.read()
        # frame = cv2.flip(frame, 1)
        cv2.imshow("Tracking", frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    bbox = cv2.selectROI(frame, False)
    cv2.imwrite('file.jpg', frame)
    with open('coordinates.json', 'w') as fh:
        fh.write(json.dumps({"coord": list(bbox)}))
    cv2.destroyWindow("ROI selector")


frame = cv2.imread('file.jpg')
with open('coordinates.json') as fh:
    obj = json.loads(fh.read())
    bbox = tuple(obj['coord'])

ok = tracker.init(frame, bbox)

while True:
    ok, frame = video.read()
    # frame = cv2.flip(frame, 1)
    ok, bbox = tracker.update(frame)

    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]),
              int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 0, 255), 2, 2)

    cv2.imshow("Tracking", frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break