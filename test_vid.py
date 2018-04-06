import cv2
import bgsnet

# parameter setting
datadir = '/home/yzhq/data/DrDu/'
video_file = 'testhalf.mp4'
min_area = 256
speed = 1.0
max_stay = 240
width = 800
height = 600
output = 0  # 0 for display, 1 for file
out_file = 'out.mp4'

video = cv2.VideoCapture(datadir + video_file)
video.open(datadir + video_file)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(out_file, fourcc, 20.0, (width, height))

bgs = bgsnet.detector('vgg16partial.npy', 'train/bgsnet30.npy')
objects = []


# object
class Object:
    def __init__(self, frame, box):
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, box)
        self.box = box
        self.cnt = 1
        self.matched = True
        self.updated = False
        self.area = box[2] * box[3]

    def update(self, frame):
        self.updated, self.box = self.tracker.update(frame)
        self.area = self.box[2] * self.box[3]
        self.cnt += 1


# utility
def overlap_ratio(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    f1 = 0.0
    f2 = 0.0
    if x1 < x2:
        if x1 + w1 > x2 + w2:
            f1 = w2
        else:
            f1 = x1 + w1 - x2
    else:
        if x2 + w2 > x1 + w1:
            f1 = w1
        else:
            f1 = x2 + w2 - x1
    if f1 <= 0: return -1.0

    if y1 < y2:
        if y1 + h1 > y2 + h2:
            f2 = h2
        else:
            f2 = y1 + h1 - y2
    else:
        if y2 + h2 > y1 + h1:
            f2 = h1
        else:
            f2 = y2 + h2 - y1
    if f2 <= 0: return -1.0

    overlap = f1 * f2

    a1 = w1 * h1
    a2 = w2 * h2

    r1 = 1.0 * overlap / a1
    r2 = 1.0 * overlap / a2

    return max(r1, r2)


# bgs initialization
print('initializing...')
for i in range(50):
    ok, frame = video.read()
    if not ok: break
    frame = cv2.resize(frame, (width, height))
    bgs.apply(frame)

# main process
print('tracking...')
rf = 0
vf = round(rf / speed)
while True:
    # read frame
    ok, frame = video.read()
    if not ok: break
    rf += 1
    if round(rf / speed) > vf:
        vf = round(rf / speed)
    else:
        continue

    # pre-processing
    frame = cv2.resize(frame, (width, height))

    for i in range(len(objects)):
        objects[i].update(frame)
    objects = [obj for obj in objects if (obj.updated and obj.area >= min_area)]
    for obj in objects: obj.matched = False
    print 'tracking %d objects' % len(objects)

    masked = bgs.apply(frame)
    #masked = cv2.dilate(masked, None, iterations=1)

    (_, cnts, _) = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in cnts]
    boxes = [box for box in boxes if box[2] * box[3] >= min_area]
    for box in boxes:
        matched = False
        for obj in objects:
            ratio = overlap_ratio(box, obj.box)
            if ratio > 0.5:
                matched = True
                obj.matched = True
                break
        if not matched:
            obj = Object(frame, box)
            objects.append(obj)

    objects = [obj for obj in objects if obj.matched]

    # visualize
    for obj in objects:
        p1 = (int(obj.box[0]), int(obj.box[1]))
        p2 = (int(obj.box[0] + obj.box[2]), int(obj.box[1] + obj.box[3]))
        stay = min(obj.cnt, max_stay)
        red = 255 * stay / max_stay
        green = 255 - 255 * stay / max_stay
        cv2.rectangle(frame, p1, p2, (0, green, red), 1, 1)

    # output
    if output == 1:
        out.write(frame)
        print 'writting frame ', vf
    else:
        cv2.imshow("1", frame)
        cv2.imshow("2", masked)

    k = cv2.waitKey(1) & 0xff
    if k == 27: break
