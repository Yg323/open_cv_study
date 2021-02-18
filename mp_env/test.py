import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# The landmarks array has the following structur: [x0, y0, x1, y1, ....., x20, y20]
# with for example x0 and y0 the x and y values of the landmark at index 0.
test_landmarks_data = [
    0.499651, 0.849638, 0.614354, 0.796254,
    0.686660, 0.692482, 0.743792, 0.606666,
    0.809362, 0.512337, 0.538779, 0.499517,
    0.513829, 0.361394, 0.484049, 0.260214,
    0.452508, 0.173999, 0.445565, 0.512067,
    0.396448, 0.358399, 0.355494, 0.245083, 0.318670,
    0.157915, 0.355069, 0.562040, 0.278774,
    0.435983, 0.221781, 0.345394, 0.178977, 0.273430,
    0.288238, 0.631016, 0.219506, 0.544787,
    0.162939, 0.483343, 0.110222, 0.422808]  # true label: 5


### Functions
def recognizeHandGesture(landmarks):
    thumbState = 'UNKNOW'
    indexFingerState = 'UNKNOW'
    middleFingerState = 'UNKNOW'
    ringFingerState = 'UNKNOW'
    littleFingerState = 'UNKNOW'
    recognizedHandGesture = None

    pseudoFixKeyPoint = landmarks[2]['x']
    if (landmarks[3]['x'] < pseudoFixKeyPoint and landmarks[4]['x'] < landmarks[3]['x']):
        thumbState = 'CLOSE'
    elif (pseudoFixKeyPoint < landmarks[3]['x'] and landmarks[3]['x'] < landmarks[4]['x']):
        thumbState = 'OPEN'

    pseudoFixKeyPoint = landmarks[6]['y']
    if (landmarks[7]['y'] < pseudoFixKeyPoint and landmarks[8]['y'] < landmarks[7]['y']):
        indexFingerState = 'OPEN'
    elif (pseudoFixKeyPoint < landmarks[7]['y'] and landmarks[7]['y'] < landmarks[8]['y']):
        indexFingerState = 'CLOSE'

    pseudoFixKeyPoint = landmarks[10]['y']
    if (landmarks[11]['y'] < pseudoFixKeyPoint and landmarks[12]['y'] < landmarks[11]['y']):
        middleFingerState = 'OPEN'
    elif (pseudoFixKeyPoint < landmarks[11]['y'] and landmarks[11]['y'] < landmarks[12]['y']):
        middleFingerState = 'CLOSE'

    pseudoFixKeyPoint = landmarks[14]['y']
    if (landmarks[15]['y'] < pseudoFixKeyPoint and landmarks[16]['y'] < landmarks[15]['y']):
        ringFingerState = 'OPEN'
    elif (pseudoFixKeyPoint < landmarks[15]['y'] and landmarks[15]['y'] < landmarks[16]['y']):
        ringFingerState = 'CLOSE'

    pseudoFixKeyPoint = landmarks[18]['y']
    if (landmarks[19]['y'] < pseudoFixKeyPoint and landmarks[20]['y'] < landmarks[19]['y']):
        littleFingerState = 'OPEN'
    elif (pseudoFixKeyPoint < landmarks[19]['y'] and landmarks[19]['y'] < landmarks[20]['y']):
        littleFingerState = 'CLOSE'

    if (
            thumbState == 'OPEN' and indexFingerState == 'OPEN' and middleFingerState == 'OPEN' and ringFingerState == 'OPEN' and littleFingerState == 'OPEN'):
        recognizedHandGesture = 5  # "FIVE"
    elif (
            thumbState == 'CLOSE' and indexFingerState == 'OPEN' and middleFingerState == 'OPEN' and ringFingerState == 'OPEN' and littleFingerState == 'OPEN'):
        recognizedHandGesture = 4  # "FOUR"
    elif (
            thumbState == 'OPEN' and indexFingerState == 'OPEN' and middleFingerState == 'OPEN' and ringFingerState == 'CLOSE' and littleFingerState == 'CLOSE'):
        recognizedHandGesture = 3  # "TREE"
    elif (
            thumbState == 'OPEN' and indexFingerState == 'OPEN' and middleFingerState == 'CLOSE' and ringFingerState == 'CLOSE' and littleFingerState == 'CLOSE'):
        recognizedHandGesture = 2  # "TWO"
    elif (
            thumbState == 'CLOSE' and indexFingerState == 'OPEN' and middleFingerState == 'CLOSE' and ringFingerState == 'CLOSE' and littleFingerState == 'CLOSE'):
        recognizedHandGesture = 1  # "ONE"
    else:
        recognizedHandGesture = 0  # "UNKNOW"
    return recognizedHandGesture


def getStructuredLandmarks(landmarks):
    structuredLandmarks = []
    for j in range(42):
        if (j % 2 == 1):
            structuredLandmarks.append({'x': landmarks[j - 1], 'y': landmarks[j]})
    return structuredLandmarks


# For webcam input:
hands = mp_hands.Hands(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = True
    results = hands.process(image)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data = []
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            print(hand_landmarks)
            x = [landmark.x for landmark in hand_landmarks.landmark]
            y = [landmark.y for landmark in hand_landmarks.landmark]
            for i in range(21):
                data.append(x[i])
                data.append(y[i])
            recognizedHandGesture = recognizeHandGesture(getStructuredLandmarks(data))
            if recognizedHandGesture == 1:
                text = "ONE"
            elif recognizedHandGesture == 2:
                text = "TWO"
            elif recognizedHandGesture == 3:
                text = "THREE"
            elif recognizedHandGesture == 4:
                text = "FOUR"
            elif recognizedHandGesture == 5:
                text = "FIVE"
            else:
                text = "ZERO"
            cv2.putText(image, text, (int(x[12] * 360), int(y[12] * 360)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, lineType=cv2.LINE_AA )

    cv2.imshow('OPEN_CV_HAND', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break
hands.close()
cap.release()
