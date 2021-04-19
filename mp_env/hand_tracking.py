import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def recognize_gesture(landmarks):
    print(landmarks)
    fingers = ['UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN']

    pseudoFixKeyPoint = landmarks[2]['x']
    if landmarks[1]['x'] > landmarks[17]['x']:
        if pseudoFixKeyPoint > landmarks[3]['x'] > landmarks[4]['x']:
            fingers[0] = 'CLOSE'
        elif pseudoFixKeyPoint < landmarks[3]['x'] < landmarks[4]['x']:
            fingers[0] = 'OPEN'
    else:
        if pseudoFixKeyPoint > landmarks[3]['x'] > landmarks[4]['x']:
            fingers[0] = 'OPEN'
        elif pseudoFixKeyPoint < landmarks[3]['x'] < landmarks[4]['x']:
            fingers[0] = 'CLOSE'

    pseudoFixKeyPoint = landmarks[6]['y']
    if pseudoFixKeyPoint > landmarks[7]['y'] > landmarks[8]['y']:
        fingers[1] = 'OPEN'
    elif pseudoFixKeyPoint < landmarks[7]['y'] < landmarks[8]['y']:
        fingers[1] = 'CLOSE'

    pseudoFixKeyPoint = landmarks[10]['y']
    if pseudoFixKeyPoint > landmarks[11]['y'] > landmarks[12]['y']:
        fingers[2] = 'OPEN'
    elif pseudoFixKeyPoint < landmarks[11]['y'] < landmarks[12]['y']:
        fingers[2] = 'CLOSE'

    pseudoFixKeyPoint = landmarks[14]['y']
    if pseudoFixKeyPoint > landmarks[15]['y'] > landmarks[16]['y']:
        fingers[3] = 'OPEN'
    elif pseudoFixKeyPoint < landmarks[15]['y'] < landmarks[16]['y']:
        fingers[3] = 'CLOSE'

    pseudoFixKeyPoint = landmarks[18]['y']
    if pseudoFixKeyPoint > landmarks[19]['y'] > landmarks[20]['y']:
        fingers[4] = 'OPEN'
    elif pseudoFixKeyPoint < landmarks[19]['y'] < landmarks[20]['y']:
        fingers[4] = 'CLOSE'

    recognized_hand_gesture = fingers.count("OPEN")

    return recognized_hand_gesture


def getStructuredLandmarks(landmarks):
    structuredLandmarks = []
    for j in range(42):
        if (j % 2 == 1):
            structuredLandmarks.append({'x': landmarks[j - 1], 'y': landmarks[j]})
    return structuredLandmarks


# f = open('C:/Users/14dnf/Desktop','w')
flags = 0
sec = "0"
hands = mp_hands.Hands(
    min_detection_confidence=0.5, max_num_hands=2, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    if flags == 100: flags = 0
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = True
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    handedness = ["UNKNOWN"]

    if results.multi_handedness:
        handedness = [
            handedness.classification[0].label
            for handedness in results.multi_handedness
        ]

    if results.multi_hand_landmarks:
        flag = 0
        for hand_landmarks in results.multi_hand_landmarks:
            data = []
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x = [landmark.x for landmark in hand_landmarks.landmark]
            y = [landmark.y for landmark in hand_landmarks.landmark]
            for i in range(21):
                data.append(x[i])
                data.append(y[i])
            recognizedHandGesture = recognize_gesture(getStructuredLandmarks(data))
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
            cv2.putText(image, handedness[flag] + " Hand : " + text, (int(x[12] * 360), int(y[12] * 360)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            flags += 1
    if flags % 10 == 0:
        sec = str(flags // 10)
    cv2.putText(image, sec, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

    cv2.imshow('OPEN_CV_HAND', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break
hands.close()
cap.release()
