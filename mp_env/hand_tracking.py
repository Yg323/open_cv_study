import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def recognize_gesture(landmarks):
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


#f = open('C:/Users/14dnf/Desktop/hand_data/user23.csv', 'w')

msec = 0
sec = "0"
hands = mp_hands.Hands(
    min_detection_confidence=0.7, max_num_hands=2, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(1)
while cap.isOpened():
    if msec >= 100: msec = 0
    success, image = cap.read()
    image = cv2.resize(image, dsize=(960, 720), interpolation=cv2.INTER_AREA)
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
    hand_information = str(len(handedness))
    if results.multi_hand_landmarks:
        flag = 0
        for hand_landmarks in results.multi_hand_landmarks:
            data = []
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x = [landmark.x for landmark in hand_landmarks.landmark]
            y = [landmark.y for landmark in hand_landmarks.landmark]
            z = [landmark.z for landmark in hand_landmarks.landmark]
            for i in range(21):
                data.append(x[i])
                data.append(y[i])
                hand_information += "," + str(x[i] * 960) + "," + str(y[i] * 720) + "," + str(z[i])
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
            cv2.putText(image, handedness[flag] + " Hand : " + text, (int(min(x) * 960), int(min(y) * 720) - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            hand_information += "," + handedness[flag] + "," + text
            flag += 1
        msec += 1
    if msec % 10 == 0:
        sec = str(msec // 10)
    cv2.putText(image, sec, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    cv2.moveWindow('OPEN_CV_HAND', 200, 100)
    cv2.imshow('OPEN_CV_HAND', image)
    #f.write(str(msec) + "," + hand_information + "\n")
    if cv2.waitKey(5) & 0xFF == 27:
        break
#f.close()
hands.close()
cap.release()
