import cv2
import mediapipe as mp

camera = cv2.VideoCapture(0)

draw_mediapipe = mp.solutions.drawing_utils
hands_mediapipe = mp.solutions.hands
hands = hands_mediapipe.Hands(max_num_hands=2)

while True:
    ret, frame = camera.read()

    # convertendo as cores do frame para melhor detecção das mãos
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    
    # vai armazenar os pontos de cada mão
    double_hands = ([], []) 

    if results.multi_hand_landmarks:
        for index, handsLms in enumerate(results.multi_hand_landmarks):
            for id, lm in enumerate(handsLms.landmark):
                h, w, _ = frame.shape
                # mapeia os pixels dos pontos x, y
                cx, cy = int(lm.x * w), int(lm.y * h)
                double_hands[index].append((cx, cy))

            # desenha a mão
            draw_mediapipe.draw_landmarks(frame, handsLms, hands_mediapipe.HAND_CONNECTIONS)
        try:
            # desenha as linhas ligando os pontos dos dedos
            cv2.line(frame, double_hands[0][4], double_hands[1][4], (255, 0, 0), 3)
            cv2.line(frame, double_hands[0][8], double_hands[1][8], (255, 255, 0), 3)
            cv2.line(frame, double_hands[0][12], double_hands[1][12], (0, 255, 0), 3)
            cv2.line(frame, double_hands[0][16], double_hands[1][16], (0, 0, 255), 3)
            cv2.line(frame, double_hands[0][20], double_hands[1][20], (143, 0, 255), 3)
        except:
            pass

    cv2.imshow("Empowerpython", frame)

    # finaliza a aplicação ao pressionar a tecla q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
