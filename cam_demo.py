# Este é um exemplo de como usar o módulo de detecção de mãos.
# Ele abre a câmera principal, detecta até duas mãos e desenha linhas entre os pontos das mãos, além de desenhar os pontos e seus números correspondentes.
# Esse exemplo é útil para testar a detecção de mãos e visualizar os resultados em tempo real.

import cv2

from modules.hand_tracking import HandDetector

# Escolha "video" ou "image".
detection_mode = "video"

# Abre a câmera principal.
cap = cv2.VideoCapture(0)

try:
    # Prepara a detecção de até duas mãos.
    # O with fecha o detector automaticamente ao sair deste bloco.
    with HandDetector(mode=detection_mode, max_hands=2) as detector:
        while True:
            # Captura uma imagem e separa os resultados.
            capture_result = cap.read()
            capture_success = capture_result[0]
            frame = capture_result[1]

            # Encerra se a captura falhar.
            if capture_success == False:
                break

            # Detecta as mãos na imagem.
            hands = detector.detect(frame)

            for hand_points in hands:
                # Desenha linhas verdes entre os pontos.
                detector.draw_connections(
                    frame,
                    hand_points,
                    color=(0, 255, 0),
                    thickness=4,
                )

                # Desenha pontos vermelhos e números amarelos maiores.
                detector.draw_points(
                    frame,
                    hand_points,
                    color=(0, 0, 255),
                    radius=4,
                    show_ids=True,
                    number_color=(0, 255, 255),
                    number_scale=0.8,
                    number_thickness=2,
                )

            # Mostra a imagem com os desenhos.
            cv2.imshow("Maos: pontos e conexoes", frame)

            # Pressione q com a janela em foco para encerrar.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    # Libera a câmera e fecha as janelas ao encerrar.
    cap.release()
    cv2.destroyAllWindows()
