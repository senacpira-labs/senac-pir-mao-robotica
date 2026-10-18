"""Detecção de mãos reutilizável para projetos didáticos com OpenCV e MediaPipe.

O módulo não abre câmera, não cria janelas e não controla equipamentos.
O programa que o importa continua responsável por essas decisões.
"""

import time
from numbers import Real
from pathlib import Path

import cv2
import mediapipe as mp


class HandDetector:
    """Detecta mãos em imagens BGR e oferece funções opcionais de desenho."""

    # Pares de pontos que formam as linhas da mão.
    CONNECTIONS = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (13, 17),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    )

    def __init__(
        self,
        max_hands=1,
        model_path=None,
        mode="video",
        detection_confidence=0.5,
        presence_confidence=0.5,
        tracking_confidence=0.5,
    ):
        """Prepara o detector. Não captura imagens nem baixa o modelo.

        mode: "video" para sequências ou "image" para imagens independentes.
        model_path: caminho opcional para o arquivo hand_landmarker.task.
        Os três limites de confiança aceitam valores entre 0 e 1.
        """
        if (
            isinstance(max_hands, bool)
            or not isinstance(max_hands, int)
            or max_hands < 1
        ):
            raise ValueError("max_hands deve ser um número inteiro maior que zero.")
        if mode not in ("video", "image"):
            raise ValueError('mode deve ser "video" ou "image".')

        for name, value in (
            ("detection_confidence", detection_confidence),
            ("presence_confidence", presence_confidence),
            ("tracking_confidence", tracking_confidence),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} deve ser um número entre 0 e 1.")

        if model_path is None:
            module_dir = Path(__file__).resolve().parent
            # Estrutura recomendada: modules/ e models/ lado a lado.
            # Também aceita o modelo ao lado deste arquivo, como na POC inicial.
            candidates = (
                module_dir.parent / "models" / "hand_landmarker.task",
                module_dir / "models" / "hand_landmarker.task",
                module_dir / "hand_landmarker.task",
            )
            model_path = next(
                (path for path in candidates if path.is_file()), candidates[0]
            )

        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Modelo não encontrado: {model_path}. "
                "Salve hand_landmarker.task em models/ ou informe model_path."
            )

        running_mode = (
            mp.tasks.vision.RunningMode.VIDEO
            if mode == "video"
            else mp.tasks.vision.RunningMode.IMAGE
        )
        config = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        self.mode = mode
        self.last_result = None
        self._start_time = time.monotonic()
        self._last_timestamp = -1
        self._detector = mp.tasks.vision.HandLandmarker.create_from_options(config)
        self._closed = False

    @staticmethod
    def _frame_size(frame):
        """Confere o formato básico e devolve altura e largura."""
        shape = getattr(frame, "shape", ())
        if len(shape) != 3 or shape[2] != 3 or shape[0] < 1 or shape[1] < 1:
            raise ValueError(
                "Forneça uma imagem BGR com três canais, como a do OpenCV."
            )
        return shape[0], shape[1]

    def detect(self, frame, timestamp_ms=None):
        """Retorna uma lista de mãos; cada mão contém 21 pontos do MediaPipe.

        Sem mãos, retorna []. Não desenha nem modifica a imagem.
        No modo video, calcula o timestamp automaticamente se ele for omitido.
        timestamp_ms permite informar o tempo de um arquivo de vídeo.
        """
        if self._closed:
            raise RuntimeError("O detector foi fechado. Crie outro para continuar.")
        self._frame_size(frame)

        if self.mode == "image" and timestamp_ms is not None:
            raise ValueError("timestamp_ms só pode ser usado no modo video.")

        # O OpenCV fornece BGR; o MediaPipe recebe RGB.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self.mode == "video":
            if timestamp_ms is None:
                timestamp = int((time.monotonic() - self._start_time) * 1000)
                timestamp = max(timestamp, self._last_timestamp + 1)
            else:
                if (
                    isinstance(timestamp_ms, bool)
                    or not isinstance(timestamp_ms, int)
                    or timestamp_ms < 0
                    or timestamp_ms <= self._last_timestamp
                ):
                    raise ValueError(
                        "timestamp_ms deve ser inteiro, não negativo e crescente."
                    )
                timestamp = timestamp_ms

            result = self._detector.detect_for_video(image, timestamp)
            self._last_timestamp = timestamp
        else:
            result = self._detector.detect(image)

        # Mantém o resultado completo disponível para usos mais avançados.
        self.last_result = result
        return result.hand_landmarks

    def to_pixels(self, frame, hand_points):
        """Converte os pontos proporcionais em uma lista de posições (x, y)."""
        height, width = self._frame_size(frame)
        return [(int(point.x * width), int(point.y * height)) for point in hand_points]

    def draw_points(
        self,
        frame,
        hand_points,
        color=(0, 0, 255),
        radius=4,
        show_ids=False,
        number_color=None,
        number_scale=0.4,
        number_thickness=1,
    ):
        """Desenha na própria imagem e a retorna. As cores usam BGR.

        number_color: cor dos números; None usa a mesma cor dos pontos.
        number_scale: escala do texto; valores maiores ampliam os números.
        number_thickness: espessura do texto em pixels.
        """
        if isinstance(radius, bool) or not isinstance(radius, int) or radius < 1:
            raise ValueError("radius deve ser um número inteiro maior que zero.")
        if (
            isinstance(number_scale, bool)
            or not isinstance(number_scale, Real)
            or not 0 < number_scale < float("inf")
        ):
            raise ValueError("number_scale deve ser um número positivo e finito.")
        if (
            isinstance(number_thickness, bool)
            or not isinstance(number_thickness, int)
            or number_thickness < 1
        ):
            raise ValueError("number_thickness deve ser um inteiro maior que zero.")

        if number_color is None:
            number_color = color

        for index, (x, y) in enumerate(self.to_pixels(frame, hand_points)):
            cv2.circle(frame, (x, y), radius, color, -1)
            if show_ids:
                cv2.putText(
                    frame,
                    str(index),
                    (x + radius + 3, y - radius - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    number_scale,
                    number_color,
                    number_thickness,
                )
        return frame

    def draw_connections(self, frame, hand_points, color=(0, 255, 0), thickness=2):
        """Liga os 21 pontos com linhas. Modifica e retorna a própria imagem."""
        if (
            isinstance(thickness, bool)
            or not isinstance(thickness, int)
            or thickness < 1
        ):
            raise ValueError("thickness deve ser um número inteiro maior que zero.")
        pixels = self.to_pixels(frame, hand_points)
        if len(pixels) != 21:
            raise ValueError("draw_connections precisa dos 21 pontos de uma mão.")

        for start, end in self.CONNECTIONS:
            cv2.line(frame, pixels[start], pixels[end], color, thickness)
        return frame

    def close(self):
        """Fecha o detector. Chamar novamente após o fechamento não faz nada."""
        if not self._closed:
            self._detector.close()
            self._closed = True

    def __enter__(self):
        """Permite usar with para fechar o detector automaticamente."""
        if self._closed:
            raise RuntimeError("O detector já foi fechado.")
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()
