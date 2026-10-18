# Teste de movimento dos dedos da mão robótica.
# O código fecha e abre cada dedo, um por vez, repetindo a sequência conforme a quantidade informada pelo usuário.

import time

from modules.servo_controller import ServoController

# Porta do Arduino. Ajuste conforme a placa conectada.
# Windows: "COM3" | Linux: "/dev/ttyACM0".
communication_port = "/dev/cu.usbserial-A5069RR4"

# Associa cada pino ao ângulo usado para fechar o dedo.
closing_angles = {
    9: 120,  # Polegar.
    10: 120,  # Indicador.
    11: 160,  # Médio.
    12: 160,  # Anelar.
    13: 120,  # Mínimo.
}

# Solicita quantas vezes a sequência será executada.
number_tests = int(input("Quantas vezes deseja testar os dedos? "))

if number_tests < 1:
    raise ValueError("Informe pelo menos um teste.")

# list() reúne os pinos usados como chaves do dicionário.
# O módulo abre a conexão e configura esses pinos no modo SERVO.
controller = ServoController(
    port=communication_port,
    pins=list(closing_angles),
)

try:
    # Começa abrindo todos os dedos.
    for pin in closing_angles:
        controller.move(pin, 0)

    # Aguarda um segundo antes de começar os testes.
    time.sleep(1)

    # Repete a sequência conforme a quantidade informada.
    for test in range(number_tests):
        print(f"Iniciando teste {test + 1} de {number_tests}")

        # Pega o pino e o ângulo de cada dedo, um por vez.
        for pin, angle in closing_angles.items():
            # Fecha o dedo e aguarda um segundo.
            controller.move(pin, angle)
            time.sleep(1)

            # Abre o mesmo dedo antes de testar o próximo.
            controller.move(pin, 0)
            time.sleep(1)

finally:
    # Encerra a conexão mesmo se ocorrer um erro dentro do try.
    controller.close()
