"""Controle genérico de servos por Arduino e Firmata.

Não define porta, pinos, posições iniciais ou movimentos de um projeto.
Importar o módulo não abre uma conexão com a placa.
"""

from pyfirmata2 import SERVO, Arduino


class ServoController:
    """Configura os pinos recebidos e envia ângulos aos servos."""

    def __init__(self, port, pins):
        # Porta e pinos são obrigatórios e vêm do programa principal.
        if not isinstance(port, str) or not port.strip():
            raise ValueError("Informe a porta de comunicação com o Arduino.")

        self.pins = tuple(pins)
        if not self.pins:
            raise ValueError("Informe pelo menos um pino para controlar.")
        if any(type(pin) is not int or pin < 0 for pin in self.pins):
            raise ValueError("Os pinos devem ser números inteiros não negativos.")
        if len(set(self.pins)) != len(self.pins):
            raise ValueError("Não repita pinos na configuração.")

        # A conexão só é aberta quando o programa cria o controlador.
        self.board = Arduino(port)
        self._closed = False

        try:
            for pin in self.pins:
                if pin >= len(self.board.digital):
                    raise ValueError(f"O pino {pin} não existe na placa configurada.")

            for pin in self.pins:
                self.board.digital[pin].mode = SERVO

        except Exception:
            self.close()
            raise

    def move(self, pin, angle):
        """Envia um ângulo entre 0 e 180 graus a um servo configurado.

        O limite numérico da API não garante que o ângulo seja adequado
        ao servo e à montagem; essa escolha pertence ao programa principal.
        """
        if self._closed:
            raise RuntimeError("A conexão com o Arduino já foi fechada.")
        if pin not in self.pins:
            raise ValueError(f"O pino {pin} não foi configurado neste controlador.")
        if type(angle) not in (int, float) or not 0 <= angle <= 180:
            raise ValueError("O ângulo deve ser um número entre 0 e 180 graus.")

        self.board.digital[pin].write(angle)

    def close(self):
        """Fecha a conexão, sem enviar movimentos ou posições de retorno."""
        if not self._closed:
            self.board.exit()
            self._closed = True
