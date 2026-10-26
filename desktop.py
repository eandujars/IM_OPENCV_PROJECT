import socket
import threading
import time

import uvicorn
import webview

HOST = "127.0.0.1"
PORT = 8000


def esperar_servidor(host: str, port: int) -> None:
    while True:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)


def ejecutar_servidor(server: uvicorn.Server) -> None:
    server.run()


def main() -> None:
    config = uvicorn.Config(
        "main:app",
        host=HOST,
        port=PORT,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=ejecutar_servidor, args=(server,), daemon=True)
    thread.start()
    esperar_servidor(HOST, PORT)

    webview.create_window(
        "Regla Medidora",
        f"http://{HOST}:{PORT}",
        width=1280,
        height=850,
        min_size=(960, 650),
        resizable=True,
    )
    webview.start()
    server.should_exit = True


if __name__ == "__main__":
    main()
