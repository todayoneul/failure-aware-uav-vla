"""Transparent two-port reverse TCP relay; no simulator or OS patch.

Windows initiates the link to avoid the previously observed WSL->Windows
inbound firewall path. Each NNG connection gets a fresh Windows server link.
"""
import argparse
import socket
import threading
import time


def pump(source, target):
    try:
        while True:
            data = source.recv(262144)
            if not data:
                break
            target.sendall(data)
    except OSError:
        pass
    finally:
        try:
            target.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def channel(local_port, reverse_port):
    remote_listener = socket.socket()
    remote_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    remote_listener.bind(("0.0.0.0", reverse_port))
    remote_listener.listen(4)
    local_listener = socket.socket()
    local_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    local_listener.bind(("127.0.0.1", local_port))
    local_listener.listen(4)
    print(f"READY local={local_port} reverse={reverse_port}", flush=True)
    while True:
        remote, remote_address = remote_listener.accept()
        print(f"WINDOWS_CONNECTED {reverse_port} {remote_address}", flush=True)
        local, _ = local_listener.accept()
        for sock in (local, remote):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threads = [threading.Thread(target=pump,args=pair,daemon=True) for pair in ((local,remote),(remote,local))]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        local.close()
        remote.close()
        print(f"SESSION_CLOSED {local_port}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=7200)
    args = parser.parse_args()
    for local, reverse in ((18989,41510),(18990,41511)):
        threading.Thread(target=channel,args=(local,reverse),daemon=True).start()
    time.sleep(args.duration)
