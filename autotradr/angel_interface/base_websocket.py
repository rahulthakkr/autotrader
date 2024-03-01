from abc import ABC, abstractmethod
import time
import threading
import websocket
import ssl
from datetime import timedelta
from time import sleep
from autotradr.config import logger, ERROR_NOTIFICATION_SETTINGS
from autotradr.utils.core import current_time
from autotradr.utils.communication import notifier
from autotradr.angel_interface.active_session import ActiveSession


class BaseWebsocket(ABC):
    HEART_BEAT_MESSAGE = "ping"
    MAX_RETRY_ATTEMPT = 3
    RETRY_DELAY = 10

    def __init__(self, auth_token, api_key, client_code, feed_token, manager, **kwargs):
        self.auth_token = auth_token
        self.api_key = api_key
        self.client_code = client_code
        self.feed_token = feed_token
        self.wsapp = None
        self.last_pong_timestamp = None
        self.current_retry_attempt = 0
        self.connected = manager.Value("b", False)
        self.connection_stale = manager.Value("b", False)
        self.reconnecting = False
        self.intentionally_closed = manager.Value("b", False)
        self.data_bank = manager.dict()
        self.lock = manager.Lock()

    @classmethod
    def from_active_session(cls, manager, **kwargs):
        auth_token = ActiveSession.login_data["data"]["jwtToken"]
        feed_token = ActiveSession.obj.getfeedToken()
        api_key = ActiveSession.obj.api_key
        client_code = ActiveSession.obj.userId
        return cls(auth_token, api_key, client_code, feed_token, manager, **kwargs)

    @property
    @abstractmethod
    def root_uri(self):
        pass

    @property
    @abstractmethod
    def heart_beat_interval(self):
        pass

    @property
    @abstractmethod
    def websocket_type(self):
        pass

    @abstractmethod
    def on_data(self, wsapp, message, data_type, continue_flag):
        pass

    @abstractmethod
    def check_connection_freshness(self):
        # This method will update the connection_stale flag. The idea is to check the
        # Freshness based on data in teh databank rather than the connection status.
        pass

    def on_open(self, wsapp):
        logger.info(f"{self.websocket_type} connection opened.")
        self.connected.value = True
        self.reconnecting = False
        self.current_retry_attempt = 0
        parallel_tasks = [
            attr for attr in dir(self) if attr.startswith("periodically_")
        ]
        parallel_tasks = [
            getattr(self, attr)
            for attr in parallel_tasks
            if callable(getattr(self, attr))
        ]
        time.sleep(1)
        for task in parallel_tasks:
            threading.Thread(target=task, daemon=True).start()

    def on_error(self, wsapp, error):
        logger.error(f"{self.websocket_type} connection error: {error}")

    def on_close(self, wsapp, close_status_code, close_msg):
        self.connected.value = False
        if self.intentionally_closed:
            logger.info(
                f"{self.websocket_type} intentionally closed. "
                f"Status code: {close_status_code}, Message: {close_msg}"
            )
            return
        elif close_status_code not in [
            1000,
            1001,
        ]:  # Normal closure status codes
            notifier(
                f"{self.websocket_type} connection closed for unknown reason. "
                f"Status code: {close_status_code}, Message: {close_msg}",
                ERROR_NOTIFICATION_SETTINGS["url"],
            )

    def on_ping(self, wsapp, data):
        timestamp = current_time()
        logger.info(
            f"{self.websocket_type} on_ping function ==> {data}, Timestamp: {timestamp}"
        )

    def on_pong(self, wsapp, data):
        self.last_pong_timestamp = current_time()

    def maintain_connection(self):
        if not self.wsapp.sock or not self.wsapp.sock.connected:
            self.connected.value = False
            logger.error(f"{self.websocket_type} connection lost at {current_time()}")
            if not self.reconnecting:
                self.retry_connect()
        else:
            self.connected.value = True

    def periodically_maintain_connection(self):
        while True and not self.intentionally_closed:
            self.maintain_connection()
            time.sleep(5)

    def periodically_send_heart_beat(self):
        while True and not self.intentionally_closed:
            if self.connected.value:
                self.wsapp.send(self.HEART_BEAT_MESSAGE)
            time.sleep(self.heart_beat_interval)

    def periodically_check_connection_freshness(self):
        last_check_time = current_time()
        while True and not self.intentionally_closed.value:
            self.check_connection_freshness()
            if current_time() - last_check_time > timedelta(seconds=30):
                logger.info(
                    f"{self.websocket_type} connected: {not self.connection_stale.value}"
                )
                last_check_time = current_time()
            sleep(5)

    def connect(self):
        headers = {
            "Authorization": self.auth_token,
            "x-api-key": self.api_key,
            "x-client-code": self.client_code,
            "x-feed-token": self.feed_token,
        }
        try:
            self.wsapp = websocket.WebSocketApp(
                self.root_uri,
                header=headers,
                on_open=self.on_open,
                on_error=self.on_error,
                on_close=self.on_close,
                on_data=self.on_data,
                on_ping=self.on_ping,
                on_pong=self.on_pong,
            )
            self.wsapp.run_forever(
                sslopt={"cert_reqs": ssl.CERT_NONE},
                ping_interval=self.heart_beat_interval,
                ping_payload=self.HEART_BEAT_MESSAGE,
            )
        except Exception as e:
            if self.current_retry_attempt < self.MAX_RETRY_ATTEMPT:
                self.current_retry_attempt += 1
                logger.info(
                    f"{self.websocket_type} reconnecting. Attempt: {self.current_retry_attempt}"
                )
                time.sleep(self.RETRY_DELAY)
                self.connect()
            else:
                logger.error(f"{self.websocket_type} connection error: {e}")

    def retry_connect(self):
        logger.info(f"{self.websocket_type} retrying connection at {current_time()}")
        self.reconnecting = True
        if self.current_retry_attempt < self.MAX_RETRY_ATTEMPT:
            self.current_retry_attempt += 1
            logger.info(
                f"{self.websocket_type} reconnecting attempt: {self.current_retry_attempt}"
            )
            time.sleep(self.RETRY_DELAY)
            threading.Thread(target=self.connect).start()
        else:
            logger.warning(
                f"{self.websocket_type} connection retry limit exceeded. Max attempts: {self.MAX_RETRY_ATTEMPT}"
            )
            self.connected.value = False

    def close_connection(self):
        self.intentionally_closed.value = True
        if self.wsapp:
            self.wsapp.close()
