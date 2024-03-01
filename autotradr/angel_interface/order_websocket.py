import json
from autotradr.config import logger
from autotradr.angel_interface.base_websocket import BaseWebsocket


class OrderWebsocket(BaseWebsocket):
    def __init__(self, auth_token, api_key, client_code, feed_token, manager):
        super().__init__(auth_token, api_key, client_code, feed_token, manager)
        self.command_queue = manager.Queue()

    @property
    def root_uri(self):
        return "wss://tns.angelone.in/smart-order-update"

    @property
    def heart_beat_interval(self):
        return 10

    @property
    def websocket_type(self):
        return "Order websocket"

    def edit_message(self, message):
        # If the message is in bytes, decode it
        if isinstance(message, bytes):
            message = message.decode("utf-8")

        # If the message is a string, parse it as JSON
        if isinstance(message, str):
            message = json.loads(message)
        # Now, access the elements as a dictionary
        order_data = message["orderData"]
        order_id = order_data["orderid"]
        self.data_bank[order_id] = order_data

    def on_data(self, wsapp, message, data_type, continue_flag):
        if data_type == 1 and message != "pong":
            self.edit_message(message)

    def check_connection_freshness(self):
        # This method will update the connection_stale flag. The idea is to check the
        # Freshness based on data in the databank rather than the connection status.

        # Temporarily not using this method in order websocket because I need a way to fetch order book in this module.
        if self.data_bank:
            try:
                if self.wsapp and self.wsapp.sock and self.wsapp.sock.connected:
                    self.connection_stale.value = False
                else:
                    self.connection_stale.value = True
            except Exception as e:
                logger.error(f"Error in checking freshness of data: {e}")
