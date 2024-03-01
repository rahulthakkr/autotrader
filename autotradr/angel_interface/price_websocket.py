import json
import struct
from datetime import datetime, timezone
import itertools
from collections import defaultdict
from time import sleep
from datetime import timedelta
from autotradr.config import token_symbol_dict, logger, token_exchange_dict
from autotradr.utils import (
    current_time,
    get_symbol_token,
    strike_range_different,
    log_error,
)
from autotradr.angel_interface.base_websocket import BaseWebsocket


class PriceWebsocket(BaseWebsocket):
    LITTLE_ENDIAN_BYTE_ORDER = "<"

    # Available Actions
    SUBSCRIBE_ACTION = 1
    UNSUBSCRIBE_ACTION = 0

    # Possible Subscription Mode
    LTP_MODE = 1
    QUOTE = 2
    SNAP_QUOTE = 3
    DEPTH = 4

    # Exchange Type
    EXCHANGE_TYPE_MAP = {
        "NSE": 1,
        "NFO": 2,
        "BSE": 3,
        "BFO": 4,
        "MCX": 5,
        "NCDEX": 7,
        "CDS": 13,
    }

    # Subscription Mode Map
    SUBSCRIPTION_MODE_MAP = {1: "LTP", 2: "QUOTE", 3: "SNAP_QUOTE", 4: "DEPTH"}

    def __init__(
        self,
        auth_token,
        api_key,
        client_code,
        feed_token,
        manager,
        correlation_id="default",
        webhook_url=None,
        default_strike_range=7,
    ):
        self.correlation_id = correlation_id
        self.webhook_url = webhook_url
        self.default_strike_range = default_strike_range

        super().__init__(auth_token, api_key, client_code, feed_token, manager)

        self.command_queue = manager.Queue()
        self.input_request_dict = manager.dict()
        self.underlying_options_subscribed = manager.dict()
        self.subscribed_to_options = False
        self.RESUBSCRIBE_FLAG = False
        self._processing_subscriptions = False

    @staticmethod
    def processing_subscriptions(
        func,
    ):  # This decorator is used to avoid updating strike range while
        # subscription processes are running
        def wrapper(self, *args, **kwargs):
            self._processing_subscriptions = True
            result = func(self, *args, **kwargs)
            self._processing_subscriptions = False
            return result

        return wrapper

    @property
    def root_uri(self):
        return "wss://smartapisocket.angelone.in/smart-stream"

    @property
    def heart_beat_interval(self):
        return 10

    @property
    def websocket_type(self):
        return "Price websocket"

    def on_open(self, wsapp):
        if self.RESUBSCRIBE_FLAG:
            self.resubscribe()
        super().on_open(wsapp)

    def on_data(self, wsapp, message, data_type, continue_flag):
        if data_type == 2:
            parsed_data_packet = self._parse_binary_data(message)
            self.handle_data(parsed_data_packet)

    def handle_data(self, message):
        self.data_bank[message["token"]] = {
            "ltp": message["last_traded_price"] / 100,
            "best_bid": message["best_5_sell_data"][0]["price"] / 100
            if "best_5_sell_data" in message
            else None,
            # 'best_5_sell_data' is not present in 'mode 1' messages
            "best_bid_qty": message["best_5_sell_data"][0]["quantity"]
            if "best_5_sell_data" in message
            else None,
            "best_ask": message["best_5_buy_data"][0]["price"] / 100
            if "best_5_buy_data" in message
            else None,
            "best_ask_qty": message["best_5_buy_data"][0]["quantity"]
            if "best_5_buy_data" in message
            else None,
            "timestamp": datetime.fromtimestamp(
                message["exchange_timestamp"] / 1000,
                tz=timezone(timedelta(hours=5, minutes=30)),
            ).replace(tzinfo=None),
            "last_traded_datetime": datetime.fromtimestamp(
                message["last_traded_timestamp"],
                tz=timezone(timedelta(hours=5, minutes=30)),
            ).replace(tzinfo=None)
            if "last_traded_timestamp" in message
            else None,
            **message,
        }

    def on_error(self, wsapp, error):
        self.RESUBSCRIBE_FLAG = True
        if self.current_retry_attempt < self.MAX_RETRY_ATTEMPT:
            self.current_retry_attempt += 1
            sleep(self.RETRY_DELAY)

            try:
                self.close_connection()
                self.connect()
            except Exception as e:
                logger.error(f"{self.websocket_type} error in on_error: {e}")
        else:
            self.close_connection()
            logger.error(f"{self.websocket_type} error in on_error: {error}")

    def _create_payload(self, tokens: list):
        payload = defaultdict(list)
        for token in tokens:
            exchange = token_exchange_dict.get(token)
            if exchange:
                payload[self.EXCHANGE_TYPE_MAP[exchange]].append(token)
        return [{"exchangeType": key, "tokens": val} for key, val in payload.items()]

    def subscribe(self, tokens: list, mode: int = 1):
        payload = self._create_payload(tokens)
        self._subscribe(self.correlation_id, mode, payload)

    def unsubscribe(self, tokens: list, mode: int = 1):
        payload = self._create_payload(tokens)
        self._unsubscribe(self.correlation_id, mode, payload)

    def get_current_usage(self):
        return sum(
            len(tokens) * mode
            for mode, val in self.input_request_dict.items()
            for _, tokens in val.items()
        )

    def close_connection(self):
        self.intentionally_closed.value = True
        self.RESUBSCRIBE_FLAG = False
        if self.wsapp:
            self.wsapp.close()

    @processing_subscriptions
    def _subscribe(self, correlation_id: str, mode: int, token_list: list[dict]):
        """
        This Function subscribe the price data for the given token
        Parameters
        ------
        correlation_id: string
            A 10 character alphanumeric ID client may provide which will be returned by the server in error response
            to indicate which request generated error response.
            Clients can use this optional ID for tracking purposes between request and corresponding error response.
        mode: integer
            It denotes the subscription type
            possible values -> 1, 2 and 3
            1 -> LTP
            2 -> Quote
            3 -> Snap Quote
        token_list: list of dict
            Sample Value ->
                [
                    { "exchangeType": 1, "tokens": ["10626", "5290"]},
                    {"exchangeType": 5, "tokens": [ "234230", "234235", "234219"]}
                ]
                exchangeType: integer
                possible values ->
                    1 -> nse_cm
                    2 -> nse_fo
                    3 -> bse_cm
                    4 -> bse_fo
                    5 -> mcx_fo
                    7 -> ncx_fo
                    13 -> cde_fo
                tokens: list of string
        """
        try:
            request_data = {
                "correlationID": correlation_id,
                "action": self.SUBSCRIBE_ACTION,
                "params": {"mode": mode, "tokenList": token_list},
            }
            if mode == 4:
                for token in token_list:
                    if token.get("exchangeType") != 1:
                        error_message = (
                            f"{self.websocket_type} subscribe error\n"
                            f"Invalid ExchangeType:{token.get('exchangeType')} "
                            f"Please check the exchange type and try again it support only 1 exchange type"
                        )
                        logger.error(error_message)
                        raise ValueError(error_message)

            if mode == self.DEPTH:
                total_tokens = sum(len(token["tokens"]) for token in token_list)
                quota_limit = 50
                if total_tokens > quota_limit:
                    error_message = (
                        f"Price websocket quota exceeded: "
                        f"You can subscribe to a maximum of {quota_limit} tokens only."
                    )
                    logger.error(error_message)
                    raise Exception(error_message)

            if self.input_request_dict.get(mode) is None:
                self.input_request_dict[mode] = {}
            for token in token_list:
                if token["exchangeType"] in self.input_request_dict[mode]:
                    self.input_request_dict[mode][token["exchangeType"]].extend(
                        token["tokens"]
                    )
                else:
                    self.input_request_dict[mode][token["exchangeType"]] = token[
                        "tokens"
                    ]

            self.wsapp.send(json.dumps(request_data))
            self.RESUBSCRIBE_FLAG = True
            sleep(1)

        except Exception as e:
            logger.error(f"Price websocket error occurred during subscribe: {e}")
            raise e

    @processing_subscriptions
    def _unsubscribe(self, correlation_id, mode, token_list):
        """
        This function unsubscribe the data for given token
        Parameters
        ------
        correlation_id: string
            A 10 character alphanumeric ID client may provide which will be returned by the server in error response
            to indicate which request generated error response.
            Clients can use this optional ID for tracking purposes between request and corresponding error response.
        mode: integer
            It denotes the subscription type
            possible values -> 1, 2 and 3
            1 -> LTP
            2 -> Quote
            3 -> Snap Quote
        token_list: list of dict
            Sample Value ->
                [
                    { "exchangeType": 1, "tokens": ["10626", "5290"]},
                    {"exchangeType": 5, "tokens": [ "234230", "234235", "234219"]}
                ]
                exchangeType: integer
                possible values ->
                    1 -> nse_cm
                    2 -> nse_fo
                    3 -> bse_cm
                    4 -> bse_fo
                    5 -> mcx_fo
                    7 -> ncx_fo
                    13 -> cde_fo
                tokens: list of string
        """
        # Remove unsubscribed tokens from input_request_dict
        for token_dict in token_list:
            exchange_type = token_dict["exchangeType"]
            tokens_to_remove = token_dict["tokens"]
            if (
                mode in self.input_request_dict
                and exchange_type in self.input_request_dict[mode]
            ):
                self.input_request_dict[mode][exchange_type] = [
                    token
                    for token in self.input_request_dict[mode][exchange_type]
                    if token not in tokens_to_remove
                ]
        try:
            request_data = {
                "correlationID": correlation_id,
                "action": self.UNSUBSCRIBE_ACTION,
                "params": {"mode": mode, "tokenList": token_list},
            }
            self.wsapp.send(json.dumps(request_data))
            self.RESUBSCRIBE_FLAG = True
            sleep(1)
        except Exception as e:
            logger.error(f"Price websocket error occurred during unsubscribe: {e}")
            raise e

        # Remove unsubscribed tokens from data_bank
        for token_dict in token_list:
            tokens_to_remove = token_dict["tokens"]
            for token in tokens_to_remove:
                self.data_bank.pop(token, None)

    @processing_subscriptions
    def resubscribe(self):
        try:
            for key, val in self.input_request_dict.items():
                token_list = []
                for key1, val1 in val.items():
                    temp_data = {"exchangeType": key1, "tokens": val1}
                    token_list.append(temp_data)
                request_data = {
                    "action": self.SUBSCRIBE_ACTION,
                    "params": {"mode": key, "tokenList": token_list},
                }
                self.wsapp.send(json.dumps(request_data))
                sleep(1)
        except Exception as e:
            logger.error(f"{self.websocket_type} resubscribe error: {e}")
            raise e

    def _parse_binary_data(self, binary_data):
        parsed_data = {
            "subscription_mode": self._unpack_data(binary_data, 0, 1, byte_format="B")[
                0
            ],
            "exchange_type": self._unpack_data(binary_data, 1, 2, byte_format="B")[0],
            "token": self._parse_token_value(binary_data[2:27]),
            "sequence_number": self._unpack_data(binary_data, 27, 35, byte_format="q")[
                0
            ],
            "exchange_timestamp": self._unpack_data(
                binary_data, 35, 43, byte_format="q"
            )[0],
            "last_traded_price": self._unpack_data(
                binary_data, 43, 51, byte_format="q"
            )[0],
        }
        try:
            parsed_data["subscription_mode_val"] = self.SUBSCRIPTION_MODE_MAP.get(
                parsed_data["subscription_mode"]
            )

            if parsed_data["subscription_mode"] in [self.QUOTE, self.SNAP_QUOTE]:
                parsed_data["last_traded_quantity"] = self._unpack_data(
                    binary_data, 51, 59, byte_format="q"
                )[0]
                parsed_data["average_traded_price"] = self._unpack_data(
                    binary_data, 59, 67, byte_format="q"
                )[0]
                parsed_data["volume_trade_for_the_day"] = self._unpack_data(
                    binary_data, 67, 75, byte_format="q"
                )[0]
                parsed_data["total_buy_quantity"] = self._unpack_data(
                    binary_data, 75, 83, byte_format="d"
                )[0]
                parsed_data["total_sell_quantity"] = self._unpack_data(
                    binary_data, 83, 91, byte_format="d"
                )[0]
                parsed_data["open_price_of_the_day"] = self._unpack_data(
                    binary_data, 91, 99, byte_format="q"
                )[0]
                parsed_data["high_price_of_the_day"] = self._unpack_data(
                    binary_data, 99, 107, byte_format="q"
                )[0]
                parsed_data["low_price_of_the_day"] = self._unpack_data(
                    binary_data, 107, 115, byte_format="q"
                )[0]
                parsed_data["closed_price"] = self._unpack_data(
                    binary_data, 115, 123, byte_format="q"
                )[0]

            if parsed_data["subscription_mode"] == self.SNAP_QUOTE:
                parsed_data["last_traded_timestamp"] = self._unpack_data(
                    binary_data, 123, 131, byte_format="q"
                )[0]
                parsed_data["open_interest"] = self._unpack_data(
                    binary_data, 131, 139, byte_format="q"
                )[0]
                parsed_data["open_interest_change_percentage"] = self._unpack_data(
                    binary_data, 139, 147, byte_format="q"
                )[0]
                parsed_data["upper_circuit_limit"] = self._unpack_data(
                    binary_data, 347, 355, byte_format="q"
                )[0]
                parsed_data["lower_circuit_limit"] = self._unpack_data(
                    binary_data, 355, 363, byte_format="q"
                )[0]
                parsed_data["52_week_high_price"] = self._unpack_data(
                    binary_data, 363, 371, byte_format="q"
                )[0]
                parsed_data["52_week_low_price"] = self._unpack_data(
                    binary_data, 371, 379, byte_format="q"
                )[0]
                best_5_buy_and_sell_data = self._parse_best_5_buy_and_sell_data(
                    binary_data[147:347]
                )
                parsed_data["best_5_buy_data"] = best_5_buy_and_sell_data[
                    "best_5_sell_data"
                ]
                parsed_data["best_5_sell_data"] = best_5_buy_and_sell_data[
                    "best_5_buy_data"
                ]

            if parsed_data["subscription_mode"] == self.DEPTH:
                parsed_data.pop("sequence_number", None)
                parsed_data.pop("last_traded_price", None)
                parsed_data.pop("subscription_mode_val", None)
                parsed_data["packet_received_time"] = self._unpack_data(
                    binary_data, 35, 43, byte_format="q"
                )[0]
                depth_data_start_index = 43
                depth_20_data = self._parse_depth_20_buy_and_sell_data(
                    binary_data[depth_data_start_index:]
                )
                parsed_data["depth_20_buy_data"] = depth_20_data["depth_20_buy_data"]
                parsed_data["depth_20_sell_data"] = depth_20_data["depth_20_sell_data"]

            return parsed_data
        except Exception as e:
            logger.error(
                f"Price websocket error occurred during binary data parsing: {e}"
            )
            raise e

    def _unpack_data(self, binary_data, start, end, byte_format="I"):
        """
        Unpack Binary Data to the integer according to the specified byte_format.
        This function returns the tuple
        """
        return struct.unpack(
            self.LITTLE_ENDIAN_BYTE_ORDER + byte_format, binary_data[start:end]
        )

    @staticmethod
    def _parse_token_value(binary_packet):
        token = ""
        for i in range(len(binary_packet)):
            if chr(binary_packet[i]) == "\x00":
                return token
            token += chr(binary_packet[i])
        return token

    def _parse_best_5_buy_and_sell_data(self, binary_data):
        def split_packets(binary_packets):
            packets = []

            i = 0
            while i < len(binary_packets):
                packets.append(binary_packets[i : i + 20])
                i += 20
            return packets

        best_5_buy_sell_packets = split_packets(binary_data)

        best_5_buy_data = []
        best_5_sell_data = []

        for packet in best_5_buy_sell_packets:
            each_data = {
                "flag": self._unpack_data(packet, 0, 2, byte_format="H")[0],
                "quantity": self._unpack_data(packet, 2, 10, byte_format="q")[0],
                "price": self._unpack_data(packet, 10, 18, byte_format="q")[0],
                "no of orders": self._unpack_data(packet, 18, 20, byte_format="H")[0],
            }

            if each_data["flag"] == 0:
                best_5_buy_data.append(each_data)
            else:
                best_5_sell_data.append(each_data)

        return {
            "best_5_buy_data": best_5_buy_data,
            "best_5_sell_data": best_5_sell_data,
        }

    def _parse_depth_20_buy_and_sell_data(self, binary_data):
        depth_20_buy_data = []
        depth_20_sell_data = []

        for i in range(20):
            buy_start_idx = i * 10
            sell_start_idx = 200 + i * 10

            # Parse buy data
            buy_packet_data = {
                "quantity": self._unpack_data(
                    binary_data, buy_start_idx, buy_start_idx + 4, byte_format="i"
                )[0],
                "price": self._unpack_data(
                    binary_data, buy_start_idx + 4, buy_start_idx + 8, byte_format="i"
                )[0],
                "num_of_orders": self._unpack_data(
                    binary_data, buy_start_idx + 8, buy_start_idx + 10, byte_format="h"
                )[0],
            }

            # Parse sell data
            sell_packet_data = {
                "quantity": self._unpack_data(
                    binary_data, sell_start_idx, sell_start_idx + 4, byte_format="i"
                )[0],
                "price": self._unpack_data(
                    binary_data, sell_start_idx + 4, sell_start_idx + 8, byte_format="i"
                )[0],
                "num_of_orders": self._unpack_data(
                    binary_data,
                    sell_start_idx + 8,
                    sell_start_idx + 10,
                    byte_format="h",
                )[0],
            }

            depth_20_buy_data.append(buy_packet_data)
            depth_20_sell_data.append(sell_packet_data)

        return {
            "depth_20_buy_data": depth_20_buy_data,
            "depth_20_sell_data": depth_20_sell_data,
        }

    def parse_price_dict(self):
        new_price_dict = {
            token_symbol_dict[token]: value for token, value in self.data_bank.items()
        }
        return new_price_dict

    def get_active_subscriptions(self, options_only=True) -> dict[int, list[str]]:
        active_subscriptions = defaultdict(list)
        for mode, exchange_subscriptions in self.input_request_dict.items():
            for exchange, tokens in exchange_subscriptions.items():
                if options_only and exchange in [1, 3]:
                    continue
                active_subscriptions[mode].extend(tokens)
        return dict(active_subscriptions)

    def get_active_strike_range(
        self, underlying, range_of_strikes: int = None
    ) -> list[int]:
        range_of_strikes = (
            self.default_strike_range if range_of_strikes is None else range_of_strikes
        )
        underlying_ltp = self.data_bank[underlying.token]["ltp"]
        return underlying.get_active_strikes(range_of_strikes, ltp=underlying_ltp)

    @staticmethod
    def _get_tokens_for_strike_expiry(name: str, strike: int, expiry: str):
        try:
            _, call_token = get_symbol_token(name, expiry, strike, "CE")
        except Exception as e:
            logger.error(
                f"Error in fetching call token for {strike, expiry} for {name}: {e}"
            )
            call_token = "abc"
        try:
            _, put_token = get_symbol_token(name, expiry, strike, "PE")
        except Exception as e:
            logger.error(
                f"Error in fetching put token for {strike, expiry} for {name}: {e}"
            )
            put_token = "abc"
        return call_token, put_token

    def _prepare_subscription_dict(
        self,
        underlying,
        strike_range: list[int],
    ) -> dict[int, list[str]]:
        subscription_dict = defaultdict(list)
        expiry_sub_modes = {
            underlying.current_expiry: 3,
            underlying.next_expiry: 1,
            underlying.far_expiry: 1,
        }
        for expiry, mode in expiry_sub_modes.items():
            for strike in strike_range:
                call_token, put_token = self._get_tokens_for_strike_expiry(
                    underlying.name, strike, expiry
                )
                subscription_dict[mode].append(call_token)
                subscription_dict[mode].append(put_token)
        return dict(subscription_dict)

    def subscribe_indices(self):
        self.subscribe(["99926000", "99926009", "99926037", "99926074", "99919000"], 1)

    def subscribe_options(self, *underlyings, range_of_strikes: int = None):
        for underlying in underlyings:
            strike_range = self.get_active_strike_range(underlying, range_of_strikes)
            subscription_dict = self._prepare_subscription_dict(
                underlying, strike_range=strike_range
            )
            for mode, tokens in subscription_dict.items():
                self.subscribe(tokens, mode)
            self.underlying_options_subscribed[underlying] = strike_range
        self.subscribed_to_options = True

    def update_strike_range(self):
        for underlying in self.underlying_options_subscribed.keys():
            refreshed_strike_range = self.get_active_strike_range(underlying)
            current_strike_range = self.underlying_options_subscribed[underlying]

            if not strike_range_different(refreshed_strike_range, current_strike_range):
                continue

            new_strikes = set(refreshed_strike_range) - set(current_strike_range)
            obsolete_strikes = set(current_strike_range) - set(refreshed_strike_range)
            if len(new_strikes) >= 0.4 * len(current_strike_range):  # Hardcoded 40%
                logger.info(
                    f"New strike range for {underlying.name}: {refreshed_strike_range}. "
                    f"Old strike range: {current_strike_range}."
                )

                subscription_dict = self._prepare_subscription_dict(
                    underlying, strike_range=list(new_strikes)
                )
                unsubscription_dict = self._prepare_subscription_dict(
                    underlying, strike_range=list(obsolete_strikes)
                )
                for mode in subscription_dict:
                    self.subscribe(subscription_dict[mode], mode)
                for mode in unsubscription_dict:
                    self.unsubscribe(unsubscription_dict[mode], mode)

                # Updating the strike range in the underlying_options_subscribed dict
                self.underlying_options_subscribed[underlying] = refreshed_strike_range

                all_tokens_subscribed = list(
                    itertools.chain(*subscription_dict.values())
                )
                all_tokens_unsubscribed = list(
                    itertools.chain(*unsubscription_dict.values())
                )
                all_symbols_subscribed = [
                    token_symbol_dict[token] for token in all_tokens_subscribed
                ]
                all_symbols_unsubscribed = [
                    token_symbol_dict[token] for token in all_tokens_unsubscribed
                ]
                logger.debug(
                    f"{underlying.name} subscribed to: {all_symbols_subscribed} "
                    f"and unsubscribed from: {all_symbols_unsubscribed}"
                )

    def periodically_execute_command_queue(self):
        while True and not self.intentionally_closed.value:
            try:
                command = self.command_queue.get()
                if command == "subscribe_indices":
                    self.subscribe_indices()
                elif isinstance(command, tuple):
                    command, args = command
                    if command == "subscribe_options":
                        underlyings: list = args["underlyings"]
                        range_of_strikes = args["range_of_strikes"]
                        self.subscribe_options(
                            *underlyings, range_of_strikes=range_of_strikes
                        )
                elif command == "close_connection":
                    self.close_connection()
            except Exception as e:
                logger.error(f"Error in executing command queue: {e}")
            sleep(0.1)

    def check_connection_freshness(self):
        # This method will update the connection_stale flag. The idea is to check the
        # Freshness based on data in teh databank rather than the connection status.
        if self.data_bank:
            try:
                time_now = current_time()
                most_recent_timestamp = max(
                    [value["timestamp"] for value in self.data_bank.values()]
                )
                if time_now - most_recent_timestamp > timedelta(seconds=5):
                    self.connection_stale.value = True
                else:
                    self.connection_stale.value = False
            except Exception as e:
                logger.error(f"Error in checking freshness of data: {e}")

    @log_error(raise_error=True)
    def periodically_update_strike_range(self):
        while True and not self.intentionally_closed.value:
            if not self.reconnecting:
                self.update_strike_range()
            sleep(5)
