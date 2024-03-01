import numpy as np
from collections import defaultdict
from time import sleep
from collections import deque
import functools
from datetime import datetime
import pandas as pd
import threading
from SmartApi.smartExceptions import DataException
from typing import Callable
from multiprocessing import Manager, Process
from autotradr.exceptions import APIFetchError
from autotradr.utils import current_time
from autotradr.config import logger, token_exchange_dict, latency_logger, thread_local
from autotradr.decorators import (
    classproperty,
    timeit,
)
from autotradr.angel_interface.active_session import ActiveSession
from autotradr.angel_interface.login import wait_for_login
from autotradr.angel_interface.order_websocket import OrderWebsocket
from autotradr.angel_interface.price_websocket import PriceWebsocket


class LiveFeeds:
    price_feed: PriceWebsocket = None
    order_feed: OrderWebsocket = None
    price_feed_process = None
    order_feed_process = None

    @classproperty
    def order_book(self) -> list:
        return list(self.order_feed.data_bank.values())

    @classmethod
    def order_feed_connected(cls):
        return cls.order_feed is not None and cls.order_feed.connected.value

    @classmethod
    def price_feed_connected(cls):
        return (
            cls.price_feed is not None
            and cls.price_feed.connected.value
            and not cls.price_feed.connection_stale.value
        )

    @classmethod
    def close_feeds(cls):
        try:
            if cls.price_feed is not None and cls.price_feed_process is not None:
                cls.price_feed.command_queue.put("close_connection")
                cls.price_feed_process.terminate()
            if cls.order_feed is not None:
                cls.order_feed.command_queue.put("close_connection")
                cls.order_feed_process.terminate()
        except Exception as e:
            logger.error(f"Error while closing live feeds: {e}")

    @classmethod
    @wait_for_login
    def start_price_feed(cls, manager: Manager):
        pf = PriceWebsocket.from_active_session(manager=manager)
        process = Process(target=pf.connect)
        cls.price_feed_process = process
        cls.price_feed = pf
        process.start()
        while not pf.connected.value:
            logger.info("Waiting for price feed to connect...")
            sleep(2)
        pf.command_queue.put("subscribe_indices")

    @classmethod
    @wait_for_login
    def start_order_feed(cls, manager: Manager):
        of = OrderWebsocket.from_active_session(manager)
        process = Process(target=of.connect)
        cls.order_feed_process = process
        cls.order_feed = of
        process.start()


# A class based implementation for educating myself on decorators
class AccessRateHandler:
    def __init__(self, delay=1):
        self.delay = delay + 0.1  # Add a small buffer to the delay
        self.last_call_time = datetime(
            1997, 12, 30
        )  # A date with an interesting trivia in the field of CS

    def __call__(self, func):
        def wrapped(*args, **kwargs):
            time_since_last_call = (
                current_time() - self.last_call_time
            ).total_seconds()
            if time_since_last_call < self.delay:
                sleep(self.delay - time_since_last_call)
            result = func(*args, **kwargs)
            self.last_call_time = current_time()
            return result

        return wrapped


def _access_rate_handler_rolling(*, max_requests, per_seconds):
    lock = threading.Lock()
    request_times = deque(maxlen=max_requests)

    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            with lock:
                time_now = current_time()

                # If queue is full, check if we need to wait
                if (
                    len(request_times) == max_requests
                    and (time_now - request_times[0]).total_seconds() < per_seconds
                ):
                    sleep(per_seconds - (time_now - request_times[0]).total_seconds())
                try:
                    result = func(*args, **kwargs)
                finally:
                    request_times.append(current_time())

            return result

        return wrapped

    return decorator


def _access_rate_handler_static(*, delay: float):
    last_call_time = datetime(1997, 12, 30)  # An interesting trivia date in CS
    lock = threading.Lock()

    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            nonlocal last_call_time
            with lock:
                time_since_last_call = (current_time() - last_call_time).total_seconds()
                if time_since_last_call < delay:
                    sleep(delay - time_since_last_call)
                try:
                    result = func(*args, **kwargs)
                finally:
                    last_call_time = current_time()
            return result

        return wrapped

    return decorator


def access_rate_handler(kind: str, **kwargs):
    if kind == "static":
        return _access_rate_handler_static(**kwargs)
    elif kind == "rolling":
        return _access_rate_handler_rolling(**kwargs)
    else:
        raise ValueError(f"Invalid access rate handler type '{kind}'.")


def retry_angel_api(
    data_type: str | Callable = None,
    max_attempts: int = 10,
    wait_increase_factor: float = 1.1,
):
    def handle_retry_logic(
        function: str,
        attempt: int,
        exception: Exception,
        msg: str,
        additional_msg: str,
    ) -> int | None:
        if should_raise_exception(exception, attempt):
            logger.error(f"{msg}. Additional info: {additional_msg}")
            if not isinstance(exception, DataException) and not isinstance(
                exception, ValueError
            ):
                raise APIFetchError(msg)
            else:
                raise exception

        if (
            getattr(thread_local, "robust_handling", False)
            and attempt == max_attempts - 2
        ):
            logger.warning(f"Entering robust handling mode for function {function}.")
            seconds_to_day_end: int = (
                datetime(
                    *current_time().date().timetuple()[:3],
                    hour=15,
                    minute=29,
                )
                - current_time()
            ).seconds
            max_sleep = max(min(60, seconds_to_day_end // 4), 1)
            return max_sleep

    def should_raise_exception(exception, attempt):
        if isinstance(exception, ValueError) and "Invalid book type" in str(exception):
            return True
        else:
            return attempt == max_attempts

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sleep_duration = 1
            data = {}
            for attempt in range(1, max_attempts + 1):
                try:
                    data = func(*args, **kwargs)
                    if callable(data_type):
                        return data_type(data)
                    if data_type == "ltp":
                        return data["data"]["ltp"]
                    return data["data"]

                except Exception as e:
                    function = func.__name__
                    msg = f"Attempt {attempt}: Error in function {function}: {e}"
                    additional_msg = (
                        data.get("message", "No additional message available")
                        if isinstance(data, dict)
                        else ""
                    )
                    custom_sleep = handle_retry_logic(
                        function,
                        attempt,
                        e,
                        msg,
                        additional_msg,
                    )
                    sleep_duration = (
                        custom_sleep or sleep_duration * wait_increase_factor
                    )
                    logger.info(
                        f"{msg}. Additional info: {additional_msg}. Retrying in {sleep_duration} seconds."
                    )
                    sleep(sleep_duration)

        return wrapper

    return decorator


def increase_robustness(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        thread_local.robust_handling = True
        try:
            return func(*args, **kwargs)
        finally:
            thread_local.robust_handling = False

    return wrapper


@retry_angel_api(data_type=lambda x: x)
@access_rate_handler("rolling", max_requests=20, per_seconds=1.05)
@timeit(logger=latency_logger)
def _place_order(params: dict) -> str:
    return ActiveSession.obj.placeOrder(params)


@retry_angel_api(data_type=lambda x: None)
@access_rate_handler("rolling", max_requests=20, per_seconds=1.05)
@timeit(logger=latency_logger)
def _modify_order(params: dict) -> None:
    return ActiveSession.obj.modifyOrder(params)


@retry_angel_api(data_type=lambda x: x["data"]["fetched"])
@access_rate_handler("static", delay=1.05)
@timeit(logger=latency_logger)
def _fetch_quotes(tokens: list, mode: str = "FULL"):
    payload = defaultdict(list)
    for token in tokens:
        exchange = token_exchange_dict.get(token)
        if exchange:
            payload[exchange].append(token)
    payload = dict(payload)
    return ActiveSession.obj.market_data(mode, payload)


def fetch_quotes(tokens: list, mode: str = "FULL", structure: str = "list"):
    quote_data = _fetch_quotes(tokens, mode)

    if structure.lower() == "dict":
        return {entry["symbolToken"]: entry for entry in quote_data}
    elif structure.lower() == "list":
        return quote_data
    else:
        raise ValueError(f"Invalid structure '{structure}'.")


@retry_angel_api(data_type="ltp")
@access_rate_handler("rolling", max_requests=10, per_seconds=1.05)
@timeit(logger=latency_logger)
def _fetch_ltp(exchange_seg, symbol, token):
    price_data = ActiveSession.obj.ltpData(exchange_seg, symbol, token)
    return price_data


def fetch_ltp(exchange_seg, symbol, token, field="ltp"):
    if (
        LiveFeeds.price_feed_connected()
        and token in LiveFeeds.price_feed.data_bank.copy()
    ):
        price = LiveFeeds.price_feed.data_bank[token][field]
    else:
        price = _fetch_ltp(exchange_seg, symbol, token)
    return price


@retry_angel_api(max_attempts=10)
@access_rate_handler("static", delay=1.05)
@timeit(logger=latency_logger)
def _fetch_book(fetch_func):
    data = fetch_func()
    return data


def fetch_book(book: str, from_api: bool = False) -> list:
    if book == "orderbook":
        if LiveFeeds.order_feed_connected() and not from_api:
            return LiveFeeds.order_book
        return _fetch_book(ActiveSession.obj.orderBook)
    elif book in {"positions", "position"}:
        return _fetch_book(ActiveSession.obj.position)
    else:
        raise ValueError(f"Invalid book type '{book}'.")


def lookup_and_return(
    book, field_to_lookup, value_to_lookup, field_to_return=None
) -> np.ndarray | dict:
    def filter_and_return(data: list):
        if not isinstance(field_to_lookup, (list, tuple, np.ndarray)):
            field_to_lookup_ = [field_to_lookup]
            value_to_lookup_ = [value_to_lookup]
        else:
            field_to_lookup_ = field_to_lookup
            value_to_lookup_ = value_to_lookup

        if field_to_return is None:  # Return the entire entry
            return np.array(
                [
                    entry
                    for entry in data
                    if all(
                        (
                            entry[field] == value
                            if not isinstance(value, (list, tuple, np.ndarray))
                            else entry[field] in value
                        )
                        for field, value in zip(field_to_lookup_, value_to_lookup_)
                    )
                    and all(entry[field] != "" for field in field_to_lookup_)
                ]
            )

        elif isinstance(
            field_to_return, (list, tuple, np.ndarray)
        ):  # multiple fields are requested
            bucket = []
            for entry in data:
                if all(
                    (
                        entry[field] == value
                        if not isinstance(value, (list, tuple, np.ndarray))
                        else entry[field] in value
                    )
                    for field, value in zip(field_to_lookup_, value_to_lookup_)
                ) and all(entry[field] != "" for field in field_to_lookup_):
                    bucket.append({field: entry[field] for field in field_to_return})
            if len(bucket) == 0:
                return np.array([])
            else:
                return np.array(bucket)
        else:  # Return a numpy array as only one field is requested
            # Check if 'orderid' is in field_to_lookup_
            if "orderid" in field_to_lookup_:
                sort_by_orderid = True
                orderid_index = field_to_lookup_.index("orderid")
            else:
                sort_by_orderid = False
                orderid_index = None

            bucket = [
                (entry["orderid"], entry[field_to_return])
                if sort_by_orderid
                else entry[field_to_return]
                for entry in data
                if all(
                    (
                        entry[field] == value
                        if not isinstance(value, (list, tuple, np.ndarray))
                        else entry[field] in value
                    )
                    for field, value in zip(field_to_lookup_, value_to_lookup_)
                )
                and all(entry[field] != "" for field in field_to_lookup_)
            ]

            if len(bucket) == 0:
                return np.array([])
            else:
                if sort_by_orderid:
                    # Create a dict mapping order ids to their index in value_to_lookup
                    orderid_to_index = {
                        value: index
                        for index, value in enumerate(value_to_lookup_[orderid_index])
                    }
                    # Sort the bucket based on the order of 'orderid' in value_to_lookup
                    bucket.sort(key=lambda x: orderid_to_index[x[0]])
                    # Return only the field_to_return values
                    return np.array([x[1] for x in bucket])
                else:
                    return np.array(bucket)

    if not (
        isinstance(field_to_lookup, (str, list, tuple, np.ndarray))
        and isinstance(value_to_lookup, (str, list, tuple, np.ndarray))
    ):
        raise ValueError(
            "Both 'field_to_lookup' and 'value_to_lookup' must be strings or lists."
        )

    if isinstance(field_to_lookup, list) and isinstance(value_to_lookup, str):
        raise ValueError(
            "Unsupported input: 'field_to_lookup' is a list and 'value_to_lookup' is a string."
        )

    if isinstance(book, list):
        return filter_and_return(book)
    elif isinstance(book, str) and book in {"orderbook", "positions"}:
        book_data = fetch_book(book)
        return filter_and_return(book_data)
    else:
        logger.error(f"Invalid book type '{book}'.")
        raise ValueError("Invalid book type.")


@retry_angel_api()
@access_rate_handler("rolling", max_requests=3, per_seconds=1.05)
def fetch_historical_prices(
    token: str, interval: str, from_date: datetime, to_date: datetime
):
    from_date = pd.to_datetime(from_date) if isinstance(from_date, str) else from_date
    to_date = pd.to_datetime(to_date) if isinstance(to_date, str) else to_date
    exchange = token_exchange_dict[token]
    historic_param = {
        "exchange": exchange,
        "symboltoken": token,
        "interval": interval,
        "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
        "todate": to_date.strftime("%Y-%m-%d %H:%M"),
    }
    return ActiveSession.obj.getCandleData(historic_param)
