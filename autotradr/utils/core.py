import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, time
import re
from autotradr.config import holidays, logger
from autotradr import config


def word_to_num(s):
    word = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }
    multiplier = {
        "thousand": 1000,
        "hundred": 100,
        "million": 1000000,
        "billion": 1000000000,
    }

    words = s.lower().split()
    if words[0] == "a":
        words[0] = "one"
    total = 0
    current = 0
    for w in words:
        if w in word:
            current += word[w]
        if w in multiplier:
            current *= multiplier[w]
        if w == "and":
            continue
        if w == "thousand" or w == "million" or w == "billion":
            total += current
            current = 0
    total += current
    return total


def current_time():
    # Adjusting for timezones
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).replace(tzinfo=None)


def market_hours():
    if time(9, 15) <= current_time().time() <= time(15, 30):
        return True
    else:
        return False


def last_market_close_time():
    if current_time().time() < time(9, 15):
        wip_time = current_time() - timedelta(days=1)
        wip_time = wip_time.replace(hour=15, minute=30, second=0, microsecond=0)
    elif current_time().time() > time(15, 30):
        wip_time = current_time().replace(hour=15, minute=30, second=0, microsecond=0)
    else:
        wip_time = current_time()

    if wip_time.weekday() not in [5, 6] and wip_time.date() not in holidays:
        return wip_time
    else:
        # Handling weekends and holidays
        while wip_time.weekday() in [5, 6] or wip_time.date() in holidays:
            wip_time = wip_time - timedelta(days=1)

    last_close_day_time = wip_time.replace(hour=15, minute=30, second=0, microsecond=0)
    return last_close_day_time


def find_strike(x, base):
    number = base * round(x / base)
    return int(number)


def custom_round(x, base=0.05):
    """Used in place_order function to round off the price to the nearest 0.05"""
    if x == 0:
        return 0

    num = base * round(x / base)
    if num == 0:
        num = base
    return round(num, 2)


def round_to_nearest(x, digits=2):
    if x is None or x == 0 or np.isnan(x):
        return np.nan
    return round(x, digits)


def find_strike_with_offset(
    underlying_ltp: float,
    offset: float,
    base: float,
):
    strike = find_strike((underlying_ltp * (1 + offset)), base)
    return strike


def splice_orders(quantity_in_lots, freeze_qty):
    if quantity_in_lots > freeze_qty:
        loops = int(quantity_in_lots / freeze_qty)
        if loops > config.LARGE_ORDER_THRESHOLD:
            raise Exception(
                "Order too big. This error was raised to prevent accidental large order placement."
            )

        remainder = quantity_in_lots % freeze_qty
        if remainder == 0:
            spliced_orders = [freeze_qty] * loops
        else:
            spliced_orders = [freeze_qty] * loops + [remainder]
    else:
        spliced_orders = [quantity_in_lots]
    return spliced_orders


def time_to_expiry(
    expiry: str, effective_time: bool = False, in_days: bool = False
) -> float:
    """Return time left to expiry"""
    if in_days:
        multiplier = 365
    else:
        multiplier = 1

    expiry = datetime.strptime(expiry, "%d%b%y")
    time_left_to_expiry = (
        (expiry + pd.DateOffset(minutes=930)) - current_time()
    ) / timedelta(days=365)

    # Subtracting holidays and weekends
    if effective_time:
        date_range = pd.date_range(current_time().date(), expiry - timedelta(days=1))
        numer_of_weekdays = sum(date_range.dayofweek > 4)
        number_of_holidays = sum(date_range.isin(holidays))
        time_left_to_expiry -= (numer_of_weekdays + number_of_holidays) / 365
    return round(time_left_to_expiry * multiplier, 5)


def strike_range_different(
    refreshed_strike_range: list[int | float], current_strike_range: list[int | float]
) -> bool:
    if set(refreshed_strike_range) == set(current_strike_range):
        return False
    new_strikes = set(refreshed_strike_range) - set(current_strike_range)
    if len(new_strikes) >= 0.4 * len(current_strike_range):  # Hardcoded 40%
        return True
    return False


def round_shares_to_lot_size(shares, lot_size):
    number = lot_size * round(shares / lot_size)
    return int(number)


def convert_exposure_to_lots(
    exposure: int | float, spot_price: float, lot_size: int, round_to: int = None
) -> int:
    shares = round_shares_to_lot_size(exposure / spot_price, lot_size)
    lots = shares / lot_size
    if round_to is not None:
        lots = custom_round(lots, round_to)
        # Return at least 1 lot
    return max(1, int(lots))


def check_for_weekend(expiry: str) -> bool:
    expiry = datetime.strptime(expiry, "%d%b%y")
    expiry = expiry + pd.DateOffset(minutes=930)
    date_range = pd.date_range(current_time().date(), expiry - timedelta(days=1))
    return date_range.weekday.isin([5, 6]).any()


def find_next_trading_day(date: datetime | str = None) -> datetime:
    if date is None:
        date = current_time().date()
    elif isinstance(date, str):
        date = pd.to_datetime(date)
        date = date.date()
    elif isinstance(date, datetime):
        date = date.date()
    date = date + timedelta(days=1)
    while date.weekday() in [5, 6] or date in holidays:
        date = date + timedelta(days=1)
    return date


def calculate_ema(new_price, prev_ema, alpha):
    """
    Calculate Exponential Moving Average (EMA)

    Parameters:
    - new_price (float): The new price data point
    - prev_ema (float or None): The previous EMA value, or None if not calculated yet
    - alpha (float): The smoothing factor

    Returns:
    float: The new EMA value
    """
    if prev_ema is None:
        return new_price
    return (new_price * alpha) + (prev_ema * (1 - alpha))


def spot_price_from_future(future_price, interest_rate, time_to_future):
    """
    Calculate the spot price from the future price, interest rate, and time.

    :param future_price: float, the future price of the asset
    :param interest_rate: float, the annual interest rate (as a decimal, e.g., 0.05 for 5%)
    :param time_to_future: float, the time to maturity (in years)
    :return: float, the spot price of the asset
    """
    spot_price = future_price / ((1 + interest_rate) ** time_to_future)
    return spot_price


def charges(buy_premium, contract_size, num_contracts, freeze_quantity=None):
    if freeze_quantity:
        number_of_orders = np.ceil(num_contracts / freeze_quantity)
    else:
        number_of_orders = 1

    buy_brokerage = 40 * number_of_orders
    sell_brokerage = 40 * number_of_orders
    transaction_charge_rate = 0.05 / 100
    stt_ctt_rate = 0.0625 / 100
    gst_rate = 18 / 100

    buy_transaction_charges = (
        buy_premium * contract_size * num_contracts * transaction_charge_rate
    )
    sell_transaction_charges = (
        buy_premium * contract_size * num_contracts * transaction_charge_rate
    )
    stt_ctt = buy_premium * contract_size * num_contracts * stt_ctt_rate

    buy_gst = (buy_brokerage + buy_transaction_charges) * gst_rate
    sell_gst = (sell_brokerage + sell_transaction_charges) * gst_rate

    total_charges = (
        buy_brokerage
        + sell_brokerage
        + buy_transaction_charges
        + sell_transaction_charges
        + stt_ctt
        + buy_gst
        + sell_gst
    )
    charges_per_share = total_charges / (num_contracts * contract_size)

    return round(charges_per_share, 1)


def parse_symbol(symbol):
    match = re.match(r"([A-Za-z]+)(\d{2}[A-Za-z]{3}\d{2})(\d+)(\w+)", symbol)
    if match:
        return match.groups()
    return None


def get_background_tasks(obj: object, task_name: str):
    """obj is the instance of the class where the tasks are defined"""
    parallel_tasks = [attr for attr in dir(obj) if attr.startswith(task_name)]
    parallel_tasks = [
        getattr(obj, attr) for attr in parallel_tasks if callable(getattr(obj, attr))
    ]
    return parallel_tasks


def filter_orderbook_by_time(
    orderbook: list[dict], start_time: datetime = None, end_time: datetime = None
) -> list[dict]:
    def check_eligibility(order):
        for field in ["updatetime", "exchtime", "exchorderupdatetime"]:
            try:
                return (
                    start_time
                    < datetime.strptime(order.get(field), "%d-%b-%Y %H:%M:%S")
                    < end_time
                )
            except Exception as e:
                logger.error(
                    f"Error in filter_orderbook_by_time for order {order}: {e}"
                )
                continue
        return False

    if start_time is None:
        start_time = datetime.now() - timedelta(days=1)
    elif isinstance(start_time, str):
        start_time = pd.to_datetime(start_time, infer_datetime_format=True)
    if end_time is None:
        end_time = datetime.now() + timedelta(days=1)
    elif isinstance(end_time, str):
        end_time = pd.to_datetime(end_time, infer_datetime_format=True)
    filtered_orders = [order for order in orderbook if check_eligibility(order)]
    return filtered_orders
