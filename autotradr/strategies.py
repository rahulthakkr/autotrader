import numpy as np
from time import sleep
from typing import Optional
from threading import Thread
from datetime import timedelta, time
from autotradr.config import logger
from autotradr.utils.core import (
    current_time,
    find_strike,
    time_to_expiry,
    calculate_ema,
    convert_exposure_to_lots,
)
from autotradr.utils.communication import notifier, log_error
from autotradr.exceptions import IntrinsicValueError
from autotradr.blackscholes import calculate_strangle_iv
from autotradr.angel_interface.interface import (
    fetch_book,
    lookup_and_return,
    increase_robustness,
)
from autotradr.trade_interface import (
    Strangle,
    Action,
    Index,
    Stock,
    place_option_order_and_notify,
    cancel_pending_orders,
)


def most_equal_strangle():
    pass


def execute_instructions():
    pass


def simulate_price():
    pass


def process_stop_loss_order_statuses():
    pass


def intraday_strangle(
        underlying: Index | Stock,
        exposure: int | float,
        call_strike_offset: Optional[float] = 0,
        put_strike_offset: Optional[float] = 0,
        strike_selection: Optional[str] = "equal",
        stop_loss: Optional[float | str] = "dynamic",
        call_stop_loss: Optional[float] = None,
        put_stop_loss: Optional[float] = None,
        combined_stop_loss: Optional[float] = None,
        exit_time: tuple[int, int] = (15, 29),
        sleep_time: Optional[int] = 5,
        seconds_to_avg: Optional[int] = 30,
        simulation_safe_guard: Optional[float] = 1.15,
        catch_trend: Optional[bool] = False,
        trend_qty_ratio: Optional[float] = 1,
        disparity_threshold: Optional[float] = 1000,
        place_sl_orders: Optional[bool] = False,
        move_sl_to_cost: Optional[bool] = False,
        place_orders_on_sl: Optional[bool] = False,
        convert_to_butterfly: Optional[bool] = False,
        conversion_method: Optional[str] = "pct",
        conversion_threshold_pct: Optional[float] = 0.175,
        take_profit: Optional[float] = 0,
        notification_url: Optional[str] = None,
        strategy_tag: Optional[str] = "Intraday strangle",
):
    """Intraday strangle strategy. Trades strangle with stop loss. All offsets are in percentage terms.
    Parameters
    ----------
    underlying : Index | Stock
        Underlying object
    exposure : int | float
        Exposure in rupees
    strike_selection : str, optional {'equal', 'resilient', 'atm'}
        Mode for finding the strangle, by default 'equal'
    call_strike_offset : float, optional
        Call strike offset in percentage terms, by default 0
    put_strike_offset : float, optional
        Put strike offset in percentage terms, by default 0
    stop_loss : float or string, optional
        Stop loss percentage, by default 'dynamic'
    call_stop_loss : float, optional
        Call stop loss percentage, by default None. If None then stop loss is same as stop_loss.
    put_stop_loss : float, optional
        Put stop loss percentage, by default None. If None then stop loss is same as stop_loss.
    combined_stop_loss : float, optional
        Combined stop loss percentage, by default None. If None then individual stop losses are used.
    exit_time : tuple, optional
        Exit time, by default (15, 29)
    sleep_time : int, optional
        Sleep time in seconds for updating prices, by default 5
    seconds_to_avg : int, optional
        Seconds to average prices over, by default 30
    simulation_safe_guard : float, optional
        The multiple over the simulated price that will reject stop loss, by default 1.15
    catch_trend : bool, optional
        Catch trend or not, by default False
    trend_qty_ratio : int, optional
        Ratio of trend quantity to strangle quantity, by default 1
    disparity_threshold : float, optional
        Disparity threshold for equality of strikes, by default np.inf
    place_sl_orders : bool, optional
        Place stop loss orders or not, by default False
    move_sl_to_cost : bool, optional
        Move other stop loss to cost or not, by default False
    place_orders_on_sl : bool, optional
        Place orders on stop loss or not, by default False
    convert_to_butterfly : bool, optional
        Convert to butterfly or not, by default False
    conversion_method : str, optional
        Conversion method for butterfly, by default 'breakeven'
    conversion_threshold_pct : float, optional
        Conversion threshold for butterfly if conversion method is 'pct', by default 0.175
    take_profit : float, optional
        Take profit percentage, by default 0
    notification_url : str, optional
        URL for sending notifications, by default None
    strategy_tag : str, optional
        Strategy tag for logging, by default 'Intraday strangle'
    """

    @log_error(notify=True, raise_error=True)
    @increase_robustness
    def position_monitor(info_dict):
        c_avg_price = info_dict["call_avg_price"]
        p_avg_price = info_dict["put_avg_price"]
        traded_strangle = info_dict["traded_strangle"]

        # EMA parameters
        periods = max(int(seconds_to_avg / sleep_time), 1) if sleep_time >= 1 else 1
        alpha = 2 / (periods + 1)
        ema_values = {
            "call": None,
            "put": None,
            "underlying": None,
        }

        # Conversion to butterfly settings
        ctb_notification_sent = False
        ctb_message = ""
        ctb_hedge = None
        conversion_threshold_break_even = None

        def process_ctb(
                h_strangle: Strangle,
                method: str,
                threshold_break_even: float,
                threshold_pct: float,
                total_price: float,
        ) -> bool:
            hedge_total_ltp = h_strangle.fetch_total_ltp()

            if method == "breakeven":
                hedge_profit = total_price - hedge_total_ltp - underlying.base
                return hedge_profit >= threshold_break_even

            elif method == "pct":
                if (
                        total_price - (hedge_total_ltp + underlying.base)
                        < threshold_break_even
                ):
                    return False  # Ensuring that this is better than break even method
                return hedge_total_ltp <= total_price * threshold_pct

            else:
                raise ValueError(
                    f"Invalid conversion method: {method}. Valid methods are 'breakeven' and 'pct'."
                )

        if convert_to_butterfly:
            ctb_call_strike = traded_strangle.call_strike + underlying.base
            ctb_put_strike = traded_strangle.put_strike - underlying.base
            ctb_hedge = Strangle(
                ctb_call_strike, ctb_put_strike, underlying.name, expiry
            )
            c_sl = call_stop_loss if call_stop_loss is not None else stop_loss
            p_sl = put_stop_loss if put_stop_loss is not None else stop_loss
            profit_if_call_sl = p_avg_price - (c_avg_price * (c_sl - 1))
            profit_if_put_sl = c_avg_price - (p_avg_price * (p_sl - 1))

            conversion_threshold_break_even = max(profit_if_call_sl, profit_if_put_sl)

        threshold_points = (
            (take_profit * (c_avg_price + p_avg_price)) if take_profit > 0 else np.inf
        )

        last_print_time = current_time()
        last_log_time = current_time()
        last_notify_time = current_time()
        print_interval = timedelta(seconds=10)
        log_interval = timedelta(minutes=25)
        notify_interval = timedelta(minutes=180)

        while not info_dict["trade_complete"]:
            # Fetching prices
            spot_price = underlying.fetch_ltp()
            c_ltp, p_ltp = traded_strangle.fetch_ltp()
            info_dict["underlying_ltp"] = spot_price
            info_dict["call_ltp"] = c_ltp
            info_dict["put_ltp"] = p_ltp

            # Calculate EMA for each series
            for series, price in zip(
                    ["call", "put", "underlying"], [c_ltp, p_ltp, spot_price]
            ):
                ema_values[series] = calculate_ema(price, ema_values[series], alpha)

            c_ltp_avg = ema_values["call"]
            p_ltp_avg = ema_values["put"]
            spot_price_avg = ema_values["underlying"]

            info_dict["call_ltp_avg"] = c_ltp_avg
            info_dict["put_ltp_avg"] = p_ltp_avg
            info_dict["underlying_ltp_avg"] = spot_price_avg

            # Combined stop loss detection
            if combined_stop_loss is not None and not np.isnan(combined_stop_loss):
                if (c_ltp_avg + p_ltp_avg) > info_dict["combined_stop_loss_price"]:
                    info_dict["exit_triggers"].update({"combined_stop_loss": True})
                    notifier(
                        f"{underlying.name} Combined stop loss triggered with "
                        f"combined price of {c_ltp_avg + p_ltp_avg}",
                        notification_url,
                        "INFO",
                    )

            # Calculate IV
            call_iv, put_iv, avg_iv = calculate_strangle_iv(
                call_price=c_ltp,
                put_price=p_ltp,
                call_strike=traded_strangle.call_strike,
                put_strike=traded_strangle.put_strike,
                spot=spot_price,
                time_left=time_to_expiry(expiry),
            )
            info_dict["call_iv"] = call_iv
            info_dict["put_iv"] = put_iv
            info_dict["avg_iv"] = avg_iv

            # Calculate mtm price
            call_exit_price = info_dict.get("call_exit_price", c_ltp)
            put_exit_price = info_dict.get("put_exit_price", p_ltp)
            mtm_price = call_exit_price + put_exit_price

            # Calculate profit
            profit_in_pts = (c_avg_price + p_avg_price) - mtm_price
            profit_in_rs = profit_in_pts * underlying.lot_size * quantity_in_lots
            info_dict["profit_in_pts"] = profit_in_pts
            info_dict["profit_in_rs"] = profit_in_rs

            if take_profit > 0:
                if profit_in_pts >= threshold_points:
                    info_dict["exit_triggers"].update({"take_profit": True})
                    notifier(
                        f"{underlying.name} Take profit triggered with profit of {profit_in_pts} points",
                        notification_url,
                        "INFO",
                    )

            # Conversion to butterfly working
            if (
                    not (info_dict["call_sl"] or info_dict["put_sl"])
                    and info_dict["time_left_day_start"] * 365 < 1
                    and convert_to_butterfly
                    and not ctb_notification_sent
                    and current_time().time() < time(14, 15)
            ):
                try:
                    ctb_trigger = process_ctb(
                        ctb_hedge,
                        conversion_method,
                        conversion_threshold_break_even,
                        conversion_threshold_pct,
                        info_dict["total_avg_price"],
                    )
                    if ctb_trigger:
                        notifier(
                            f"{underlying.name} Convert to butterfly triggered\n",
                            notification_url,
                            "INFO",
                        )
                        info_dict["exit_triggers"].update(
                            {"convert_to_butterfly": True}
                        )
                        ctb_message = f"Hedged with: {ctb_hedge}\n"
                        info_dict["ctb_hedge"] = ctb_hedge
                        ctb_notification_sent = True
                except Exception as _e:
                    logger.error(f"Error in process_ctb: {_e}")

            message = (
                    f"\nUnderlying: {underlying.name}\n"
                    f"Time: {current_time(): %d-%m-%Y %H:%M:%S}\n"
                    f"Underlying LTP: {spot_price}\n"
                    f"Call Strike: {traded_strangle.call_strike}\n"
                    f"Put Strike: {traded_strangle.put_strike}\n"
                    f"Call Price: {c_ltp}\n"
                    f"Put Price: {p_ltp}\n"
                    f"MTM Price: {mtm_price}\n"
                    f"Call last n avg: {c_ltp_avg}\n"
                    f"Put last n avg: {p_ltp_avg}\n"
                    f"IVs: {call_iv}, {put_iv}, {avg_iv}\n"
                    f"Call SL: {info_dict['call_sl']}\n"
                    f"Put SL: {info_dict['put_sl']}\n"
                    f"Profit Pts: {info_dict['profit_in_pts']:.2f}\n"
                    f"Profit: {info_dict['profit_in_rs']:.2f}\n" + ctb_message
            )
            if current_time() - last_print_time > print_interval:
                print(message)
                last_print_time = current_time()
            if current_time() - last_log_time > log_interval:
                logger.info(message)
                last_log_time = current_time()
            if current_time() - last_notify_time > notify_interval:
                notifier(message, notification_url, "INFO")
                last_notify_time = current_time()
            sleep(sleep_time)

    @log_error(raise_error=True, notify=True)
    @increase_robustness
    def trend_catcher(info_dict, sl_type, qty_ratio):

        def check_trade_eligibility(option, price):
            if option.fetch_ltp() > price * 0.70:
                return True

        traded_strangle = info_dict["traded_strangle"]
        og_price = (
            info_dict["call_avg_price"]
            if sl_type == "put"
            else info_dict["put_avg_price"]
        )
        trend_option = (
            traded_strangle.call_option
            if sl_type == "put"
            else traded_strangle.put_option
        )

        qty_in_lots = max(int(quantity_in_lots * qty_ratio), 1)

        while not check_trade_eligibility(
                trend_option, og_price
        ) and current_time().time() < time(*exit_time):
            logger.info(f"Waiting for trend option to reach 70% of original price")
            sleep(sleep_time)

        # Placing the trend option order
        exec_details = execute_instructions(
            {
                trend_option: {
                    "action": Action.SELL,
                    "quantity_in_lots": qty_in_lots,
                    "order_tag": f"{strategy_tag} Trend Catcher",
                }
            }
        )
        sell_avg_price = exec_details[trend_option]

        # Setting up the stop loss on the trend option

        trend_sl_hit = False
        notifier(
            f"{underlying.name} strangle {sl_type} trend catcher starting. "
            f"Placed {qty_in_lots} lots of {trend_option} at {sell_avg_price}. "
            f"Stoploss prices: {og_price}",
            notification_url,
            "INFO",
        )

        last_print_time = current_time()
        print_interval = timedelta(seconds=10)
        while all(
                [
                    current_time().time() < time(*exit_time),
                    not info_dict["trade_complete"],
                ]
        ):
            option_price = trend_option.fetch_ltp()
            trend_sl_hit = option_price >= og_price
            if trend_sl_hit:
                break
            sleep(sleep_time)
            if current_time() - last_print_time > print_interval:
                last_print_time = current_time()
                logger.info(
                    f"{underlying.name} {sl_type} trend catcher running\n"
                    f"Stoploss price: {og_price}\n"
                )

        if trend_sl_hit:
            notifier(
                f"{underlying.name} strangle {sl_type} trend catcher stoploss hit.",
                notification_url,
                "INFO",
            )
            square_off = True
        else:
            notifier(
                f"{underlying.name} strangle {sl_type} trend catcher exiting.",
                notification_url,
                "INFO",
            )
            if info_dict["time_left_day_start"] * 365 < 1:  # expiry day
                square_off = False
            else:
                square_off = True

        if square_off:
            # Buying the trend option back
            exec_details = execute_instructions(
                {
                    trend_option: {
                        "action": Action.BUY,
                        "quantity_in_lots": qty_in_lots,
                        "order_tag": f"{strategy_tag} Trend Catcher",
                    }
                }
            )
            square_up_avg_price = exec_details[trend_option]
        else:
            square_up_avg_price = trend_option.fetch_ltp()

        points_captured = sell_avg_price - square_up_avg_price
        info_dict["trend_catcher_points_captured"] = points_captured

    def justify_stop_loss(info_dict, side):
        entry_spot = info_dict.get("spot_at_entry")
        current_spot = info_dict.get("underlying_ltp")
        stop_loss_price = info_dict.get(f"{side}_stop_loss_price")

        time_left_day_start = info_dict.get("time_left_day_start")
        time_left_now = time_to_expiry(expiry)
        time_delta_minutes = (time_left_day_start - time_left_now) * 525600
        time_delta_minutes = int(time_delta_minutes)
        time_delta_minutes = min(
            time_delta_minutes, 300
        )  # Hard coded number. At most 300 minutes and not more.
        try:
            simulated_option_price = simulate_price(
                strike=(
                    info_dict.get("traded_strangle").call_strike
                    if side == "call"
                    else info_dict.get("traded_strangle").put_strike
                ),
                flag=side,
                original_atm_iv=info_dict.get("atm_iv_at_entry"),
                original_spot=entry_spot,
                original_time_to_expiry=time_left_day_start,
                new_spot=current_spot,
                time_delta_minutes=time_delta_minutes,
            )
        except (Exception, IntrinsicValueError) as ex:
            error_message = (
                f"Error in justify_stop_loss for {underlying.name} {side} strangle: {ex}\n"
                f"Setting stop loss to True"
            )
            logger.error(error_message)
            notifier(error_message, notification_url, "ERROR")
            return True

        actual_price = info_dict.get(f"{side}_ltp_avg")
        unjust_increase = (
                actual_price / simulated_option_price > simulation_safe_guard
                and simulated_option_price < stop_loss_price
        )
        if unjust_increase:
            if not info_dict.get(f"{side}_sl_check_notification_sent"):
                message = (
                    f"{underlying.name} strangle {side} stop loss appears to be unjustified. "
                    f"Actual price: {actual_price}, Simulated price: {simulated_option_price}"
                )
                notifier(message, notification_url, "CRUCIAL")
                info_dict[f"{side}_sl_check_notification_sent"] = True

            # Additional check for unjustified stop loss (forcing stoploss to trigger even if unjustified only if
            # the price has increased by more than 2 times AND spot has moved by more than 0.5%)
            spot_change = (current_spot / entry_spot) - 1
            spot_moved = (
                spot_change > 0.012 if side == "call" else spot_change < -0.0035
            )  # Hard coded number
            if (
                    spot_moved and (actual_price / stop_loss_price) > 1.6
            ):  # Hard coded number
                message = (
                    f"{underlying.name} strangle {side} stop loss forced to trigger due to price increase. "
                    f"Price increase from stop loss price: {actual_price / simulated_option_price}"
                )
                notifier(message, notification_url, "CRUCIAL")
                return True
            else:
                return False
        else:
            message = (
                f"{underlying.name} strangle {side} stop loss triggered. "
                f"Actual price: {actual_price}, Simulated price: {simulated_option_price}"
            )
            notifier(message, notification_url, "CRUCIAL")
            return True

    def check_for_stop_loss(info_dict, side):
        """Check for stop loss."""

        stop_loss_order_ids = info_dict.get(f"{side}_stop_loss_order_ids")

        if stop_loss_order_ids is None:  # If stop loss order ids are not provided
            ltp_avg = info_dict.get(f"{side}_ltp_avg", info_dict.get(f"{side}_ltp"))
            stop_loss_price = info_dict.get(f"{side}_stop_loss_price")
            stop_loss_triggered = ltp_avg > stop_loss_price
            if stop_loss_triggered:
                stop_loss_justified = justify_stop_loss(info_dict, side)
                if stop_loss_justified:
                    info_dict[f"{side}_sl"] = True

        else:  # If stop loss order ids are provided
            orderbook = fetch_book("orderbook")
            orders_triggered, orders_complete = process_stop_loss_order_statuses(
                orderbook,
                stop_loss_order_ids,
                context=side,
                notify_url=notification_url,
            )
            if orders_triggered:
                justify_stop_loss(info_dict, side)
                info_dict[f"{side}_sl"] = True
                if not orders_complete:
                    info_dict[f"{side}_stop_loss_order_ids"] = None

    def process_stop_loss(info_dict, sl_type):
        if (
                info_dict["call_sl"] and info_dict["put_sl"]
        ):  # Check to avoid double processing
            return

        traded_strangle = info_dict["traded_strangle"]
        other_side: str = "call" if sl_type == "put" else "put"

        # Buying the stop loss option back if it is not already bought
        if info_dict[f"{sl_type}_stop_loss_order_ids"] is None:
            option_to_buy = (
                traded_strangle.call_option
                if sl_type == "call"
                else traded_strangle.put_option
            )
            exec_details = execute_instructions(
                {
                    option_to_buy: {
                        "action": Action.BUY,
                        "quantity_in_lots": quantity_in_lots,
                        "order_tag": strategy_tag,
                    }
                }
            )
            exit_price = exec_details[option_to_buy]

        else:
            orderbook = fetch_book("orderbook")
            exit_price = (
                lookup_and_return(
                    orderbook,
                    "orderid",
                    info_dict[f"{sl_type}_stop_loss_order_ids"],
                    "averageprice",
                )
                .astype(float)
                .mean()
            )
        info_dict[f"{sl_type}_exit_price"] = exit_price

        if move_sl_to_cost:
            info_dict[f"{other_side}_stop_loss_price"] = info_dict[
                f"{other_side}_avg_price"
            ]
            if (
                    info_dict[f"{other_side}_stop_loss_order_ids"] is not None
                    or place_orders_on_sl
            ):
                if info_dict[f"{other_side}_stop_loss_order_ids"] is not None:
                    cancel_pending_orders(
                        info_dict[f"{other_side}_stop_loss_order_ids"], "STOPLOSS"
                    )
                option_to_repair = (
                    traded_strangle.call_option
                    if other_side == "call"
                    else traded_strangle.put_option
                )
                info_dict[f"{other_side}_stop_loss_order_ids"] = (
                    place_option_order_and_notify(
                        instrument=option_to_repair,
                        action="BUY",
                        qty_in_lots=quantity_in_lots,
                        prices=info_dict[f"{other_side}_stop_loss_price"],
                        order_tag=f"{other_side.capitalize()} stop loss {strategy_tag}",
                        webhook_url=notification_url,
                        stop_loss_order=True,
                        target_status="trigger pending",
                        return_avg_price=False,
                    )
                )

        # Starting the trend catcher
        if catch_trend:
            trend_thread = Thread(
                target=trend_catcher,
                args=(
                    info_dict,
                    sl_type,
                    trend_qty_ratio,
                ),
                name=f"{underlying.name} {sl_type} trend catcher",
            )
            trend_thread.start()
            info_dict["active_threads"].append(trend_thread)

        sleep(5)  # To ensure that the stop loss orders are reflected in the orderbook

        # Wait for exit or other stop loss to hit
        while all(
                [
                    current_time().time() < time(*exit_time),
                    not info_dict["exit_triggers"]["take_profit"],
                ]
        ):
            check_for_stop_loss(info_dict, other_side)
            if info_dict[f"{other_side}_sl"]:
                if info_dict[f"{other_side}_stop_loss_order_ids"] is None:
                    other_sl_option = (
                        traded_strangle.call_option
                        if other_side == "call"
                        else traded_strangle.put_option
                    )
                    notifier(
                        f"{underlying.name} strangle {other_side} stop loss hit.",
                        notification_url,
                        "CRUCIAL",
                    )
                    exec_details = execute_instructions(
                        {
                            other_sl_option: {
                                "action": Action.BUY,
                                "quantity_in_lots": quantity_in_lots,
                                "order_tag": strategy_tag,
                            }
                        }
                    )
                    other_exit_price = exec_details[other_sl_option]
                else:
                    orderbook = fetch_book("orderbook")
                    other_exit_price = (
                        lookup_and_return(
                            orderbook,
                            "orderid",
                            info_dict[f"{other_side}_stop_loss_order_ids"],
                            "averageprice",
                        )
                        .astype(float)
                        .mean()
                    )
                info_dict[f"{other_side}_exit_price"] = other_exit_price
                break
            sleep(1)

    # Entering the main function
    if time(*exit_time) < current_time().time():
        notifier(
            f"{underlying.name} intraday strangle not being deployed after exit time",
            notification_url,
            "INFO",
        )
        return
    expiry = underlying.current_expiry
    quantity_in_lots = convert_exposure_to_lots(
        exposure, underlying.fetch_ltp(), underlying.lot_size
    )

    if combined_stop_loss is None:
        # If combined stop loss is not provided, then it is set to np.nan, and
        # individual stop losses are calculated
        combined_stop_loss = np.nan
        # Setting stop loss
        stop_loss_dict = {
            "fixed": {"BANKNIFTY": 1.7, "NIFTY": 1.5},
            "dynamic": {"BANKNIFTY": 1.7, "NIFTY": 1.5},
        }

        if isinstance(stop_loss, str):
            if stop_loss == "dynamic" and time_to_expiry(expiry, in_days=True) < 1:
                stop_loss = 1.7
            else:
                stop_loss = stop_loss_dict[stop_loss].get(underlying.name, 1.6)
        else:
            stop_loss = stop_loss
    else:
        # If combined stop loss is provided, then individual stop losses are set to np.nan
        stop_loss = np.nan

    if strike_selection == "equal":
        strangle = most_equal_strangle(
            underlying=underlying,
            call_strike_offset=call_strike_offset,
            put_strike_offset=put_strike_offset,
            disparity_threshold=disparity_threshold,
            exit_time=exit_time,
            expiry=expiry,
            notification_url=notification_url,
        )
        if strangle is None:
            notifier(
                f"{underlying.name} no strangle found within disparity threshold {disparity_threshold}",
                notification_url,
                "INFO",
            )
            return
    elif strike_selection == "resilient":
        strangle = underlying.most_resilient_strangle(
            stop_loss=stop_loss, expiry=expiry
        )
    elif strike_selection == "atm":
        atm_strike = find_strike(underlying.fetch_ltp(), underlying.base)
        strangle = Strangle(atm_strike, atm_strike, underlying.name, expiry)
    else:
        raise ValueError(f"Invalid find mode: {strike_selection}")

    call_ltp, put_ltp = strangle.fetch_ltp()

    # Placing the main order
    execution_details = execute_instructions(
        {
            strangle: {
                "action": Action.SELL,
                "quantity_in_lots": quantity_in_lots,
                "order_tag": strategy_tag,
            }
        }
    )
    call_avg_price, put_avg_price = execution_details[strangle]
    total_avg_price = call_avg_price + put_avg_price

    # Calculating stop loss prices
    call_stop_loss_price = (
        call_avg_price * call_stop_loss
        if call_stop_loss
        else call_avg_price * stop_loss
    )
    put_stop_loss_price = (
        put_avg_price * put_stop_loss if put_stop_loss else put_avg_price * stop_loss
    )
    combined_stop_loss_price = total_avg_price * combined_stop_loss

    underlying_ltp = underlying.fetch_ltp()

    # Logging information and sending notification
    trade_log = {
        "Time": current_time().strftime("%d-%m-%Y %H:%M:%S"),
        "Index": underlying.name,
        "Underlying price": underlying_ltp,
        "Call strike": strangle.call_strike,
        "Put strike": strangle.put_strike,
        "Expiry": expiry,
        "Action": "SELL",
        "Call price": call_avg_price,
        "Put price": put_avg_price,
        "Total price": total_avg_price,
        "Order tag": strategy_tag,
    }

    summary_message = "\n".join(f"{k}: {v}" for k, v in trade_log.items())

    # Setting the IV information at entry

    traded_call_iv, traded_put_iv, traded_avg_iv = calculate_strangle_iv(
        call_price=call_avg_price,
        put_price=put_avg_price,
        call_strike=strangle.call_strike,
        put_strike=strangle.put_strike,
        spot=underlying_ltp,
        time_left=time_to_expiry(expiry),
    )
    try:
        atm_iv_at_entry = underlying.fetch_atm_info()["avg_iv"]
    except Exception as e:
        logger.error(f"Error in fetching ATM IV: {e}")
        atm_iv_at_entry = np.nan
    time_left_at_trade = time_to_expiry(expiry)

    # Sending the summary message
    summary_message += (
        f"\nTraded IVs: {traded_call_iv}, {traded_put_iv}, {traded_avg_iv}\n"
        f"ATM IV at entry: {atm_iv_at_entry}\n"
        f"Call SL: {call_stop_loss_price}, Put SL: {put_stop_loss_price}\n"
        f"Combined SL: {combined_stop_loss_price}\n"
    )
    notifier(summary_message, notification_url, "INFO")

    if place_sl_orders:
        call_stop_loss_order_ids = place_option_order_and_notify(
            instrument=strangle.call_option,
            action="BUY",
            qty_in_lots=quantity_in_lots,
            prices=call_stop_loss_price,
            order_tag=strategy_tag,
            webhook_url=notification_url,
            stop_loss_order=True,
            target_status="trigger pending",
            return_avg_price=False,
        )
        put_stop_loss_order_ids = place_option_order_and_notify(
            instrument=strangle.put_option,
            action="BUY",
            qty_in_lots=quantity_in_lots,
            prices=put_stop_loss_price,
            order_tag=strategy_tag,
            webhook_url=notification_url,
            stop_loss_order=True,
            target_status="trigger pending",
            return_avg_price=False,
        )
    else:
        call_stop_loss_order_ids = None
        put_stop_loss_order_ids = None

    # Setting up shared info dict
    shared_info_dict = {
        "traded_strangle": strangle,
        "spot_at_entry": underlying_ltp,
        "call_avg_price": call_avg_price,
        "put_avg_price": put_avg_price,
        "total_avg_price": total_avg_price,
        "atm_iv_at_entry": atm_iv_at_entry,
        "call_stop_loss_price": call_stop_loss_price,
        "put_stop_loss_price": put_stop_loss_price,
        "combined_stop_loss_price": combined_stop_loss_price,
        "call_stop_loss_order_ids": call_stop_loss_order_ids,
        "put_stop_loss_order_ids": put_stop_loss_order_ids,
        "time_left_day_start": time_left_at_trade,
        "call_ltp": call_ltp,
        "put_ltp": put_ltp,
        "underlying_ltp": underlying_ltp,
        "call_iv": traded_call_iv,
        "put_iv": traded_put_iv,
        "avg_iv": traded_avg_iv,
        "call_sl": False,
        "put_sl": False,
        "exit_triggers": {
            "convert_to_butterfly": False,
            "take_profit": False,
            "combined_stop_loss": False,
        },
        "trade_complete": False,
        "call_sl_check_notification_sent": False,
        "put_sl_check_notification_sent": False,
        "active_threads": [],
        "trend_catcher_points_captured": 0,
    }

    position_monitor_thread = Thread(
        target=position_monitor, args=(shared_info_dict,), name="Position monitor"
    )
    position_monitor_thread.start()
    shared_info_dict["active_threads"].append(position_monitor_thread)
    sleep(
        5
    )  # To ensure that the position monitor thread has started and orders are reflected in the orderbook

    # Wait for exit time or both stop losses to hit (Main Loop)
    while all(
            [
                current_time().time() < time(*exit_time),
                not any(shared_info_dict["exit_triggers"].values()),
            ]
    ):
        if combined_stop_loss is not None and not np.isnan(combined_stop_loss):
            pass
        else:
            check_for_stop_loss(shared_info_dict, "call")
            if shared_info_dict["call_sl"]:
                process_stop_loss(shared_info_dict, "call")
                break
            check_for_stop_loss(shared_info_dict, "put")
            if shared_info_dict["put_sl"]:
                process_stop_loss(shared_info_dict, "put")
                break
        sleep(1)

    # Out of the while loop, so exit time reached or both stop losses hit, or we are hedged

    # If we are hedged then wait till exit time
    # noinspection PyTypeChecker
    if shared_info_dict["exit_triggers"]["convert_to_butterfly"]:
        hedge_strangle = shared_info_dict["ctb_hedge"]
        execute_instructions(
            {
                hedge_strangle: {
                    "action": Action.BUY,
                    "quantity_in_lots": quantity_in_lots,
                    "order_tag": strategy_tag,
                }
            }
        )
        if place_sl_orders:
            cancel_pending_orders(
                shared_info_dict["call_stop_loss_order_ids"]
                + shared_info_dict["put_stop_loss_order_ids"]
            )
        notifier(f"{underlying.name}: Converted to butterfly", notification_url, "INFO")
        while current_time().time() < time(*exit_time):
            sleep(3)

    call_sl = shared_info_dict["call_sl"]
    put_sl = shared_info_dict["put_sl"]

    if not call_sl and not put_sl:  # Both stop losses not hit
        execution_details = execute_instructions(
            {
                strangle: {
                    "action": Action.BUY,
                    "quantity_in_lots": quantity_in_lots,
                    "order_tag": strategy_tag,
                }
            }
        )
        call_exit_avg_price, put_exit_avg_price = execution_details[strangle]

        # noinspection PyTypeChecker
        if (
                place_sl_orders
                and not shared_info_dict["exit_triggers"]["convert_to_butterfly"]
        ):
            cancel_pending_orders(
                shared_info_dict["call_stop_loss_order_ids"]
                + shared_info_dict["put_stop_loss_order_ids"]
            )
        shared_info_dict["call_exit_price"] = call_exit_avg_price
        shared_info_dict["put_exit_price"] = put_exit_avg_price

    elif (call_sl or put_sl) and not (call_sl and put_sl):  # Only one stop loss hit
        exit_option_type: str = "put" if call_sl else "call"
        exit_option = strangle.put_option if call_sl else strangle.call_option
        execution_details = execute_instructions(
            {
                exit_option: {
                    "action": Action.BUY,
                    "quantity_in_lots": quantity_in_lots,
                    "order_tag": strategy_tag,
                }
            }
        )
        non_sl_exit_price = execution_details[exit_option]
        if place_sl_orders or place_orders_on_sl:
            cancel_pending_orders(
                shared_info_dict[f"{exit_option_type}_stop_loss_order_ids"]
            )
        shared_info_dict[f"{exit_option_type}_exit_price"] = non_sl_exit_price

    else:  # Both stop losses hit
        pass

    shared_info_dict["trade_complete"] = True
    for thread in shared_info_dict["active_threads"]:
        thread.join()

    # Calculate profit
    total_exit_price = (
            shared_info_dict["call_exit_price"] + shared_info_dict["put_exit_price"]
    )
    # Exit message
    exit_message = (
        f"{underlying.name} strangle exited.\n"
        f"Time: {current_time(): %d-%m-%Y %H:%M:%S}\n"
        f"Underlying LTP: {shared_info_dict['underlying_ltp']}\n"
        f"Call Price: {shared_info_dict['call_ltp']}\n"
        f"Put Price: {shared_info_dict['put_ltp']}\n"
        f"Call SL: {shared_info_dict['call_sl']}\n"
        f"Put SL: {shared_info_dict['put_sl']}\n"
        f"Call Exit Price: {shared_info_dict['call_exit_price']}\n"
        f"Put Exit Price: {shared_info_dict['put_exit_price']}\n"
        f"Total Exit Price: {total_exit_price}\n"
        f"Total Entry Price: {total_avg_price}\n"
        f"Profit Points: {total_avg_price - total_exit_price}\n"
        f"Chase Points: {shared_info_dict['trend_catcher_points_captured']}\n"
    )
    # Exit dict
    exit_dict = {
        "Call exit price": shared_info_dict["call_exit_price"],
        "Put exit price": shared_info_dict["put_exit_price"],
        "Total exit price": total_exit_price,
        "Points captured": total_avg_price - total_exit_price,
        "Call stop loss": shared_info_dict["call_sl"],
        "Put stop loss": shared_info_dict["put_sl"],
        "Trend catcher points": shared_info_dict["trend_catcher_points_captured"],
    }

    notifier(exit_message, notification_url, "CRUCIAL")
    trade_log.update(exit_dict)
    underlying.strategy_log[strategy_tag].append(trade_log)

    return shared_info_dict
