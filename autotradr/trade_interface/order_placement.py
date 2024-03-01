from time import sleep
import numpy as np
import itertools
from autotradr.decorators import timeit
from autotradr import config
from autotradr.config import logger, order_placed
from autotradr.utils.core import time_to_expiry
from autotradr.utils.communication import notifier, log_error
from autotradr.angel_interface.active_session import ActiveSession
from autotradr.angel_interface.interface import fetch_book, lookup_and_return
from autotradr.trade_interface.blocks import (
    Option,
    Strangle,
    Straddle,
    SyntheticFuture,
    Action,
)


@timeit()
def place_option_order_and_notify(
    instrument: Option | Strangle | Straddle | SyntheticFuture,
    action: Action | str,
    qty_in_lots: int,
    prices: str | int | float | tuple | list | np.ndarray = "LIMIT",
    order_tag: str = "",
    webhook_url=None,
    stop_loss_order: bool = False,
    target_status: str = "complete",
    return_avg_price: bool = True,
    square_off_order: bool = False,
    **kwargs,
) -> list | tuple | float | None:
    """Returns either a list of order ids or a tuple of avg prices or a float of avg price"""

    def return_avg_price_from_orderbook(
        orderbook: list, ids: list | tuple | np.ndarray
    ):
        avg_prices = lookup_and_return(
            orderbook, ["orderid", "status"], [ids, "complete"], "averageprice"
        )
        return avg_prices.astype(float).mean() if avg_prices.size > 0 else None

    action = action.value if isinstance(action, Action) else action

    # If square_off_order is True, check if the expiry is within 3 minutes
    if square_off_order and time_to_expiry(instrument.expiry, in_days=True) < (
        3 / (24 * 60)
    ):
        logger.info(
            f"Square off order not placed for {instrument} as expiry is within 5 minutes"
        )
        return instrument.fetch_ltp() if return_avg_price else None

    notify_dict = {
        "order_tag": order_tag,
        "Underlying": instrument.underlying,
        "Action": action,
        "Expiry": instrument.expiry,
        "Qty": qty_in_lots,
    }

    order_params = {
        "transaction_type": action,
        "quantity_in_lots": qty_in_lots,
        "stop_loss_order": stop_loss_order,
        "order_tag": order_tag,
    }

    if isinstance(instrument, (Strangle, Straddle, SyntheticFuture)):
        notify_dict.update({"Strikes": [instrument.call_strike, instrument.put_strike]})
        order_params.update({"prices": prices})
    elif isinstance(instrument, Option):
        notify_dict.update(
            {"Strike": instrument.strike, "OptionType": instrument.option_type.value}
        )
        order_params.update({"price": prices})
    else:
        raise ValueError("Invalid instrument type")

    notify_dict.update(kwargs)

    if stop_loss_order:
        assert isinstance(
            prices, (int, float, tuple, list, np.ndarray)
        ), "Stop loss order requires a price"
        target_status = "trigger pending"

    # Placing the order
    order_ids = instrument.place_order(**order_params)

    if isinstance(order_ids, tuple):  # Strangle/Straddle/SyntheticFuture
        call_order_ids, put_order_ids = order_ids[0], order_ids[1]
        order_ids = list(itertools.chain(call_order_ids, put_order_ids))
    else:  # Option
        call_order_ids, put_order_ids = False, False

    order_placed.set()

    # Waiting for the orders to reflect
    sleep(0.5)

    order_book = fetch_book("orderbook")
    order_statuses_ = lookup_and_return(order_book, "orderid", order_ids, "status")
    if isinstance(order_statuses_, np.ndarray) and order_statuses_.size > 0:
        check_and_notify_order_placement_statuses(
            statuses=order_statuses_,
            target_status=target_status,
            webhook_url=webhook_url,
            **notify_dict,
        )
    else:
        notifier(
            f"Unable to check statuses. Order statuses is {order_statuses_} for orderid(s) {order_ids}. "
            f"Please confirm execution.",
            webhook_url,
            "ERROR",
        )

    if return_avg_price:
        if call_order_ids and put_order_ids:  # Strangle/Straddle/SyntheticFuture
            call_avg_price = (
                return_avg_price_from_orderbook(order_book, call_order_ids)
                or instrument.call_option.fetch_ltp()
            )
            put_avg_price = (
                return_avg_price_from_orderbook(order_book, put_order_ids)
                or instrument.put_option.fetch_ltp()
            )
            result = call_avg_price, put_avg_price
        else:  # Option
            avg_price = (
                return_avg_price_from_orderbook(order_book, order_ids)
                or instrument.fetch_ltp()
            )
            result = avg_price
        return result

    return order_ids


def check_and_notify_order_placement_statuses(
    statuses, target_status="complete", webhook_url=None, **kwargs
):
    order_prefix = (
        f"{kwargs['order_tag']}: "
        if ("order_tag" in kwargs and kwargs["order_tag"])
        else ""
    )
    order_message = [f"{k}-{v}" for k, v in kwargs.items() if k != "order_tag"]
    order_message = ", ".join(order_message)

    if all(statuses == target_status):
        logger.info(f"{order_prefix}Order(s) placed successfully for {order_message}")
    elif any(statuses == "rejected"):
        if all(statuses == "rejected"):
            notifier(
                f"{order_prefix}All orders rejected for {order_message}",
                [config.ERROR_NOTIFICATION_SETTINGS["url"], webhook_url],
                "ERROR",
            )
            raise Exception("Orders rejected")
        notifier(
            f"{order_prefix}Some orders rejected for {order_message}. Please repair.",
            [config.ERROR_NOTIFICATION_SETTINGS["url"], webhook_url],
            "CRUCIAL",
        )
    elif any(["open" in status or "modi" in status for status in statuses]):
        logger.info(
            f"{order_prefix}Orders open for {order_message}. Awaiting modification."
        )
    elif any(statuses == target_status):
        notifier(
            f"{order_prefix}Some orders successful for {order_message}. Please repair the remaining orders.",
            [config.ERROR_NOTIFICATION_SETTINGS["url"], webhook_url],
            "CRUCIAL",
        )
    else:
        notifier(
            f"{order_prefix}No orders successful. Please intervene.",
            [config.ERROR_NOTIFICATION_SETTINGS["url"], webhook_url],
            "ERROR",
        )
        raise Exception("No orders successful")


def process_stop_loss_order_statuses(
    order_book,
    order_ids,
    context="",
    notify_url=None,
):
    pending_text = "trigger pending"
    context = f"{context.capitalize()} " if context else ""

    statuses = lookup_and_return(order_book, "orderid", order_ids, "status")

    if not isinstance(statuses, np.ndarray) or statuses.size == 0:
        logger.error(f"Statuses is {statuses} for orderid(s) {order_ids}")

    if all(statuses == pending_text):
        return False, False

    elif all(statuses == "rejected") or all(statuses == "cancelled"):
        rejection_reasons = lookup_and_return(order_book, "orderid", order_ids, "text")
        if all(rejection_reasons == "17070 : The Price is out of the LPP range"):
            return True, False
        else:
            notifier(
                f"{context}Order(s) rejected or cancelled. Reasons: {rejection_reasons[0]}",
                notify_url,
                "ERROR",
            )
            raise Exception(f"Order(s) rejected or cancelled.")

    elif all(statuses == "pending"):
        sleep(5)
        order_book = fetch_book("orderbook")
        statuses = lookup_and_return(order_book, "orderid", order_ids, "status")

        if all(statuses == "pending"):
            try:
                cancel_pending_orders(order_ids, "NORMAL")
            except Exception:
                try:
                    cancel_pending_orders(order_ids, "STOPLOSS")
                except Exception as e:
                    notifier(
                        f"{context}Could not cancel orders: {e}", notify_url, "ERROR"
                    )
                    raise Exception(f"Could not cancel orders: {e}")
            notifier(
                f"{context}Orders pending and cancelled. Please check.",
                notify_url,
                "ERROR",
            )
            return True, False

        elif all(statuses == "complete"):
            return True, True

        else:
            logger.error(
                f"Orders in unknown state. Statuses: {statuses}, Order ids: {order_ids}"
            )
            raise Exception(f"Orders in unknown state.")

    elif all(statuses == "complete"):
        return True, True

    else:
        notifier(
            f"{context}Orders in unknown state. Statuses: {statuses}",
            notify_url,
            "ERROR",
        )
        raise Exception(f"Orders in unknown state.")


@log_error()
def cancel_pending_orders(order_ids, variety="STOPLOSS"):
    if isinstance(order_ids, (list, np.ndarray)):
        for order_id in order_ids:
            ActiveSession.obj.cancelOrder(order_id, variety)
    else:
        ActiveSession.obj.cancelOrder(order_ids, variety)
