from enum import Enum
import numpy as np
from attrs import define, field
from autotradr import config
from autotradr.decorators import SingletonInstances
from autotradr.config import symbol_df
from autotradr.utils import (
    get_symbol_token,
    get_lot_size,
    time_to_expiry,
    splice_orders,
)
import autotradr.blackscholes as bs
from autotradr.angel_interface.interface import fetch_ltp
from autotradr.angel_interface.orders import place_order


class OptionType(Enum):
    CALL = "CE"
    PUT = "PE"


class Action(Enum):
    BUY = "BUY"
    SELL = "SELL"


@define(repr=False, eq=False)
class _Option:
    strike: int = field(converter=lambda x: round(int(x), 0))
    option_type: OptionType = field()
    underlying: str = field(converter=lambda x: str(x).upper())
    expiry: str = field(converter=lambda x: str(x).upper())

    # Additional attributes
    underlying_symbol: str = field(init=False)
    underlying_token: str = field(init=False)
    underlying_exchange: str = field(init=False)
    exchange: str = field(init=False)
    symbol: str = field(init=False)
    token: str = field(init=False)
    lot_size: int = field(init=False)
    freeze_qty_in_shares: int = field(init=False)
    freeze_qty_in_lots: int = field(init=False)

    def __attrs_post_init__(self):
        self.symbol, self.token = get_symbol_token(
            self.underlying, self.expiry, self.strike, self.option_type.value
        )
        self.lot_size = get_lot_size(self.underlying, expiry=self.expiry)
        _set_underlying_attributes(self, self.underlying)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(strike={self.strike}, option_type={self.option_type.value}, "
            f"underlying={self.underlying}, expiry={self.expiry})"
        )

    def __hash__(self):
        return hash((self.strike, self.option_type.value, self.underlying, self.expiry))

    def __eq__(self, other):
        if not isinstance(other, _Option):
            return False
        return (
            self.strike == other.strike
            and self.expiry == other.expiry
            and self.option_type == other.option_type
            and self.underlying == other.underlying
        )

    def __lt__(self, other):
        if isinstance(other, (int, float, np.int32, np.int64, np.float32, np.float64)):
            return self.strike < other
        return self.strike < other.strike

    def __gt__(self, other):
        if isinstance(other, (int, float, np.int32, np.int64, np.float32, np.float64)):
            return self.strike > other
        return self.strike > other.strike

    def __le__(self, other):
        if isinstance(other, (int, float, np.int32, np.int64, np.float32, np.float64)):
            return self.strike <= other
        return self.strike <= other.strike

    def __ge__(self, other):
        if isinstance(other, (int, float, np.int32, np.int64, np.float32, np.float64)):
            return self.strike >= other
        return self.strike >= other.strike

    def __ne__(self, other):
        if isinstance(other, (int, float, np.int32, np.int64, np.float32, np.float64)):
            return self.strike != other
        return self.strike != other.strike

    def __sub__(self, other):
        if isinstance(other, (int, float, np.int32, np.int64, np.float32, np.float64)):
            return self.strike - other
        return self.strike - other.strike

    def __add__(self, other):
        if isinstance(other, (int, float, np.int32, np.int64, np.float32, np.float64)):
            return self.strike + other
        return self.strike + other.strike

    def fetch_symbol_token(self):
        return self.symbol, self.token

    def fetch_ltp(self):
        return fetch_ltp(self.exchange, self.symbol, self.token)

    def underlying_ltp(self):
        return fetch_ltp(
            self.underlying_exchange, self.underlying_symbol, self.underlying_token
        )

    def fetch_iv(
        self,
        spot: float | None = None,
        price: float | None = None,
        t: float | None = None,
        r: float = 0.06,
        effective_iv: bool = False,
    ):
        spot = spot if spot is not None else self.underlying_ltp()
        t = (
            t
            if t is not None
            else time_to_expiry(self.expiry, effective_time=effective_iv)
        )
        price = price if price is not None else self.fetch_ltp()
        return bs.error_handled_iv(
            price, spot, self.strike, t, opt_type=self.option_type.value, r=r
        )

    def fetch_greeks(
        self,
        spot: float | None = None,
        price: float | None = None,
        t: float | None = None,
        r: float = 0.06,
        effective_iv: bool = False,
    ) -> bs.Greeks:
        spot = self.underlying_ltp() if spot is None else spot
        t = time_to_expiry(self.expiry) if t is None else t
        price = self.fetch_ltp() if price is None else price
        iv = self.fetch_iv(spot=spot, t=t, effective_iv=effective_iv, price=price, r=r)
        return bs.greeks(
            spot,
            self.strike,
            t,
            r,
            iv,
            self.option_type.value,
        )

    def simulate_price(
        self,
        atm_iv: float,
        new_spot: float | None = None,
        movement: float | None = None,
        time_delta: float | None = None,
        time_delta_minutes: float | int | None = None,
        effective_iv: bool = False,
        retain_original_iv: bool = False,
        original_spot: float | None = None,
        original_iv: float | None = None,
    ):
        """
        Effective iv should be set to true when the square off is going to be at a higher iv. In other words,
        this is practical when the square off is likely to be after a holiday after taking position.

        IMPORTANT: When effective_iv is set to True, the function automatically assumes that the square off is going
        to happen at the next trading day after the holiday/weekend. So ensure you are not double calculating the
        holiday/weekend effect.
        """

        original_time_to_expiry = time_to_expiry(
            self.expiry, effective_time=effective_iv
        )
        original_spot = (
            original_spot if original_spot is not None else self.underlying_ltp()
        )
        original_iv = (
            original_iv
            if original_iv is not None
            else self.fetch_iv(spot=original_spot, t=original_time_to_expiry)
        )

        simulated_price = bs.simulate_price(
            strike=self.strike,
            flag=self.option_type.value,
            original_atm_iv=atm_iv,
            original_iv=original_iv,
            original_spot=original_spot,
            original_time_to_expiry=original_time_to_expiry,
            movement=movement,
            new_spot=new_spot,
            time_delta=time_delta,
            time_delta_minutes=time_delta_minutes,
            retain_original_iv=retain_original_iv,
        )
        return simulated_price

    def place_order(
        self,
        transaction_type,
        quantity_in_lots,
        price="LIMIT",
        stop_loss_order=False,
        order_tag="",
    ):
        if isinstance(price, str):
            if price.upper() == "LIMIT":
                price = self.fetch_ltp()
                modifier = (
                    (1 + config.LIMIT_PRICE_BUFFER)
                    if transaction_type == "BUY"
                    else (1 - config.LIMIT_PRICE_BUFFER)
                )
                price = price * modifier
        spliced_orders = splice_orders(quantity_in_lots, self.freeze_qty_in_lots)
        order_ids = []
        for qty in spliced_orders:
            order_id = place_order(
                self.symbol,
                self.token,
                qty * self.lot_size,
                transaction_type,
                price,
                stop_loss_order=stop_loss_order,
                order_tag=order_tag,
            )
            order_ids.append(order_id)
        return order_ids


Option = SingletonInstances("Option", (_Option,), {})


@define(repr=False, eq=False)
class Strangle:
    call_strike: int = field(converter=lambda x: round(int(x), 0))
    put_strike: int = field(converter=lambda x: round(int(x), 0))
    underlying: str = field(converter=lambda x: str(x).upper())
    expiry: str = field(converter=lambda x: str(x).upper())

    # Additional attributes
    call_option: Option = field(init=False)
    put_option: Option = field(init=False)
    underlying_symbol: str = field(init=False)
    underlying_token: str = field(init=False)
    underlying_exchange: str = field(init=False)
    exchange: str = field(init=False)
    call_symbol: str = field(init=False)
    call_token: str = field(init=False)
    put_symbol: str = field(init=False)
    put_token: str = field(init=False)
    lot_size: int = field(init=False)
    freeze_qty_in_shares: int = field(init=False)
    freeze_qty_in_lots: int = field(init=False)

    def __attrs_post_init__(self):
        self.call_option = Option(
            self.call_strike, OptionType.CALL, self.underlying, self.expiry
        )
        self.put_option = Option(
            self.put_strike, OptionType.PUT, self.underlying, self.expiry
        )
        self.call_symbol, self.call_token = self.call_option.fetch_symbol_token()
        self.put_symbol, self.put_token = self.put_option.fetch_symbol_token()
        self.lot_size = self.call_option.lot_size
        _set_underlying_attributes(self, self.underlying)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(callstrike={self.call_option.strike}, putstrike={self.put_option.strike}, "
            f"underlying={self.underlying}, expiry={self.expiry})"
        )

    def __hash__(self):
        return hash((self.call_strike, self.put_strike, self.underlying, self.expiry))

    def __eq__(self, other):
        return (
            self.call_option == other.call_option
            and self.put_option == other.put_option
        )

    def fetch_ltp(self):
        return fetch_ltp(self.exchange, self.call_symbol, self.call_token), fetch_ltp(
            self.exchange, self.put_symbol, self.put_token
        )

    def underlying_ltp(self):
        return fetch_ltp(
            self.underlying_exchange, self.underlying_symbol, self.underlying_token
        )

    def fetch_ivs(
        self,
        spot: float | None = None,
        prices: tuple[float, float] | None = None,
        t: float | None = None,
        effective_iv: bool = False,
        r: float = 0.06,
    ) -> tuple[float | None, float | None, float | None]:
        spot = spot if spot is not None else self.underlying_ltp()
        t = (
            t
            if t is not None
            else time_to_expiry(self.expiry, effective_time=effective_iv)
        )

        call_price, put_price = prices if prices is not None else self.fetch_ltp()

        return bs.calculate_strangle_iv(
            call_price,
            put_price,
            spot,
            call_strike=self.call_strike,
            put_strike=self.put_strike,
            time_left=t,
            r=r,
        )

    def fetch_greeks(
        self,
        spot: float | None = None,
        prices: tuple[float, float] | None = None,
        t: float | None = None,
        effective_iv: bool = False,
        r: float = 0.06,
    ) -> tuple[bs.Greeks, bs.Greeks]:
        spot = spot if spot is not None else self.underlying_ltp()
        t = time_to_expiry(self.expiry, effective_time=effective_iv) if t is None else t
        call_price, put_price = prices if prices is not None else self.fetch_ltp()

        call_greeks = bs.greeks(
            spot,
            self.call_strike,
            t,
            r,
            self.call_option.fetch_iv(spot=spot, price=call_price, t=t, r=r),
            "c",
        )
        put_greeks = bs.greeks(
            spot,
            self.put_strike,
            t,
            r,
            self.put_option.fetch_iv(spot=spot, price=put_price, t=t, r=r),
            "p",
        )
        return call_greeks, put_greeks

    def fetch_total_ltp(self):
        call_ltp, put_ltp = fetch_ltp(
            self.exchange, self.call_symbol, self.call_token
        ), fetch_ltp(self.exchange, self.put_symbol, self.put_token)
        return call_ltp + put_ltp

    def price_disparity(self):
        call_ltp, put_ltp = self.fetch_ltp()
        disparity = abs(call_ltp - put_ltp) / min(call_ltp, put_ltp)
        return disparity

    def fetch_symbol_token(self):
        return self.call_symbol, self.call_token, self.put_symbol, self.put_token

    def simulate_price(
        self,
        atm_iv: float,
        new_spot: float | None = None,
        movement: float | None = None,
        time_delta: float | None = None,
        time_delta_minutes: float | int | None = None,
        effective_iv: bool = False,
        retain_original_iv: bool = False,
        return_total: bool = True,
        original_spot: float | None = None,
        original_ivs: tuple[float, float] | None = None,
    ):
        original_time_to_expiry = time_to_expiry(
            self.expiry, effective_time=effective_iv
        )
        original_spot = (
            original_spot if original_spot is not None else self.underlying_ltp()
        )

        if original_ivs is not None:
            call_original_iv, put_original_iv = original_ivs
        else:
            call_original_iv = self.call_option.fetch_iv(
                spot=original_spot, t=original_time_to_expiry
            )
            put_original_iv = self.put_option.fetch_iv(
                spot=original_spot, t=original_time_to_expiry
            )

        call_simulated_price = self.call_option.simulate_price(
            atm_iv=atm_iv,
            new_spot=new_spot,
            movement=movement,
            time_delta=time_delta,
            time_delta_minutes=time_delta_minutes,
            effective_iv=effective_iv,
            retain_original_iv=retain_original_iv,
            original_spot=original_spot,
            original_iv=call_original_iv,
        )
        put_simulated_price = self.put_option.simulate_price(
            atm_iv=atm_iv,
            new_spot=new_spot,
            movement=movement,
            time_delta=time_delta,
            time_delta_minutes=time_delta_minutes,
            effective_iv=effective_iv,
            retain_original_iv=retain_original_iv,
            original_spot=original_spot,
            original_iv=put_original_iv,
        )
        if return_total:
            return call_simulated_price + put_simulated_price
        else:
            return call_simulated_price, put_simulated_price

    def simulate_price_both_directions(
        self,
        atm_iv: float,
        movement: float,
        time_delta: float | None = None,
        time_delta_minutes: float | int | None = None,
        effective_iv: bool = False,
        retain_original_iv: bool = False,
        return_total: bool = True,
        up_weightage: float = 0.55,
        original_spot: float | None = None,
        original_ivs: tuple[float, float] | None = None,
    ):
        """
        Movement should be absolute value as the function will simulate movement in both directions.
        The up weightage is the weightage given to the up movement. The down weightage is 1 - up weightage.
        The default up weightage is 0.55 which is based on analysis.
        """

        original_time_to_expiry = time_to_expiry(
            self.expiry, effective_time=effective_iv
        )
        original_spot = (
            original_spot if original_spot is not None else self.underlying_ltp()
        )
        original_ivs = (
            original_ivs
            if original_ivs is not None
            else (
                self.call_option.fetch_iv(
                    spot=original_spot, t=original_time_to_expiry
                ),
                self.put_option.fetch_iv(spot=original_spot, t=original_time_to_expiry),
            )
        )

        price_if_up = self.simulate_price(
            atm_iv=atm_iv,
            movement=movement,
            time_delta=time_delta,
            time_delta_minutes=time_delta_minutes,
            effective_iv=effective_iv,
            retain_original_iv=retain_original_iv,
            return_total=True,
            original_spot=original_spot,
            original_ivs=original_ivs,
        )
        price_if_down = self.simulate_price(
            atm_iv=atm_iv,
            movement=-movement,
            time_delta=time_delta,
            time_delta_minutes=time_delta_minutes,
            effective_iv=effective_iv,
            retain_original_iv=retain_original_iv,
            return_total=True,
            original_spot=original_spot,
            original_ivs=original_ivs,
        )
        if return_total:
            return (up_weightage * price_if_up) + ((1 - up_weightage) * price_if_down)
        else:
            return price_if_up, price_if_down

    def place_order(
        self,
        transaction_type,
        quantity_in_lots,
        prices="LIMIT",
        stop_loss_order=False,
        order_tag="",
    ):
        if stop_loss_order:
            assert isinstance(
                prices, (tuple, list, np.ndarray)
            ), "Prices must be a tuple of prices for stop loss order"
            call_price, put_price = prices
        else:
            if isinstance(prices, (tuple, list, np.ndarray)):
                call_price, put_price = prices
            elif prices.upper() == "LIMIT":
                call_price, put_price = self.fetch_ltp()
                modifier = (
                    (1 + config.LIMIT_PRICE_BUFFER)
                    if transaction_type == "BUY"
                    else (1 - config.LIMIT_PRICE_BUFFER)
                )
                call_price, put_price = call_price * modifier, put_price * modifier
            elif prices.upper() == "MARKET":
                call_price = put_price = prices
            else:
                raise ValueError(
                    "Prices must be either 'LIMIT' or 'MARKET' or a tuple of prices"
                )

        spliced_orders = splice_orders(quantity_in_lots, self.freeze_qty_in_lots)
        call_order_ids = []
        put_order_ids = []
        for qty in spliced_orders:
            call_order_id = place_order(
                self.call_symbol,
                self.call_token,
                qty * self.lot_size,
                transaction_type,
                call_price,
                stop_loss_order=stop_loss_order,
                order_tag=order_tag,
            )
            put_order_id = place_order(
                self.put_symbol,
                self.put_token,
                qty * self.lot_size,
                transaction_type,
                put_price,
                stop_loss_order=stop_loss_order,
                order_tag=order_tag,
            )
            call_order_ids.append(call_order_id)
            put_order_ids.append(put_order_id)
        return call_order_ids, put_order_ids


class Straddle(Strangle):
    def __init__(self, strike, underlying, expiry):
        super().__init__(strike, strike, underlying, expiry)
        self.strike = strike


class SyntheticFuture(Strangle):
    def __init__(self, strike, underlying, expiry):
        super().__init__(strike, strike, underlying, expiry)

    def place_order(
        self,
        transaction_type,
        quantity_in_lots,
        prices: str | tuple = "LIMIT",
        stop_loss_order=False,
        order_tag="",
    ):
        if isinstance(prices, (tuple, list, np.ndarray)):
            call_price, put_price = prices
        elif prices.upper() == "LIMIT":
            call_price, put_price = self.fetch_ltp()
            c_modifier, p_modifier = (
                (1 + config.LIMIT_PRICE_BUFFER, 1 - config.LIMIT_PRICE_BUFFER)
                if transaction_type.upper() == "BUY"
                else (1 - config.LIMIT_PRICE_BUFFER, 1 + config.LIMIT_PRICE_BUFFER)
            )
            call_price, put_price = call_price * c_modifier, put_price * p_modifier
        elif prices.upper() == "MARKET":
            call_price = put_price = prices
        else:
            raise ValueError(
                "Prices must be either 'LIMIT' or 'MARKET' or a tuple of prices"
            )

        call_transaction_type = "BUY" if transaction_type.upper() == "BUY" else "SELL"
        put_transaction_type = "SELL" if transaction_type.upper() == "BUY" else "BUY"

        spliced_orders = splice_orders(quantity_in_lots, self.freeze_qty_in_lots)
        call_order_ids = []
        put_order_ids = []
        for qty in spliced_orders:
            call_order_id = place_order(
                self.call_symbol,
                self.call_token,
                qty * self.lot_size,
                call_transaction_type,
                call_price,
                order_tag=order_tag,
            )
            put_order_id = place_order(
                self.put_symbol,
                self.put_token,
                qty * self.lot_size,
                put_transaction_type,
                put_price,
                order_tag=order_tag,
            )
            call_order_ids.append(call_order_id)
            put_order_ids.append(put_order_id)
        return call_order_ids, put_order_ids


def _set_underlying_attributes(instance, underlying: str):
    is_bse = underlying in ["SENSEX", "BANKEX"]
    instance.exchange = "BFO" if is_bse else "NFO"
    instance.underlying_symbol, instance.underlying_token = get_symbol_token(underlying)
    instance.underlying_exchange = "BSE" if is_bse else "NSE"
    try:
        instance.freeze_qty_in_shares = symbol_df[symbol_df["SYMBOL"] == underlying][
            "VOL_FRZ_QTY"
        ].values[0]
    except IndexError:
        freeze_limit = 100 if underlying == "SENSEX" else 30
        instance.freeze_qty_in_shares = instance.lot_size * freeze_limit
    instance.freeze_qty_in_lots = int(instance.freeze_qty_in_shares / instance.lot_size)
