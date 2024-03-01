from collections import defaultdict
import pandas as pd
import numpy as np
import requests
import itertools
from typing import Iterable
from fuzzywuzzy import process
from datetime import datetime, timedelta
from autotradr.decorators import timeit
from autotradr.exceptions import ScripsLocationError
from autotradr.config import logger, symbol_df
from autotradr import config
from autotradr.utils import (
    current_time,
    get_expiry_dates,
    get_symbol_token,
    get_lot_size,
    get_base,
    find_strike,
    find_strike_with_offset,
    time_to_expiry,
    get_available_strikes,
)
from autotradr.angel_interface.interface import fetch_ltp
from autotradr.trade_interface.blocks import Straddle, Strangle, Option, OptionType


class Index:
    """Initialize an index with the name of the index in uppercase"""

    EXPIRY_FREQUENCY: dict = {
        "MIDCPNIFTY": 0,
        "FINNIFTY": 1,
        "BANKNIFTY": 2,
        "NIFTY": 3,
        "SENSEX": 4,
    }

    def __init__(self, name, caching=False):
        self.name = name.upper()
        self.current_expiry = None
        self.next_expiry = None
        self.far_expiry = None
        self.month_expiry = None
        self.fut_expiry = None
        self.exchange = "BSE" if self.name in ["SENSEX", "BANKEX"] else "NSE"
        self.fno_exchange = "BFO" if self.name in ["SENSEX", "BANKEX"] else "NFO"
        self.symbol, self.token = get_symbol_token(self.name)
        self.future_symbol_tokens = {}
        self.fetch_exps()
        self.lot_size = get_lot_size(self.name, self.current_expiry)
        self.freeze_qty = self.fetch_freeze_limit()
        self.available_strikes = None
        self.available_straddle_strikes = None
        self.base = get_base(self.name, self.current_expiry)
        self.strategy_log = defaultdict(list)
        self.exchange_type = 1

        # Caching attributes
        self.caching = caching
        self._ltp = None
        self._last_ltp_fetch_time = datetime(1997, 12, 30)
        self._basis = {}
        self._last_basis_fetch_time = datetime(1997, 12, 30)

        logger.info(
            f"Initialized {self.name} with lot size {self.lot_size}, base {self.base} and freeze qty {self.freeze_qty}"
        )

    def __hash__(self):
        # Hash will be base on the name
        return hash(self.name)

    def __eq__(self, other):
        # Equality will be based on the name
        if isinstance(other, Index):
            return self.name == other.name
        return False

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(Name: {self.name}, Lot Size: {self.lot_size}, "
            f"Freeze Qty: {self.freeze_qty}, Current Expiry: {self.current_expiry}, Symbol: {self.symbol}, "
            f"Token: {self.token})"
        )

    def fetch_freeze_limit(self):
        try:
            freeze_qty_url = "https://archives.nseindia.com/content/fo/qtyfreeze.xls"
            response = requests.get(freeze_qty_url, timeout=10)  # Set the timeout value
            response.raise_for_status()  # Raise an exception if the response contains an HTTP error status
            df = pd.read_excel(response.content)
            df.columns = df.columns.str.strip()
            df["SYMBOL"] = df["SYMBOL"].str.strip()
            freeze_qty = df[df["SYMBOL"] == self.name]["VOL_FRZ_QTY"].values[0]
            freeze_qty_in_lots = freeze_qty / self.lot_size
            return int(freeze_qty_in_lots)
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error in fetching freeze limit for {self.name}: {e}")
            freeze_qty_in_lots = 30
            return int(freeze_qty_in_lots)
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error in fetching freeze limit for {self.name}: {e}")
            freeze_qty_in_lots = 30
            return int(freeze_qty_in_lots)
        except Exception as e:
            logger.error(f"Error in fetching freeze limit for {self.name}: {e}")
            freeze_qty_in_lots = 100 if self.name == "SENSEX" else 30  # Hardcoded
            return int(freeze_qty_in_lots)

    def fetch_exps(self):
        exps = get_expiry_dates(self.EXPIRY_FREQUENCY.get(self.name, "monthly"))
        exps = pd.DatetimeIndex(exps).strftime("%d%b%y").str.upper().tolist()

        self.current_expiry = exps[0]
        self.next_expiry = exps[1]
        self.far_expiry = exps[2]

        if self.name in self.EXPIRY_FREQUENCY:
            self.fut_expiry = self.month_expiry = exps[3]
        else:
            self.fut_expiry = self.month_expiry = exps[0]

    def set_future_symbol_tokens(self):
        if not self.future_symbol_tokens:
            for i in range(0, 3):
                try:
                    self.future_symbol_tokens[i] = get_symbol_token(self.name, future=i)
                except ScripsLocationError:
                    self.future_symbol_tokens[i] = (None, None)
                    continue

    def _fetch_future_ltp(self, future):
        try:
            ltp = fetch_ltp(
                self.fno_exchange,
                self.future_symbol_tokens[future][0],
                self.future_symbol_tokens[future][1],
            )
        except Exception as e:
            error_message_to_catch = (
                "Error in fetching LTP: 'NoneType' object is not subscriptable"
            )
            if str(e) == error_message_to_catch:
                ltp = np.nan
            else:
                raise e
        return ltp

    def fetch_ltp(self, future=None):
        """Fetch LTP of the index."""
        if isinstance(future, int):
            ltp = self._fetch_future_ltp(future)
            return ltp
        else:  # Spot price
            if (
                self.caching
                and self._ltp is not None
                and current_time() - self._last_ltp_fetch_time
                < timedelta(seconds=config.CACHE_INTERVAL)
            ):
                logger.debug("Using cache in fetch_ltp")
                return self._ltp
            else:  # Fetch from source
                ltp = fetch_ltp(self.exchange, self.symbol, self.token)
                self._ltp = ltp
                self._last_ltp_fetch_time = current_time()
                return ltp

    def get_atm_straddle(
        self,
        expiry: str = None,
        underlying_price: float = None,
    ) -> Straddle:
        expiry = self.current_expiry if expiry is None else expiry
        underlying_price = (
            self.fetch_ltp() if underlying_price is None else underlying_price
        )
        atm_strike = find_strike(underlying_price, self.base)
        atm_straddle = Straddle(atm_strike, self.name, expiry)
        return atm_straddle

    def get_basis_for_expiry(
        self,
        expiry: str = None,
        underlying_price: float = None,
        future_price: float = None,
    ) -> float:
        expiry = self.current_expiry if expiry is None else expiry
        underlying_price = (
            self.fetch_ltp() if underlying_price is None else underlying_price
        )

        if (
            self.caching
            and current_time() - self._last_basis_fetch_time
            < timedelta(seconds=config.CACHE_INTERVAL)
            and expiry in self._basis
        ):
            logger.debug(f"Using cache in get_basis_for_expiry for {expiry}")
            return self._basis[expiry]

        if future_price is None:
            atm_straddle: Straddle = self.get_atm_straddle(expiry, underlying_price)
            call_price, put_price = atm_straddle.fetch_ltp()
            future_price = atm_straddle.strike + call_price - put_price
        tte = time_to_expiry(expiry)
        basis = (future_price / underlying_price) - 1
        annualized_basis = basis / tte
        adjusted_annualized_basis = (
            annualized_basis * 1.01
        )  # A small 1% adjustment to avoid intrinsic value errors
        # Can be removed later
        self._basis[expiry] = adjusted_annualized_basis
        self._last_basis_fetch_time = current_time()
        return adjusted_annualized_basis

    def fetch_atm_info(self, expiry="current", effective_iv=False):
        expiry_dict = {
            "current": self.current_expiry,
            "next": self.next_expiry,
            "month": self.month_expiry,
            "future": self.fut_expiry,
            "fut": self.fut_expiry,
        }
        expiry = expiry_dict[expiry]
        price = self.fetch_ltp()
        atm_straddle = self.get_atm_straddle(expiry, price)
        call_price, put_price = atm_straddle.fetch_ltp()
        synthetic_price = atm_straddle.strike + call_price - put_price
        r = self.get_basis_for_expiry(expiry, price, synthetic_price)
        total_price = call_price + put_price
        call_iv, put_iv, avg_iv = atm_straddle.fetch_ivs(
            spot=price, prices=(call_price, put_price), effective_iv=effective_iv, r=r
        )
        return {
            "underlying_price": price,
            "strike": atm_straddle.strike,
            "call_price": call_price,
            "put_price": put_price,
            "total_price": total_price,
            "synthetic_future_price": synthetic_price,
            "annualized_basis": r,
            "call_iv": call_iv,
            "put_iv": put_iv,
            "avg_iv": avg_iv,
        }

    def fetch_otm_info(self, strike_offset, expiry="current", effective_iv=False):
        expiry_dict = {
            "current": self.current_expiry,
            "next": self.next_expiry,
            "month": self.month_expiry,
            "future": self.fut_expiry,
            "fut": self.fut_expiry,
        }
        expiry = expiry_dict[expiry]
        price = self.fetch_ltp()
        call_strike = price * (1 + strike_offset)
        put_strike = price * (1 - strike_offset)
        call_strike = find_strike(call_strike, self.base)
        put_strike = find_strike(put_strike, self.base)
        otm_strangle = Strangle(call_strike, put_strike, self.name, expiry)
        call_price, put_price = otm_strangle.fetch_ltp()
        total_price = call_price + put_price
        call_iv, put_iv, avg_iv = otm_strangle.fetch_ivs(
            spot=price, prices=(call_price, put_price), effective_iv=effective_iv
        )
        return {
            "underlying_price": price,
            "call_strike": call_strike,
            "put_strike": put_strike,
            "call_price": call_price,
            "put_price": put_price,
            "total_price": total_price,
            "call_iv": call_iv,
            "put_iv": put_iv,
            "avg_iv": avg_iv,
        }

    def get_available_strikes(self, both_pairs=False):
        available_strikes = get_available_strikes(self.name, both_pairs)
        if not both_pairs:
            self.available_strikes = available_strikes
        else:
            self.available_straddle_strikes = available_strikes
        return available_strikes

    def get_constituents(self, cutoff_pct=101):
        constituents = (
            pd.read_csv(f"data/{self.name}_constituents.csv")
            .sort_values("Index weight", ascending=False)
            .assign(cum_weight=lambda df: df["Index weight"].cumsum())
            .loc[lambda df: df.cum_weight < cutoff_pct]
        )

        constituent_tickers, constituent_weights = (
            constituents.Ticker.to_list(),
            constituents["Index weight"].to_list(),
        )

        return constituent_tickers, constituent_weights

    def get_active_strikes(
        self, range_of_strikes: int, offset: float = 0, ltp: float = None
    ) -> list[int]:
        ltp = self.fetch_ltp() if ltp is None else ltp
        current_strike = find_strike_with_offset(ltp, offset, self.base)
        strike_range = np.arange(
            current_strike - (self.base * range_of_strikes),
            current_strike + (self.base * range_of_strikes),
            self.base,
        )
        strike_range = [*map(int, strike_range)]
        return strike_range

    def get_otm_strikes(
        self, strike_range: Iterable[int], option_type: OptionType
    ) -> list[int]:
        """Filters out itm strikes and returns tradeable strikes. Itm is defined as strike which is
        more than 4 bases away from the ltp. Direction of itm is determined by option_type
        HARDCODED 4 BASES"""
        ltp = self.fetch_ltp()
        return [
            strike
            for strike in strike_range
            if option_type == OptionType.CALL
            and strike > (ltp - 4 * self.base)
            or option_type == OptionType.PUT
            and strike < (ltp + 4 * self.base)
        ]

    def get_range_of_strangles(
        self, c_strike, p_strike, strike_range, exp=None
    ) -> list[Strangle | Straddle]:
        """Gets a range of strangles around the given strikes. If c_strike == p_strike, returns a range of straddles"""

        if exp is None:
            exp = self.current_expiry

        if strike_range % 2 != 0:
            strike_range += 1
        c_strike_range = np.arange(
            c_strike - (strike_range / 2) * self.base,
            c_strike + (strike_range / 2) * self.base + self.base,
            self.base,
        )
        if c_strike == p_strike:
            return [Straddle(strike, self.name, exp) for strike in c_strike_range]
        else:
            p_strike_ranges = np.arange(
                p_strike - (strike_range / 2) * self.base,
                p_strike + (strike_range / 2) * self.base + self.base,
                self.base,
            )
            pairs = itertools.product(c_strike_range, p_strike_ranges)
            return [Strangle(pair[0], pair[1], self.name, exp) for pair in pairs]

    def splice_orders(self, quantity_in_lots):
        if quantity_in_lots > self.freeze_qty:
            loops = int(quantity_in_lots / self.freeze_qty)
            if loops > config.LARGE_ORDER_THRESHOLD:
                raise Exception(
                    "Order too big. This error was raised to prevent accidental large order placement."
                )

            remainder = quantity_in_lots % self.freeze_qty
            if remainder == 0:
                spliced_orders = [self.freeze_qty] * loops
            else:
                spliced_orders = [self.freeze_qty] * loops + [remainder]
        else:
            spliced_orders = [quantity_in_lots]
        return spliced_orders

    def return_greeks_for_strikes(
        self, strike_range=4, expiry=None, option_type=OptionType.CALL
    ):
        if expiry is None:
            expiry = self.current_expiry
        underlying_price = self.fetch_ltp()
        atm_strike = find_strike(underlying_price, self.base)
        strikes = (
            np.arange(atm_strike, atm_strike + strike_range * self.base, self.base)
            if option_type == OptionType.CALL
            else np.arange(
                atm_strike - strike_range * self.base, atm_strike + self.base, self.base
            )
        )
        options = [Option(strike, option_type, self.name, expiry) for strike in strikes]
        greek_dict = {option: option.fetch_greeks() for option in options}
        return greek_dict

    @timeit()
    def most_resilient_strangle(
        self,
        strike_range=40,
        expiry=None,
        extra_buffer=1.07,
    ) -> Strangle:
        def expected_movement(option: Option):
            print(ltp_cache[option])
            raise NotImplementedError

        def find_favorite_strike(expected_moves, options, benchmark_movement):
            for i in range(1, len(expected_moves)):
                if (
                    expected_moves[i] > benchmark_movement * extra_buffer
                    and expected_moves[i] > expected_moves[i - 1]
                ):
                    return options[i]
            return None

        if expiry is None:
            expiry = self.current_expiry

        spot_price = self.fetch_ltp()
        atm_strike = find_strike(spot_price, self.base)

        half_range = int(strike_range / 2)
        strike_range = np.arange(
            atm_strike - (self.base * half_range),
            atm_strike + (self.base * (half_range + 1)),
            self.base,
        )

        options_by_type = {
            OptionType.CALL: [
                Option(
                    strike=strike,
                    option_type=OptionType.CALL,
                    underlying=self.name,
                    expiry=expiry,
                )
                for strike in strike_range
                if strike >= atm_strike
            ],
            OptionType.PUT: [
                Option(
                    strike=strike,
                    option_type=OptionType.PUT,
                    underlying=self.name,
                    expiry=expiry,
                )
                for strike in strike_range[::-1]
                if strike <= atm_strike
            ],
        }

        ltp_cache = {
            option: option.fetch_ltp()
            for option_type in options_by_type
            for option in options_by_type[option_type]
        }

        expected_movements = {
            option_type: [expected_movement(option) for option in options]
            for option_type, options in options_by_type.items()
        }

        expected_movements_ce = np.array(expected_movements[OptionType.CALL])
        expected_movements_pe = np.array(expected_movements[OptionType.PUT])
        expected_movements_pe = expected_movements_pe * -1

        benchmark_movement_ce = expected_movements_ce[0]
        benchmark_movement_pe = expected_movements_pe[0]

        logger.info(
            f"{self.name} - Call options' expected movements: "
            f"{list(zip(options_by_type[OptionType.CALL], expected_movements_ce))}"
        )
        logger.info(
            f"{self.name} - Put options' expected movements: "
            f"{list(zip(options_by_type[OptionType.PUT], expected_movements_pe))}"
        )

        favorite_strike_ce = (
            find_favorite_strike(
                expected_movements_ce,
                options_by_type[OptionType.CALL],
                benchmark_movement_ce,
            )
            or options_by_type[OptionType.CALL][0]
        )  # If no favorite strike, use ATM strike
        favorite_strike_pe = (
            find_favorite_strike(
                expected_movements_pe,
                options_by_type[OptionType.PUT],
                benchmark_movement_pe,
            )
            or options_by_type[OptionType.PUT][0]
        )  # If no favorite strike, use ATM strike

        ce_strike = favorite_strike_ce.strike
        pe_strike = favorite_strike_pe.strike
        strangle = Strangle(ce_strike, pe_strike, self.name, expiry)

        return strangle


class Stock(Index):
    def __init__(self, name):
        if name not in symbol_df["SYMBOL"].values:
            closest_match, confidence = process.extractOne(
                name, symbol_df["SYMBOL"].values
            )
            if confidence > 80:
                raise Exception(
                    f"Index {name} not found. Did you mean {closest_match}?"
                )

            else:
                raise ValueError(f"Index {name} not found")
        super().__init__(name)


class IndiaVix:
    symbol, token = None, None

    @classmethod
    def fetch_ltp(cls):
        if cls.symbol is None or cls.token is None:
            cls.symbol, cls.token = get_symbol_token("INDIA VIX")
        return fetch_ltp("NSE", cls.symbol, cls.token)
