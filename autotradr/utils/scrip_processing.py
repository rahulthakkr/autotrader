from dateutil import rrule
from datetime import datetime, timedelta
from itertools import groupby
import numpy as np
import pandas as pd
from functools import lru_cache
from autotradr.config import holidays, scrips, logger, implemented_indices
from autotradr.utils.core import current_time
from autotradr.exceptions import ScripsLocationError


@lru_cache(maxsize=1280)
def get_symbol_token(
    name=None, expiry=None, strike=None, option_type=None, tokens=None, future=None
):
    """Fetches symbol & token for a given scrip name. Provide just a single world if
    you want to fetch the symbol & token for the cash segment. If you want to fetch the
    symbol & token for the options segment, provide name, strike, expiry, option_type.
    Expiry should be in the DDMMMYY format. Optiontype should be CE or PE. Optionally, provide
    a list of tokens to fetch the corresponding symbols."""

    if tokens is None and name is None:
        raise ValueError("Either name or tokens must be specified.")

    if tokens is not None:
        token_df = scrips.loc[scrips["token"].isin(tokens)]
        symbol_token_pairs = [
            (token_df.loc[token_df["token"] == token, "symbol"].values[0], token)
            for token in tokens
        ]
        return symbol_token_pairs

    name = name.upper()

    if expiry is None and strike is None and option_type is None:  # Cash segment
        if (
            name in implemented_indices and future is None
        ):  # Index scrips and non futures
            exchange = "BSE" if name in ["SENSEX", "BANKEX"] else "NSE"
            filtered_scrips = scrips.loc[
                (scrips.name == name)
                & (scrips.exch_seg == exchange)
                & (scrips.instrumenttype == "AMXIDX")
            ]

            if len(filtered_scrips) != 1:
                if len(filtered_scrips) == 0:
                    logger.error(f"No scrips found: {filtered_scrips}")
                else:
                    logger.error(f"Multiple scrips found: {filtered_scrips}")
                raise Exception(f"Could not ascertain symbol, token for {name}")

            symbol, token = filtered_scrips[["symbol", "token"]].values[0]

        elif isinstance(future, int):  # Futures
            future = future if future is not None else 0  # Default to current month
            instrument_type = "FUTIDX" if "NIFTY" in name else "FUTSTK"
            try:
                futures = scrips.loc[
                    (scrips.name == name) & (scrips.instrumenttype == instrument_type),
                    ["expiry", "symbol", "token"],
                ]
                futures["expiry"] = pd.to_datetime(futures["expiry"], format="%d%b%Y")
                futures = futures.sort_values(by="expiry")
                symbol, token = futures.iloc[future][["symbol", "token"]].values
            except Exception as e:
                logger.error(
                    f"Could not find symbol, token for {name} future {future}. Error: {e}"
                )
                raise ScripsLocationError(
                    additional_info=f"Future {future} not found for {name}. Error: {e}"
                )

        else:  # For all other equity scrips
            filtered_scrips = scrips.loc[
                (scrips.name == name)
                & (scrips.exch_seg == "NSE")
                & (scrips.symbol.str.endswith("EQ"))
            ]
            assert (
                len(filtered_scrips) == 1
            ), "More than one equity scrip found for name."
            symbol, token = filtered_scrips[["symbol", "token"]].values[0]

    elif (
        expiry is not None and strike is not None and option_type is not None
    ):  # Options segment
        strike = int(strike)  # Handle float strikes, convert to integer first
        option_type = option_type.upper()
        filtered_scrips = scrips.loc[
            (scrips.name == name)
            & (scrips.expiry_formatted == expiry)
            & (scrips.strike == strike * 100)
            & (scrips.symbol.apply(lambda x: x[-2:]) == option_type)
        ]
        symbol, token = filtered_scrips[["symbol", "token"]].values[0]
    else:
        raise ValueError("Invalid arguments")

    return symbol, token


def special_expiry_handling_for_banknifty(
    nearest_three_weekly: pd.DatetimeIndex, nearest_monthly: pd.DatetimeIndex
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    # Again special handling for Bank-nifty. Replacing the last Wednesday with the last Thursday.
    nearest_three_weekly = pd.DatetimeIndex(
        np.where(
            nearest_monthly[0] - nearest_three_weekly == timedelta(days=1),
            nearest_monthly[0],
            nearest_three_weekly,
        )
    )

    # Now handling the calculation on the last Thursday. Manually inserting the last Thursday.
    # Because on the last Thursday the weekly expiries will produce a different list
    time_to_nearest_monthly = nearest_monthly[0] - current_time()
    if time_to_nearest_monthly < timedelta(days=1):
        nearest_three_weekly = pd.DatetimeIndex(
            nearest_three_weekly.insert(0, nearest_monthly[0])
        )
        nearest_three_weekly = nearest_three_weekly[:3]
    return nearest_three_weekly, nearest_monthly


def get_expiry_dates(expiry_frequency: int):
    if expiry_frequency == "monthly":
        monthly_expiry_dates = generate_potential_expiry_dates(
            rule=rrule.MONTHLY, weekday=rrule.TH(-1)
        )
        monthly_expiry_dates = [
            shift_for_holidays_and_weekends(date) for date in monthly_expiry_dates
        ]
        return monthly_expiry_dates[:4]
    else:
        # Generate potential expiry dates based on the given weekday and range
        weekly_expiry_dates = generate_potential_expiry_dates(
            rule=rrule.WEEKLY, weekday=expiry_frequency
        )
        # Check for the special condition: Last Thursday of the month (Bank-nifty)
        if expiry_frequency == 2:
            monthly_expiry_dates = generate_potential_expiry_dates(
                rule=rrule.MONTHLY, weekday=rrule.TH(-1), count=2
            )
        else:
            grouped_exps = groupby(weekly_expiry_dates, lambda x: x.month)
            monthly_expiry_dates = [
                max(group, key=lambda x: x.day) for _, group in grouped_exps
            ]

        # Only keep the nearest three weekly expiry dates and the nearest monthly expiry date
        nearest_three_weekly: pd.DatetimeIndex = weekly_expiry_dates[:3]
        nearest_monthly: pd.DatetimeIndex = monthly_expiry_dates[:1]

        # Special handling for Bank-nifty. Replacing the last Wednesday with the last Thursday.
        if expiry_frequency == 2:
            (
                nearest_three_weekly,
                nearest_monthly,
            ) = special_expiry_handling_for_banknifty(
                nearest_three_weekly, nearest_monthly
            )

        # Shift expiry dates that fall on holidays to the previous business day
        nearest_three_weekly: list = [
            shift_for_holidays_and_weekends(date) for date in nearest_three_weekly
        ]

        nearest_monthly: list = [
            shift_for_holidays_and_weekends(date) for date in nearest_monthly
        ]

        return nearest_three_weekly + nearest_monthly


@lru_cache
def get_lot_size(name, expiry=None):
    if expiry is None:
        expiry_mask = scrips.expiry_formatted == scrips.expiry_formatted
    else:
        expiry_mask = scrips.expiry_formatted == expiry

    exchange = "BFO" if name in ["SENSEX", "BANKEX"] else "NFO"

    filtered_df = scrips.loc[
        (scrips.name == name) & (scrips.exch_seg == exchange) & expiry_mask
    ]
    lot_sizes = filtered_df.lotsize.values

    if len(set(lot_sizes)) > 1:
        logger.info(
            f"Multiple lot sizes found for {name}. Using the closest expiry lot size."
        )
        filtered_df = filtered_df.sort_values("expiry_dt")
        return filtered_df.lotsize.iloc[0]

    else:
        return lot_sizes[0]


@lru_cache
def get_base(name, expiry):
    exchange = "BFO" if name in ["SENSEX", "BANKEX"] else "NFO"

    # Filter and sort the data frame based on the name and exchange
    strike_array = scrips.loc[
        (scrips["name"] == name)
        & (scrips["exch_seg"] == exchange)
        & (scrips["expiry_formatted"] == expiry)
    ]

    # Convert strikes to array and scale by 100
    strikes = strike_array["strike"].values / 100

    # Calculate the differences and find the mode
    strike_differences = np.diff(np.sort(np.unique(strikes)))
    values, counts = np.unique(strike_differences, return_counts=True)
    mode = values[np.argmax(counts)]

    return mode


def get_available_strikes(name, both_pairs=False):
    exchange = "BFO" if name in ["SENSEX", "BANKEX"] else "NFO"
    mask = (
        (scrips.name == name)
        & (scrips.exch_seg == exchange)
        & (scrips.instrumenttype.str.startswith("OPT"))
    )
    filtered = scrips.loc[mask].copy()
    filtered["strike"] = filtered["strike"] / 100
    filtered_dict = (
        filtered.groupby("expiry")["strike"]
        .unique()
        .apply(list)
        .apply(sorted)
        .to_dict()
    )
    new_keys = map(
        lambda x: datetime.strptime(x, "%d%b%Y").strftime("%d%b%y").upper(),
        filtered_dict.keys(),
    )
    filtered_dict = {k: v for k, v in zip(new_keys, filtered_dict.values())}
    sorted_dict = {
        k: filtered_dict[k]
        for k in sorted(filtered_dict, key=lambda x: datetime.strptime(x, "%d%b%y"))
    }
    if not both_pairs:
        return sorted_dict

    def filter_dictionary(dictionary):
        pair_filtered_dict = {
            expiry: [strike for strike in strikes if check_strike(expiry, strike)]
            for expiry, strikes in dictionary.items()
        }
        return {key: values for key, values in pair_filtered_dict.items() if values}

    def check_strike(expiry, stk):
        try:
            return get_straddle_symbol_tokens(name, stk, expiry)
        except IndexError:
            # print(f"No straddle available for {name} {expiry} {stk}")
            return False

    return filter_dictionary(sorted_dict)


def get_straddle_symbol_tokens(name, strike, expiry):
    c_symbol, c_token = get_symbol_token(name, expiry, strike, "CE")
    p_symbol, p_token = get_symbol_token(name, expiry, strike, "PE")
    return c_symbol, c_token, p_symbol, p_token


def shift_for_holidays_and_weekends(date):
    while date.date() in holidays or date.weekday() in [5, 6]:
        date -= timedelta(days=1)
    return date


def generate_potential_expiry_dates(
    rule: int | rrule.rrule, weekday: rrule.weekday | int, count: int = 5
) -> pd.DatetimeIndex:
    potential_dates = list(
        rrule.rrule(
            rule,
            byweekday=weekday,
            dtstart=current_time().date(),
            count=count,
        )
    )
    # Adding time and filtering out expired dates
    potential_dates = [date + timedelta(minutes=930) for date in potential_dates]
    potential_dates = [*filter(lambda x: x > current_time(), potential_dates)]
    return pd.DatetimeIndex(potential_dates)
