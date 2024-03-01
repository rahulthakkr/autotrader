from scipy.stats import norm
from scipy.optimize import brentq
import numpy as np
from attrs import define, field
import pandas as pd
from autotradr.exceptions import IntrinsicValueError
from autotradr.config import bs_logger

N = norm.cdf
binary_flag = {"c": 1, "p": -1}


@define(repr=False)
class Greeks:
    """A class to store the greeks"""

    iv: float = field(default=np.nan)
    _delta: float = field(default=np.nan)
    _gamma: float = field(default=np.nan)
    _theta: float = field(default=np.nan)
    _vega: float = field(default=np.nan)
    _array: np.ndarray = field(init=False)

    def __attrs_post_init__(self):
        self.iv = round(self.iv, 4)
        self._delta = round(self._delta, 4)
        self._gamma = round(self._gamma, 8)
        self._theta = round(self._theta, 2)
        self._vega = round(self._vega, 4)
        self._array = np.array([self._delta, self._gamma, self._theta, self._vega])

    def __repr__(self):
        return f"Greeks(iv={self.iv}, delta={self.delta}, gamma={self.gamma}, theta={self.theta}, vega={self.vega})"

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, value):
        self._delta = value
        self._update_array()

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self._update_array()

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        self._theta = value
        self._update_array()

    @property
    def vega(self):
        return self._vega

    @vega.setter
    def vega(self, value):
        self._vega = value
        self._update_array()

    def _update_array(self):
        self._array = np.array([self._delta, self._gamma, self._theta, self._vega])

    def __add__(self, other):
        if isinstance(other, Greeks):
            avg_iv = (self.iv + other.iv) / 2
            return Greeks(avg_iv, *(self._array + other._array))
        elif isinstance(other, (int, float)) and other == 0:
            return Greeks(self.iv, *self._array)
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Greeks):
            avg_iv = (self.iv + other.iv) / 2
            return Greeks(avg_iv, *(self._array - other._array))
        else:
            raise TypeError(
                f"unsupported operand type(s) for -: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Greeks(self.iv, *(self._array * other))
        else:
            raise TypeError(
                f"unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Greeks(self.iv, *(self._array / other))
        else:
            raise TypeError(
                f"unsupported operand type(s) for /: '{type(self).__name__}' and '{type(other).__name__}'"
            )


def pdf(x):
    """the probability density function"""
    one_over_sqrt_two_pi = 0.3989422804014326779399460599343818684758586311649
    return one_over_sqrt_two_pi * np.exp(-0.5 * x * x)


def d1(S, K, t, r, sigma):
    sigma_squared = sigma * sigma
    numerator = np.log(S / float(K)) + (r + sigma_squared / 2.0) * t
    denominator = sigma * np.sqrt(t)

    if not denominator:
        print("")
    return numerator / denominator


def d2(S, K, t, r, sigma):
    return d1(S, K, t, r, sigma) - sigma * np.sqrt(t)


def forward_price(S, t, r):
    return S / np.exp(-r * t)


def call(S, K, t, r, sigma):
    e_to_the_minus_rt = np.exp(-r * t)
    D1 = d1(S, K, t, r, sigma)
    D2 = d2(S, K, t, r, sigma)

    return S * N(D1) - K * e_to_the_minus_rt * N(D2)


def put(S, K, t, r, sigma):
    e_to_the_minus_rt = np.exp(-r * t)
    D1 = d1(S, K, t, r, sigma)
    D2 = d2(S, K, t, r, sigma)

    return -S * N(-D1) + K * e_to_the_minus_rt * N(-D2)


def delta(S, K, t, r, sigma, flag):
    d_1 = d1(S, K, t, r, sigma)

    if flag.upper().startswith("P"):
        return N(d_1) - 1.0
    else:
        return N(d_1)


def gamma(S, K, t, r, sigma):
    d_1 = d1(S, K, t, r, sigma)
    return pdf(d_1) / (S * sigma * np.sqrt(t))


def theta(S, K, t, r, sigma, flag):
    two_sqrt_t = 2 * np.sqrt(t)

    D1 = d1(S, K, t, r, sigma)
    D2 = d2(S, K, t, r, sigma)

    first_term = (-S * pdf(D1) * sigma) / two_sqrt_t

    if flag.upper().startswith("C"):
        second_term = r * K * np.exp(-r * t) * N(D2)
        return (first_term - second_term) / 365.0

    else:
        second_term = r * K * np.exp(-r * t) * N(-D2)
        return (first_term + second_term) / 365.0


def vega(S, K, t, r, sigma):
    d_1 = d1(S, K, t, r, sigma)
    return S * pdf(d_1) * np.sqrt(t) * 0.01


def rho(S, K, t, r, sigma, flag):
    d_2 = d2(S, K, t, r, sigma)
    e_to_the_minus_rt = np.exp(-r * t)
    if flag.upper().startswith("C"):
        return t * K * e_to_the_minus_rt * N(d_2) * 0.01
    else:
        return -t * K * e_to_the_minus_rt * N(-d_2) * 0.01


def implied_volatility(price, S, K, t, r, flag):
    check_for_intrinsics(price, S, K, t, r, flag)
    if flag.upper().startswith("P"):
        f = lambda sigma: price - put(S, K, t, r, sigma)
    else:
        f = lambda sigma: price - call(S, K, t, r, sigma)

    try:
        return brentq(
            f, a=1e-12, b=100, xtol=1e-15, rtol=1e-15, maxiter=1000, full_output=False
        )
    except Exception as e:
        bs_logger.error(
            f"Error in implied_volatility: {e}, price={price}, S={S}, K={K}, t={t}, r={r}, flag={flag}"
        )
        raise e


def error_handled_iv(opt_price, spot, strike, tte, opt_type, r: float = 0.06):
    try:
        return implied_volatility(opt_price, spot, strike, tte, r, opt_type)
    except IntrinsicValueError:
        return np.nan
    except Exception as e:
        bs_logger.error(f"Error in implied_volatility: {e}")
        return np.nan


def calculate_strangle_iv(
    call_price,
    put_price,
    spot,
    strike=None,
    call_strike=None,
    put_strike=None,
    time_left=None,
    r: float = 0.06,
) -> tuple[float, float, float]:
    """
    Calculate the implied volatility for options.

    :param call_price: Price of the call option.
    :param put_price: Price of the put option.
    :param spot: Current price of the underlying asset.
    :param strike: Strike price of the options. If None, assumes strangle and uses call and put strikes.
    :param call_strike: Strike price of the call option. If None, assumes straddle and uses strike.
    :param put_strike: Strike price of the put option. If None, assumes straddle and uses strike.
    :param time_left: Time left to expiration (in years).
    :param r: Interest rate.
    :return: Tuple of call IV, put IV, and average IV.
    """

    # If only one strike price is provided, use it for both call and put (straddle)
    if strike is not None:
        call_strike = strike
        put_strike = strike

    # Validate that both strike prices are now set
    if call_strike is None or put_strike is None:
        raise ValueError(
            "Strike prices for both call and put options must be provided."
        )

    # Calculate the implied volatility for the call and put options
    call_iv = error_handled_iv(call_price, spot, call_strike, time_left, "c", r)
    put_iv = error_handled_iv(put_price, spot, put_strike, time_left, "p", r)

    # If both IVs are numbers, calculate the average; otherwise, take the one that is not NaN
    if not np.isnan(call_iv) and not np.isnan(put_iv):
        avg_iv = (call_iv + put_iv) / 2
    else:
        avg_iv = call_iv if not np.isnan(call_iv) else put_iv

    return call_iv, put_iv, avg_iv


def greeks(S, K, t, r, sigma, flag):
    return Greeks(
        sigma,
        delta(S, K, t, r, sigma, flag),
        gamma(S, K, t, r, sigma),
        theta(S, K, t, r, sigma, flag),
        vega(S, K, t, r, sigma),
    )


def test_func():
    # Comparing time to calculate implied volatility using two different methods
    import timeit

    # Generate random data
    np.random.seed(42)
    Ss = np.random.uniform(40000, 45000, 100)
    Ks = np.random.uniform(40000, 45000, 100)
    ts = np.random.uniform(0.0027, 0.0191, 100)
    rs = np.array([0.05] * 100)
    flags = np.random.choice(["c", "p"], 100)
    sigmas = np.random.uniform(0.1, 0.5, 100)
    prices = np.array(
        [
            call(s, k, t, r, sigma) if f == "c" else put(s, k, t, r, sigma)
            for s, k, t, r, sigma, f in zip(Ss, Ks, ts, rs, sigmas, flags)
        ]
    )
    deltas = np.array(
        [
            delta(s, k, t, r, sigma, f)
            for s, k, t, r, sigma, f in zip(Ss, Ks, ts, rs, sigmas, flags)
        ]
    )
    gammas = np.array(
        [gamma(s, k, t, r, sigma) for s, k, t, r, sigma in zip(Ss, Ks, ts, rs, sigmas)]
    )
    thetas = np.array(
        [
            theta(s, k, t, r, sigma, f)
            for s, k, t, r, sigma, f in zip(Ss, Ks, ts, rs, sigmas, flags)
        ]
    )
    vegas = np.array(
        [vega(s, k, t, r, sigma) for s, k, t, r, sigma in zip(Ss, Ks, ts, rs, sigmas)]
    )

    # Calculate implied volatility using two different methods
    start = timeit.default_timer()
    ivs = []
    for price, s, k, t, r, f in zip(prices, Ss, Ks, ts, rs, flags):
        iv = implied_volatility(price, s, k, t, r, f)
        ivs.append(iv)

    stop = timeit.default_timer()
    print("Time to calculate implied volatility using brentq: ", stop - start)

    return pd.DataFrame(
        {
            "spot": Ss,
            "strike": Ks,
            "time": ts * 365,
            "rate": rs,
            "flag": flags,
            "sigma": sigmas,
            "price": prices,
            "delta": deltas,
            "gamma": gammas,
            "theta": thetas,
            "vega": vegas,
            "implied_volatility": ivs,
        }
    )


def check_for_intrinsics(price, spot, strike, time_to_expiry, rate, flag):
    flag = flag.lower()[0]
    spot = spot * (
        np.e ** (rate * time_to_expiry)
    )  # Adjusting it to the implied forward price
    intrinsic_value = max(spot - strike, 0) if flag == "c" else max(strike - spot, 0)
    if intrinsic_value > price:
        bs_logger.error(
            f"Current price {price} of {'call' if flag == 'c' else 'put'} "
            f"is less than the intrinsic value {intrinsic_value} "
            f"for spot {spot} and strike {strike}"
        )
        raise IntrinsicValueError(
            f"Current price {price} of {'call' if flag == 'c' else 'put'} "
            f"is less than the intrinsic value {intrinsic_value}"
        )
