import urllib
import requests
import pandas as pd
import logging
import os
from datetime import datetime
from pathlib import Path
from threading import local, Event
from bs4 import BeautifulSoup
from twilio.rest import Client as TwilioClient


def get_ticker_file():
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    data = urllib.request.urlopen(url).read().decode()
    df = pd.read_json(data)
    return df


def fetch_holidays():
    url = "https://www.angelone.in/nse-holidays-2023"
    backup_file = Path("holidays.csv")

    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        table = soup.find("table", attrs={"class": "inner-table"})
        headers = [th.text.strip() for th in table.find("tr").find_all("th")]
        rows = [
            [td.text.strip() for td in tr.find_all("td")]
            for tr in table.find_all("tr")[1:]
        ]

        df = pd.DataFrame(rows, columns=headers)
        df["Date"] = pd.to_datetime(df["Date"])

        # Check if the dataframe is empty and use the backup if it is
        if not df.empty:
            # Save to a CSV file
            df.to_csv(backup_file, index=False)
            holidays = df["Date"].values.astype("datetime64[D]")
        else:
            raise ValueError("Fetched data is empty, falling back to local backup.")

    except Exception as e:
        message = f"Failed to fetch holidays from {url}: {e}"
        logger.error(message)

    return holidays


def get_symbols():
    try:
        freeze_qty_url = "https://archives.nseindia.com/content/fo/qtyfreeze.xls"
        response = requests.get(freeze_qty_url, timeout=10)  # Set the timeout value
        response.raise_for_status()  # Raise an exception if the response contains an HTTP error status
        df = pd.read_excel(response.content)
        df.columns = df.columns.str.strip()
        df["SYMBOL"] = df["SYMBOL"].str.strip()
        return df
    except Exception as e:
        logger.error(f"Error while fetching symbols: {e}")
        return pd.DataFrame()


def create_logger(
    logger_name,
    file_prefix: str = "",
    info_handler=True,
    error_handler=True,
    stream_handler=True,
):
    """
    Creates a logger with specified configurations.

    Parameters:
    logger_name (str): The name of the logger.
    file_prefix (str): The prefix for log file names.
    use_info_handler (bool): Whether to use the info file handler.
    use_error_handler (bool): Whether to use the error file handler.
    use_stream_handler (bool): Whether to use the stream handler.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    today = datetime.now().strftime("%Y-%m-%d")

    file_prefix = file_prefix + "-" if file_prefix else ""

    # Common formatter
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )

    if info_handler:
        # Info handler
        info_log_filename = f"{file_prefix}info-{today}.log"
        info_handler = logging.FileHandler(info_log_filename)
        info_handler.setFormatter(formatter)
        info_handler.setLevel(logging.INFO)
        logger.addHandler(info_handler)

    if error_handler:
        # Error handler
        error_log_filename = f"{file_prefix}error-{today}.log"
        error_handler = logging.FileHandler(error_log_filename)
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        logger.addHandler(error_handler)

    if stream_handler:
        # Stream handler for console output
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.DEBUG)  # Set the level as per your requirement
        logger.addHandler(stream_handler)

    return logger


class Twilio:
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    content_sid = os.getenv("TWILIO_CONTENT_SID")
    service_sid = os.getenv("TWILIO_SERVICE_SID")
    if account_sid is not None and auth_token is not None:
        client = TwilioClient(account_sid, auth_token)
    else:
        client = None


# Set the default values for critical variables
NOTIFIER_LEVEL = "INFO"
LARGE_ORDER_THRESHOLD = 30
ERROR_NOTIFICATION_SETTINGS = {"url": None}
LIMIT_PRICE_BUFFER = 0.01
MAX_PRICE_MODIFICATION = 0.3
MODIFICATION_STEP_SIZE = 0.05
MODIFICATION_SLEEP_INTERVAL = 0.5
CACHE_INTERVAL = 3  # in seconds

# Create loggers
logger = create_logger("mylogger")  # Main logger
bs_logger = create_logger("bs", error_handler=False)  # Black-Scholes logger
latency_logger = create_logger("latency", error_handler=False, stream_handler=False)

# Get the list of scrips
scrips = get_ticker_file()
scrips["expiry_dt"] = pd.to_datetime(
    scrips[scrips.expiry != ""]["expiry"], format="%d%b%Y"
)
scrips["expiry_formatted"] = scrips["expiry_dt"].dt.strftime("%d%b%y")
scrips["expiry_formatted"] = scrips["expiry_formatted"].str.upper()

implemented_indices = [
    "NIFTY",
    "BANKNIFTY",
    "FINNIFTY",
    "MIDCPNIFTY",
    "SENSEX",
    "BANKEX",
    "INDIA VIX",
]

# Create a dictionary of token and symbol
token_symbol_dict = dict(zip(scrips["token"], scrips["symbol"]))

# Create a dictionary of token and exchange segment
token_exchange_dict = dict(zip(scrips["token"], scrips["exch_seg"]))

# Get the list of holidays
try:
    holidays = fetch_holidays()
except Exception as e:
    logger.error(f"Error while fetching holidays: {e}")
    holidays = pd.to_datetime([])

# Get the list of symbols
symbol_df = get_symbols()

# Create a thread local object
thread_local = local()

modification_fields = [
    "orderid",
    "variety",
    "symboltoken",
    "price",
    "ordertype",
    "transactiontype",
    "producttype",
    "exchange",
    "tradingsymbol",
    "quantity",
    "duration",
    "status",
]
order_placed = Event()
