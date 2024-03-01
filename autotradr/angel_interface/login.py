import pyotp
from time import sleep
import os
import logging
import functools
from datetime import datetime
from autotradr.config import logger
from autotradr.utils import current_time, notifier, set_error_notification_settings
from autotradr.angel_interface.active_session import ActiveSession
from autotradr.angel_interface.smart_connect import CustomSmartConnect


def login(user, pin, apikey, authkey, webhook_url=None, re_login=False):
    if ActiveSession.obj is None or ActiveSession.obj.refresh_token is None or re_login:
        authkey_object = pyotp.TOTP(authkey)
        obj = CustomSmartConnect(api_key=apikey)
        login_data = obj.generateSession(user, pin, authkey_object.now())
        if login_data["message"] != "SUCCESS":
            for attempt in range(2, 7):
                sleep(10)
                notifier(
                    f"Login attempt {attempt}. Error: {login_data['message']}",
                    webhook_url,
                    "ERROR",
                )
                login_data = obj.generateSession(user, pin, authkey_object.now())
                if login_data["message"] == "SUCCESS":
                    break
                if attempt == 6:
                    notifier(
                        f"Login failed. Error: {login_data['message']}",
                        webhook_url,
                        "CRUCIAL",
                    )
                    raise Exception("Login failed.")
        login_data.update(
            {"user": user, "pin": pin, "apikey": apikey, "authkey": authkey}
        )
        user_dir = f"{obj.userId}"
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)

        today = datetime.now().strftime("%Y-%m-%d")
        info_log_filename = f"{obj.userId}-info-{today}.log"
        error_log_filename = f"{obj.userId}-error-{today}.log"
        info_log_filename = os.path.join(user_dir, info_log_filename)
        error_log_filename = os.path.join(user_dir, error_log_filename)
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

        # Set obj and login_data in config
        ActiveSession.obj = obj
        ActiveSession.login_data = login_data

        # Edit file handlers
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)

        info_handler = logging.FileHandler(info_log_filename)
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(formatter)
        logger.addHandler(info_handler)

        error_handler = logging.FileHandler(error_log_filename)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

        set_error_notification_settings("user", obj.userId)
        notifier(
            f'Date: {current_time().strftime("%d %b %Y %H:%M:%S")}\nLogged in successfully.',
            webhook_url,
            "CRUCIAL",
        )
    else:
        notifier(
            f'Date: {current_time().strftime("%d %b %Y %H:%M:%S")}\nAlready logged in.',
            webhook_url,
            "CRUCIAL",
        )


def wait_for_login(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        while ActiveSession.login_data is None:
            logger.info(f"Function {func.__name__} waiting for login")
            sleep(1)
        return func(*args, **kwargs)

    return wrapper
