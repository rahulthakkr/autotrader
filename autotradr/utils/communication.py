import json
import traceback
import functools
import numpy as np
import requests
from autotradr import config
from autotradr.config import logger, Twilio


def log_error(notify: bool = False, raise_error: bool = False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                message = f"Error in function {func.__name__}: {e}\nTraceback:{traceback.format_exc()}"
                if notify:
                    notifier(
                        message,
                        webhook_url=config.ERROR_NOTIFICATION_SETTINGS["url"],
                        level="ERROR",
                    )
                else:
                    logger.error(message)
                if raise_error:
                    raise e

        return wrapper

    return decorator


def notifier(
    message: str,
    webhook_url: str | list[str] = None,
    level: str = "INFO",
    send_whatsapp: bool = False,
):
    levels = ["INFO", "CRUCIAL", "ERROR"]
    if isinstance(webhook_url, (list, tuple, set, np.ndarray)):
        notification_urls = [
            *filter(lambda x: x is not None and x is not False, webhook_url)
        ]
    elif isinstance(webhook_url, str) and webhook_url != "":
        notification_urls = [webhook_url]
    else:  # webhook_url is None or False
        notification_urls = []

    if level.lower() == "crucial":
        logger.info(message)
    else:
        getattr(logger, level.lower())(message)

    # If level is lower than config.NOTIFIER_LEVEL, don't send notification. Logging has already been done.
    if levels.index(level) < levels.index(config.NOTIFIER_LEVEL):
        return

    data = {"content": message}
    for url in notification_urls:
        try:
            requests.post(
                url,
                data=json.dumps(data),
                headers={"Content-Type": "application/json"},
            )
        except requests.exceptions.SSLError as e:
            logger.error(
                f"Error while sending notification: {e}",
                exc_info=(type(e), e, e.__traceback__),
            )

    if send_whatsapp:
        send_whatsapp_message(message, config.ERROR_NOTIFICATION_SETTINGS["whatsapp"])


@log_error(notify=True)
def send_whatsapp_message(message: str, to: str):
    if to is None:
        return
    elif not to.startswith("+91"):
        to = "+91" + to
    if "\nTraceback:" in message:
        message = message.split("\nTraceback:")[0]
    Twilio.client.messages.create(
        content_sid=Twilio.content_sid,
        from_=Twilio.service_sid,
        content_variables=json.dumps({"1": message}),
        to=f"whatsapp:{to}",
    )
