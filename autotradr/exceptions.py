class autotradrException(Exception):
    """Base class for other exceptions."""

    def __init__(self, message="autotradr Exception", code=500):
        self.message = message
        self.code = code
        super().__init__(self.message)


class ApiKeyNotFound(autotradrException):
    """Exception raised for missing API Key in the environment variables."""

    def __init__(self, message="API Key not found", code=404):
        super().__init__(message, code)


class APIFetchError(autotradrException):
    """Exception raised when unable to fetch data from Angel's API."""

    def __init__(self, message="No data returned from API", code=500):
        super().__init__(message, code)


class IntrinsicValueError(autotradrException):
    """Exception raised when unable to calculate the greeks because of mismatch of intrinsic value and market price."""

    def __init__(
        self, message="Mismatch of intrinsic value and market price", code=3501
    ):
        super().__init__(message, code)


class ScripsLocationError(autotradrException):
    """Exception raised when unable to locate something in the scrips file."""

    def __init__(
        self, message="Could not index scrips file", code=501, additional_info=""
    ):
        additional_info = (
            f"\nAdditional info: {additional_info}" if additional_info else ""
        )
        message = message + additional_info
        super().__init__(message, code)
