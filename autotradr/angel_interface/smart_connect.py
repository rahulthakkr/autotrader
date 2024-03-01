from SmartApi import SmartConnect


class CustomSmartConnect(SmartConnect):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_routes()

    def _update_routes(self):
        self._routes.update(
            {
                "api.market.quote": "https://apiconnect.angelbroking.com/rest/secure/angelbroking/market/v1/quote/"
            }
        )

    def market_data(self, mode, exchange_tokens):
        params = {"mode": mode, "exchangeTokens": exchange_tokens}
        return self._postRequest("api.market.quote", params=params)
