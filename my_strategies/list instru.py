
from openalgo import api
import pandas as pd
client = api(api_key='74f041f84571131d5bf0ae7bff3ab47802c172c4e34460f40a12cc5fe04e6fc2', host='http://127.0.0.1:5000')

for underlying, exch in [('NIFTY','NSE_INDEX'), ('SENSEX','BSE_INDEX')]:
    try:
        exps = discover_expiries(underlying, exch)   # use function from script
        print(underlying, 'expiries found:', exps[:5])
    except Exception as e:
        print('discover_expiries error for', underlying, ':', e)

