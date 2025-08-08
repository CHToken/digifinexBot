import time
import requests
import hmac
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import os
from dotenv import load_dotenv
from urllib.parse import urlencode

# Load environment variables
load_dotenv()

# Import required libraries
try:
    import ccxt
except ImportError:
    raise ImportError("ccxt library not found. Install it using: pip install ccxt")
try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas library not found. Install it using: pip install pandas")
try:
    import ta
    TA_LIB_AVAILABLE = True
except ImportError:
    try:
        from technical import ta as ta_fallback
        TA_LIB_AVAILABLE = False
    except ImportError:
        raise ImportError("Neither ta nor technical-analysis found. Install one using: pip install ta-lib or pip install technical-analysis")

# Configure logging to show only INFO and ERROR messages by default
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set to INFO to suppress most DEBUG logs

# Custom filter to allow specific DEBUG logs (e.g., critical events or RSI/EMA)
class EssentialLogFilter(logging.Filter):
    def filter(self, record):
        if record.levelno >= logging.INFO:
            return True
        if record.levelno == logging.DEBUG and ('critical' in record.msg.lower() or 'rsi' in record.msg.lower()):
            return True
        return False

# Set up rotating file handler (10MB per file, keep 5 backups)
handler = RotatingFileHandler('scalping_bot.log', maxBytes=10*1024*1024, backupCount=5)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
handler.addFilter(EssentialLogFilter())
logger.addHandler(handler)

# Add console handler for Railway logs
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# DigiFinex API credentials
API_KEY = os.getenv("DIGIFINEX_API_KEY", "")
SECRET_KEY = os.getenv("DIGIFINEX_SECRET_KEY", "")
BASE_URL = "https://openapi.digifinex.com/swap/v2"

# Trading parameters
SYMBOL = "BTCUSDTPERP"
CCXT_SYMBOL = "BTC/USDT:USDT"
MARGIN = 2.5  # For 0.001 BTC contract
LEVERAGE = 50
CONTRACT_SIZE = 0.001  # 1 contract = 0.001 BTC
PROFIT_TARGET = 0.05  # 3%
STOP_LOSS = 0.03  # 0.75%
MIN_HOLD_TIME = 200  # 5 minutes
TRADES_PER_DAY = 200  # Increased to ensure $1M monthly volume
MINIMUM_BALANCE = 2.5  # For 0.001 BTC contract
MARGIN_MODE = "isolated"

# Initialize exchange
try:
    exchange = ccxt.digifinex({
        'apiKey': API_KEY,
        'secret': SECRET_KEY,
        'enableRateLimit': True
    })
except Exception as e:
    logging.error(f"Failed to initialize exchange: {e}")
    raise

# Performance tracking
trades_log = []
total_volume = 0
total_fees = 0
total_profit = 0
last_balance_time = 0
cached_balance = 0
last_fee_time = 0
cached_taker_fee = 0.00005  # Default until fetched
current_order_id = None
tp_order_id = None
sl_order_id = None
cached_api_weights = None
last_api_weight_time = 0
server_time_offset = 0

def sync_server_time(attempts=3):
    offsets = []
    for _ in range(attempts):
        url = f"{BASE_URL}/public/time"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if data.get("code") == 0:
                server_time = int(data["data"])
                local_time = int(time.time() * 1000)
                offsets.append(server_time - local_time)
                time.sleep(0.5)
        except Exception as e:
            logging.error(f"Sync attempt failed: {e}")
    if offsets:
        global server_time_offset
        server_time_offset = sum(offsets) // len(offsets)
        logging.info(f"Server time synced. Offset: {server_time_offset} ms ({server_time_offset/1000:.3f} seconds)")
        return True
    return False
    
def get_timestamp():
    """Return current timestamp adjusted by server time offset."""
    return str(int(time.time() * 1000 + server_time_offset))

def get_api_weights():
    """Fetch API weights for rate limit management with caching."""
    global cached_api_weights, last_api_weight_time
    current_time = time.time()
    if cached_api_weights is None or current_time - last_api_weight_time > 3600:  # Cache for 1 hour
        url = f"{BASE_URL}/public/api_weight"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if data.get("code") == 0:
                logging.info("Fetched API weights successfully")
                cached_api_weights = data["data"]
                last_api_weight_time = current_time
                return cached_api_weights
            logging.error(f"Failed to fetch API weights: code={data.get('code')}, msg={data.get('msg')}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch API weights: {e}")
            return None
    return cached_api_weights

def check_api_weight_limit(endpoint):
    """Check the API weight for a given endpoint."""
    weights = get_api_weights()
    if weights:
        for item in weights:
            if item["path"] == f"/swap/v2{endpoint}":
                return item["weight"]
    return 1  # Default weight if fetch fails

def verify_signature(prehash, expected_signature):
    """Verify HMAC-SHA256 signature locally."""
    computed_signature = hmac.new(SECRET_KEY.encode(), prehash.encode(), hashlib.sha256).hexdigest()
    logging.debug(f"Computed signature: {computed_signature}, Expected: {expected_signature}")
    return computed_signature == expected_signature

def digifinex_signed_request(method, endpoint, params=None, retries=3, backoff_factor=2):
    global server_time_offset
    if server_time_offset == 0:
        sync_server_time()

    for attempt in range(retries):
        timestamp = get_timestamp()
        path = f"/swap/v2{endpoint}"
        query_string = urlencode(sorted(params.items())) if params and method.upper() == "GET" else ""
        body = json.dumps(params, sort_keys=True) if params and method.upper() == "POST" else ""
        prehash = timestamp + method.upper() + path + (query_string or body)
        signature = hmac.new(SECRET_KEY.encode(), prehash.encode(), hashlib.sha256).hexdigest()

        headers = {
            "ACCESS-KEY": API_KEY,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
            "User-Agent": "ScalpingBot/1.0",
            "Accept": "application/json"
        }

        url = f"{BASE_URL}{endpoint}" + (f"?{query_string}" if query_string else "")

        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers)
            else:
                response = requests.post(url, headers=headers, json=params)

            response.raise_for_status()
            data = response.json()

            if data.get("code") != 0:
                msg = data.get("msg", "").lower()
                if "timestamp" in msg or "access-timestamp" in msg:
                    logging.warning(f"Timestamp error detected: {msg}. Resyncing server time.")
                    sync_server_time()
                    continue
                logging.error(f"{method} {endpoint} failed: code={data.get('code')}, msg={data.get('msg')}")
                return None

            return data

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_body = e.response.text
            logging.error(f"HTTP {status_code} on {method} {endpoint}: {error_body}")
            if status_code == 429:
                retry_after = e.response.headers.get("Retry-After", "60")
                logging.warning(f"429 Too Many Requests. Retrying in {retry_after}s...")
                time.sleep(int(retry_after))
                continue
            elif status_code == 401:
                logging.warning(f"401 Unauthorized: {error_body}. Resyncing server time...")
                sync_server_time()
                continue
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
            return None

    logging.error(f"{method} {endpoint} failed after {retries} attempts")
    return None

def test_credentials():
    """Test API credentials with a simple authenticated request."""
    response = digifinex_signed_request("GET", "/account/balance")
    if response:
        logging.info("API credentials validated successfully")
        return True
    logging.error("API credential validation failed")
    return False

def set_margin_mode():
    """Set margin mode to isolated for BTCUSDTPERP."""
    position = check_position()
    if position:
        logging.error(f"Cannot set margin mode: Open position exists for {SYMBOL}")
        raise ValueError("Close existing positions before setting margin mode")
    params = {
        "instrument_id": SYMBOL,
        "margin_mode": MARGIN_MODE
    }
    response = digifinex_signed_request("POST", "/account/position_mode", params)
    if response:
        logging.info(f"Margin mode set to {MARGIN_MODE} for {SYMBOL}")
    else:
        logging.error("Failed to set margin mode")
        raise ValueError("Failed to set margin mode")
    time.sleep(5)

def set_leverage():
    """Set leverage for isolated margin mode."""
    params = {
        "instrument_id": SYMBOL,
        "leverage": LEVERAGE,
        "margin_mode": MARGIN_MODE
    }
    response = digifinex_signed_request("POST", "/account/leverage", params)
    if response:
        logging.info(f"Leverage set to {LEVERAGE}x for {SYMBOL}")
    else:
        logging.warning("Failed to set leverage. Proceeding with existing settings")
    time.sleep(5)

def get_trading_fee():
    """Fetch taker fee rate with caching."""
    global last_fee_time, cached_taker_fee
    current_time = time.time()
    if 'last_fee_time' not in globals() or current_time - last_fee_time > 10:
        params = {"instrument_id": SYMBOL}
        response = digifinex_signed_request("GET", "/account/trading_fee_rate", params)
        if response:
            cached_taker_fee = float(response["data"].get("taker_fee_rate", 0.00005))
            last_fee_time = current_time
        else:
            cached_taker_fee = 0.00005
        time.sleep(5)
    return cached_taker_fee

def get_current_price():
    """Fetch the latest price for BTCUSDTPERP."""
    url = f"{BASE_URL}/public/ticker"
    params = {"instrument_id": SYMBOL}
    try:
        response = requests.get(url, params=sorted(params.items()))
        response.raise_for_status()
        data = response.json()
        if data.get("code") == 0 and data.get("data"):
            ticker = data["data"]
            if ticker.get("instrument_id") == SYMBOL:
                logging.info("Fetched price successfully")
                return float(ticker["last"])
        logging.error(f"Failed to fetch price: code={data.get('code')}, msg={data.get('msg')}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch price: {e}")
        return None

def get_ohlcv(symbol, timeframe='5m', limit=100):
    """Fetch OHLCV data for technical analysis."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logging.info("Fetched OHLCV data successfully")
        return df
    except Exception as e:
        logging.error(f"Failed to fetch OHLCV: {e}")
        return None

def calculate_indicators(df):
    """Calculate RSI, EMAs, and Bollinger Bands, and log their statistics."""
    try:
        if TA_LIB_AVAILABLE:
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['ema9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
            df['ema21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
        else:
            df['rsi'] = ta_fallback.rsi(df['close'], period=14)
            df['ema9'] = ta_fallback.ema(df['close'], period=9)
            df['ema21'] = ta_fallback.ema(df['close'], period=21)
            # Fallback Bollinger Bands calculation
            df['bb_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
            df['bb_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()

        # Log indicator statistics
        logging.info("Calculated indicators successfully")
        logging.info(f"Number of data points: {len(df)}")
        
        # Latest values
        latest = df.iloc[-1]
        logging.info(f"Latest values - RSI: {latest['rsi']:.2f}, EMA9: {latest['ema9']:.2f}, "
                     f"EMA21: {latest['ema21']:.2f}, BB_Upper: {latest['bb_upper']:.2f}, "
                     f"BB_Lower: {latest['bb_lower']:.2f}")
        
        # Statistical summary
        stats = df[['rsi', 'ema9', 'ema21']].describe()
        logging.info("Indicator statistics:")
        logging.info(f"RSI - Mean: {stats['rsi']['mean']:.2f}, Min: {stats['rsi']['min']:.2f}, "
                     f"Max: {stats['rsi']['max']:.2f}, Std: {stats['rsi']['std']:.2f}")
        logging.info(f"EMA9 - Mean: {stats['ema9']['mean']:.2f}, Min: {stats['ema9']['min']:.2f}, "
                     f"Max: {stats['ema9']['max']:.2f}, Std: {stats['ema9']['std']:.2f}")
        logging.info(f"EMA21 - Mean: {stats['ema21']['mean']:.2f}, Min: {stats['ema21']['min']:.2f}, "
                     f"Max: {stats['ema21']['max']:.2f}, Std: {stats['ema21']['std']:.2f}")

        return df
    except Exception as e:
        logging.error(f"Failed to calculate indicators: {e}")
        return None

def place_order(side, size, order_type=1):
    """Place a market order to open a position."""
    global current_order_id
    type_map = {
        "open_long": 1,
        "open_short": 2
    }
    if side not in type_map:
        logging.error(f"Invalid order side: {side}")
        return None
    params = {
        "instrument_id": SYMBOL,
        "type": type_map[side],
        "order_type": order_type,
        "size": size
    }
    response = digifinex_signed_request("POST", "/trade/order_place", params)
    if response:
        order_id = response["data"]
        current_order_id = order_id
        return order_id
    return None

def add_algo_order(order_id, entry_price, side):
    """Set TP and SL for an open position."""
    global tp_order_id, sl_order_id
    tp_price = entry_price * (1 + PROFIT_TARGET) if side == "open_long" else entry_price * (1 - PROFIT_TARGET)
    sl_price = entry_price * (1 - STOP_LOSS) if side == "open_long" else entry_price * (1 + STOP_LOSS)
    params = {
        "order_id": order_id,
        "profit_trigger_type": 3,
        "profit_trigger_price": str(round(tp_price, 2)),
        "stop_trigger_type": 3,
        "stop_trigger_price": str(round(sl_price, 2))
    }
    response = digifinex_signed_request("POST", "/follow/add_algo", params)
    if response and response.get("data", {}).get("result") == 1:
        algos = response["data"].get("algos", [])
        if len(algos) == 2:
            tp_order_id, sl_order_id = algos
            return True
        logging.warning(f"Unexpected algo order response: {algos}")
    return False

def cancel_algo_order():
    """Cancel TP and SL orders."""
    global tp_order_id, sl_order_id
    if not (tp_order_id or sl_order_id):
        return True
    params = {
        "order_id": current_order_id,
        "stop_profit": 1 if tp_order_id else 0,
        "stop_loss": 1 if sl_order_id else 0
    }
    response = digifinex_signed_request("POST", "/follow/cancel_algo", params)
    if response and response.get("data", {}).get("result") == 1:
        tp_order_id = None
        sl_order_id = None
        return True
    return False

def close_order(order_id):
    """Close an existing position."""
    global current_order_id, tp_order_id, sl_order_id
    if not order_id:
        logging.error("No open order ID to close")
        return False
    if not cancel_algo_order():
        logging.warning("Failed to cancel TP/SL orders. Proceeding to close position")
    params = {
        "open_order_id": order_id
    }
    response = digifinex_signed_request("POST", "/follow/close_order", params)
    if response and response.get("data", {}).get("res") == 1:
        current_order_id = None
        tp_order_id = None
        sl_order_id = None
        return True
    return False

def cancel_order(order_id):
    """Cancel an existing order."""
    params = {
        "instrument_id": SYMBOL,
        "order_id": order_id
    }
    response = digifinex_signed_request("POST", "/trade/cancel_order", params)
    if response:
        return response["data"]
    return None

def check_position():
    """Check open positions for BTCUSDTPERP (no params to avoid API issue)."""
    response = digifinex_signed_request("GET", "/account/positions")
    if response:
        positions = response.get("data", [])
        for pos in positions:
            if pos["instrument_id"] == SYMBOL and float(pos["position"]) > 0:
                logging.info(
                    f"Open position found: side={pos['side']}, size={pos['position']}, timestamp={pos['timestamp']}"
                )
                return pos
        logging.info("No open positions found for BTCUSDTPERP")
        return None

    logging.warning("Failed to fetch positions. Trying CCXT fallback...")
    try:
        positions = exchange.fetch_positions([CCXT_SYMBOL])
        for pos in positions:
            if pos["symbol"] == CCXT_SYMBOL and float(pos["contracts"]) > 0:
                logging.info(f"Open position found via CCXT: {pos}")
                return pos
        logging.info("No open positions found via CCXT")
        return None
    except Exception as e:
        logging.error(f"CCXT fetch_positions failed: {e}")
        return None

def get_account_balance():
    """Fetch futures account balance with caching."""
    global last_balance_time, cached_balance
    current_time = time.time()
    if 'last_balance_time' not in globals() or current_time - last_balance_time > 30:  # Increased to 30s
        response = digifinex_signed_request("GET", "/account/balance")
        if response:
            data = response.get("data")
            if isinstance(data, dict):
                cached_balance = float(data.get("avail_balance", "0"))
            elif isinstance(data, list):
                for item in data:
                    if item.get("currency") == "USDT":
                        cached_balance = float(item.get("avail_balance", "0"))
                        break
                else:
                    cached_balance = 0
            else:
                cached_balance = 0
            last_balance_time = current_time
        else:
            cached_balance = 0
        time.sleep(5)
    logging.info(f"Current balance: ${cached_balance:.2f}")
    return cached_balance

def validate_setup():
    """Validate API credentials, margin mode, leverage, and balance."""
    if not API_KEY or not SECRET_KEY:
        logging.error("API_KEY or SECRET_KEY missing in environment variables")
        raise ValueError("API_KEY and SECRET_KEY must be set in .env")
    if not test_credentials():
        raise ValueError("API credential validation failed")
    try:
        set_margin_mode()
    except ValueError as e:
        logging.error(f"Margin mode setup failed: {e}")
        raise
    try:
        set_leverage()
    except ValueError:
        logging.warning("Leverage setup failed. Proceeding with existing settings")
    balance = get_account_balance()
    current_price = get_current_price()
    if current_price is None:
        raise ValueError("Failed to fetch current price")
    size = 1
    required_margin = (size * CONTRACT_SIZE * current_price) / LEVERAGE
    if balance < required_margin:
        raise ValueError(f"Insufficient balance: ${balance} < ${required_margin:.2f}")
    if balance < MINIMUM_BALANCE:
        logging.warning(f"Low balance: ${balance}. Consider adding funds")
    logging.info(f"Setup validated. Balance: ${balance:.2f}, Price: ${current_price:.2f}, Margin: ${required_margin:.2f}")

def main():
    global total_volume, total_fees, total_profit, current_order_id
    daily_trades = 0
    last_position_check = 0
    cached_position = None
    try:
        validate_setup()
        capital = get_account_balance()
        logging.info(f"Starting capital: ${capital:.2f}")

        while daily_trades < TRADES_PER_DAY and capital >= MINIMUM_BALANCE:
            try:
                current_time = time.time()
                current_price = get_current_price()
                if current_price is None:
                    logging.warning("Failed to fetch price. Retrying in 60s")
                    time.sleep(60)
                    continue

                size = 1
                required_margin = (size * CONTRACT_SIZE * current_price) / LEVERAGE
                logging.debug(f"Size: {size} contract, Margin: ${required_margin:.2f}")

                if capital < required_margin:
                    logging.warning(f"Insufficient capital: ${capital:.2f} < ${required_margin:.2f}")
                    break

                df = get_ohlcv(CCXT_SYMBOL)
                if df is None or len(df) < 50:
                    logging.warning("Insufficient OHLCV data. Retrying in 60s")
                    time.sleep(60)
                    continue
                df = calculate_indicators(df)
                if df is None:
                    time.sleep(60)
                    continue
                latest = df.iloc[-1]
                prev = df.iloc[-2]

                long_signal = (latest['rsi'] < 50 and latest['ema9'] < latest['ema21']) or \
                             (latest['close'] <= latest['bb_lower'] and latest['rsi'] < 55)
                short_signal = (latest['rsi'] > 55 and latest['ema9'] > latest['ema21']) or \
                              (latest['close'] >= latest['bb_upper'] and latest['rsi'] > 50)

                logging.debug(f"RSI: {latest['rsi']:.2f}, EMA9: {latest['ema9']:.2f}, EMA21: {latest['ema21']:.2f}, "
                             f"BB_Upper: {latest['bb_upper']:.2f}, BB_Lower: {latest['bb_lower']:.2f}, "
                             f"Long: {long_signal}, Short: {short_signal}")

                if current_time - last_position_check > 30:
                    position = check_position()
                    cached_position = position
                    last_position_check = current_time
                else:
                    position = cached_position

                if position and current_order_id:
                    entry_price = float(position.get('avg_cost', position.get('entryPrice', 0)))
                    entry_time = float(position.get('timestamp', time.time() * 1000)) / 1000
                    position_side = position.get('side', 'long').lower()

                    if current_time - entry_time >= MIN_HOLD_TIME:
                        success = close_order(current_order_id)
                        if success:
                            notional = size * CONTRACT_SIZE * current_price
                            fee_rate = get_trading_fee()
                            fees = notional * fee_rate * 2
                            profit = ((current_price - entry_price) * size * CONTRACT_SIZE * LEVERAGE
                                     if position_side == "long" else
                                     (entry_price - current_price) * size * CONTRACT_SIZE * LEVERAGE)
                            profit -= fees
                            total_profit += profit
                            total_fees += fees
                            total_volume += notional * 2
                            trades_log.append({
                                "time": datetime.now(),
                                "side": position_side,
                                "entry_price": entry_price,
                                "exit_price": current_price,
                                "profit": profit,
                                "fees": fees,
                                "volume": notional * 2
                            })
                            logging.info(f"Trade closed: Profit=${profit:.2f}, Volume=${notional*2:.2f}")
                            daily_trades += 1
                        else:
                            logging.warning(f"Failed to close order: {current_order_id}")
                    else:
                        logging.info("Position open. Skipping new trade")

                elif long_signal or short_signal:
                    side = "open_long" if long_signal else "open_short"
                    order_id = place_order(side, size)
                    if order_id:
                        if add_algo_order(order_id, current_price, side):
                            logging.info(f"Opened {side}: size={size}, order_id={order_id}")
                        else:
                            logging.warning("Failed to set TP/SL. Canceling order")
                            cancel_order(order_id)
                            current_order_id = None
                        time.sleep(10)
                    else:
                        logging.warning("Order placement failed")

                capital = get_account_balance()
                if capital < MINIMUM_BALANCE:
                    logging.warning(f"Capital below minimum: ${capital:.2f}")
                    break

                time.sleep(5)  # Reduced to check signals more frequently

            except Exception as e:
                logging.error(f"Main loop error: {e}")
                time.sleep(60)

        pd.DataFrame(trades_log).to_csv("trades_log.csv", index=False)
        logging.info(f"Daily trades: {daily_trades}, Volume: ${total_volume:.2f}, "
                    f"Profit: ${total_profit:.2f}, Fees: ${total_fees:.2f}")

    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    logging.info("Scalping bot main.py is running")
    try:
        while True:
            main()
            logging.info("Daily trading completed. Sleeping for 1h")
            time.sleep(3600)
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
        if current_order_id:
            cancel_algo_order()
            close_order(current_order_id)
    except Exception as e:
        logging.error(f"Bot crashed: {e}")
        if current_order_id:
            cancel_algo_order()
            close_order(current_order_id)

