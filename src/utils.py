"""
utils.py - Real-time Fraud Detection
FIXED: Chronological state isolation for bulk evaluation
"""

import os
import pickle
import warnings
from datetime import datetime, timedelta
from collections import deque

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# CONSTANTS
# ============================================================

DIVISIONS = [
    "Dhaka", "Chattogram", "Khulna", "Rajshahi",
    "Barishal", "Sylhet", "Rangpur", "Mymensingh",
]

DIVISION_COORDS = {
    "Dhaka": (23.8103, 90.4125),
    "Chattogram": (22.3569, 91.7832),
    "Khulna": (22.8456, 89.5403),
    "Rajshahi": (24.3636, 88.6241),
    "Barishal": (22.7010, 90.3535),
    "Sylhet": (24.8949, 91.8687),
    "Rangpur": (25.7439, 89.2752),
    "Mymensingh": (24.7471, 90.4203),
}

BKASH_LIMITS = {"per_transaction_max": 50000}

# ============================================================
# SAFE CONVERTERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        value = float(value)
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def safe_string(value, default=""):
    if pd.isna(value):
        return default
    value = str(value).strip()
    return value if value else default


def calculate_slope(values):
    values = list(values)
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.asarray(values, dtype=float)
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return 0.0


# ============================================================
# DISTANCE
# ============================================================

def calc_distance(loc1, loc2):
    if loc1 not in DIVISION_COORDS or loc2 not in DIVISION_COORDS:
        return 0.0

    from math import radians, sin, cos, sqrt, atan2

    lat1, lon1 = DIVISION_COORDS[loc1]
    lat2, lon2 = DIVISION_COORDS[loc2]
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (sin(dlat / 2) ** 2
         + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# ============================================================
# USER STATE CLASS
# ============================================================

class UserState:

    def __init__(self, user_id):
        self.user_id = user_id

        # Cumulative
        self.transaction_count = 0
        self.total_amount = 0.0
        self.all_time_amounts = []
        self.all_time_hours = []
        self.all_time_merchant_risks = []

        # Rolling history
        self.window_size = 100
        self.amount_history = deque(maxlen=self.window_size)
        self.location_history = deque(maxlen=self.window_size)
        self.device_history = deque(maxlen=self.window_size)
        self.hour_history = deque(maxlen=self.window_size)
        self.merchant_history = deque(maxlen=self.window_size)
        self.merchant_risk_history = deque(maxlen=self.window_size)
        self.timestamps = deque(maxlen=self.window_size)
        self.velocity_timestamps = deque(maxlen=500)

        # All-time counters
        self.all_merchant_counts = {}
        self.all_location_counts = {}
        self.all_device_counts = {}
        self.unique_merchants = set()
        self.unique_locations = set()
        self.unique_devices = set()
        self.device_first_seen = {}

        # Merchant stats
        self.merchant_avg_amounts = {}
        self.merchant_counts = {}

        # Today
        self.today_date = None
        self.today_merchants = set()
        self.today_locations = set()

        # Profile
        self.home_location = "Dhaka"
        self.preferred_hour = 12
        self.account_age_days = 365
        self.base_amount = 2000
        self.primary_device = "D0001"

        # Historical derived
        self.historical_amount_pct_change = deque(maxlen=100)
        self.historical_amount_slope = deque(maxlen=100)
        self.historical_velocity_5min = deque(maxlen=100)
        self.historical_velocity_accel = deque(maxlen=100)
        self.historical_txn_burst = deque(maxlen=100)
        self.historical_merchant_switch_amount = deque(maxlen=100)
        self.historical_merchant_switch = deque(maxlen=100)

        # Last values
        self.last_timestamp = None
        self.last_location = None
        self.last_device = None
        self.last_merchant = None

        # Track latest timestamp for cache invalidation
        self.latest_built_timestamp = None

    # --------------------------------------------------------
    def get_velocity_5min(self, current_time):
        cutoff = current_time - timedelta(minutes=5)
        return sum(1 for t in self.velocity_timestamps if cutoff <= t < current_time)

    def get_velocity_1hr(self, current_time):
        cutoff = current_time - timedelta(minutes=60)
        return sum(1 for t in self.velocity_timestamps if cutoff <= t < current_time)

    def get_transaction_count_10min(self, current_time):
        cutoff = current_time - timedelta(minutes=10)
        return sum(1 for t in self.velocity_timestamps if cutoff <= t < current_time)

    def get_expanding_mean(self, values):
        values = list(values)
        return float(np.mean(values)) if values else 0.0

    def get_expanding_std(self, values):
        values = list(values)
        if len(values) <= 1:
            return 1.0
        return float(np.std(values, ddof=1))

    # --------------------------------------------------------
    def add_transaction(self, amount, location, device_id,
                        merchant_id, merchant_risk, hour, timestamp):

        amount = safe_float(amount)
        merchant_risk = safe_float(merchant_risk, 0.5)
        hour = safe_int(hour, 12)
        timestamp = pd.to_datetime(timestamp)

        previous_amounts = list(self.amount_history)
        previous_merchants = list(self.merchant_history)
        previous_velocity_5min = list(self.historical_velocity_5min)

        # amount_pct_change
        if len(previous_amounts) >= 2:
            prev = previous_amounts[-1]
            prev_prev = previous_amounts[-2]
            pct_change = (prev - prev_prev) / prev_prev if prev_prev != 0 else np.nan
        else:
            pct_change = np.nan
        if not np.isfinite(pct_change):
            pct_change = 0.0

        # merchant_switch
        switch = int(previous_merchants[-1] != merchant_id) if previous_merchants else 0
        merchant_switch_amount = switch * amount

        # velocity_5min
        velocity_5min = self.get_velocity_5min(timestamp)

        # velocity_accel
        if len(previous_velocity_5min) >= 2:
            v_prev = previous_velocity_5min[-1]
            v_prev_prev = previous_velocity_5min[-2]
            velocity_accel = (v_prev - v_prev_prev) / v_prev_prev if v_prev_prev != 0 else 0.0
        else:
            velocity_accel = 0.0
        velocity_accel = np.clip(velocity_accel, -5, 5)

        # Store
        self.historical_amount_pct_change.append(pct_change)
        self.historical_merchant_switch_amount.append(merchant_switch_amount)
        self.historical_merchant_switch.append(switch)
        self.historical_velocity_5min.append(velocity_5min)
        self.historical_velocity_accel.append(velocity_accel)
        self.historical_txn_burst.append(int(velocity_5min > 3))

        # amount_slope
        if len(previous_amounts) >= 2:
            slope_values = previous_amounts[-10:]
            current_slope = calculate_slope(slope_values)
        else:
            current_slope = 0.0
        self.historical_amount_slope.append(current_slope)

        # Cumulative
        self.transaction_count += 1
        self.total_amount += amount
        self.all_time_amounts.append(amount)
        self.all_time_hours.append(hour)
        self.all_time_merchant_risks.append(merchant_risk)

        # Today
        current_date = timestamp.date()
        if self.today_date != current_date:
            self.today_date = current_date
            self.today_merchants = set()
            self.today_locations = set()
        self.today_merchants.add(merchant_id)
        self.today_locations.add(location)

        # Rolling
        self.amount_history.append(amount)
        self.location_history.append(location)
        self.device_history.append(device_id)
        self.hour_history.append(hour)
        self.merchant_history.append(merchant_id)
        self.merchant_risk_history.append(merchant_risk)
        self.timestamps.append(timestamp)
        self.velocity_timestamps.append(timestamp)

        # Counters
        self.all_merchant_counts[merchant_id] = self.all_merchant_counts.get(merchant_id, 0) + 1
        self.all_location_counts[location] = self.all_location_counts.get(location, 0) + 1
        self.all_device_counts[device_id] = self.all_device_counts.get(device_id, 0) + 1

        self.unique_merchants.add(merchant_id)
        self.unique_locations.add(location)
        self.unique_devices.add(device_id)

        if device_id not in self.device_first_seen:
            self.device_first_seen[device_id] = timestamp

        # Merchant average
        if merchant_id not in self.merchant_avg_amounts:
            self.merchant_avg_amounts[merchant_id] = 0.0
            self.merchant_counts[merchant_id] = 0

        count = self.merchant_counts[merchant_id]
        old_avg = self.merchant_avg_amounts[merchant_id]
        self.merchant_avg_amounts[merchant_id] = (old_avg * count + amount) / (count + 1)
        self.merchant_counts[merchant_id] += 1

        # Last
        self.last_timestamp = timestamp
        self.last_location = location
        self.last_device = device_id
        self.last_merchant = merchant_id

        if self.latest_built_timestamp is None or timestamp > self.latest_built_timestamp:
            self.latest_built_timestamp = timestamp


# ============================================================
# LOAD MODEL & SUPPORT FILES
# ============================================================

print("\n[1] Loading model and supporting files...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_file(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


model_path = find_file([
    os.path.join(BASE_DIR, "..", "models", "best_model_fixed.pkl"),
    "E:/Fraud_Project2/models/best_model_fixed.pkl",
    "../models/best_model_fixed.pkl",
    "models/best_model_fixed.pkl",
])

if model_path is None:
    raise FileNotFoundError("ERROR: best_model_fixed.pkl not found.")

with open(model_path, "rb") as f:
    model_data = pickle.load(f)

if isinstance(model_data, dict):
    model = model_data.get("model")
    feature_cols = model_data.get("features", [])
    THRESHOLD = float(model_data.get("threshold", 0.55))
    print(f"Model: {model_data.get('name', 'Unknown')}")
else:
    model = model_data
    feature_cols = []
    THRESHOLD = 0.55

print(f"Features: {len(feature_cols)}")
print(f"Threshold: {THRESHOLD:.2f}")

# ------------------------------------------------------------
# ENCODERS
# ------------------------------------------------------------

encoder_path = find_file([
    os.path.join(BASE_DIR, "..", "models", "encoders.pkl"),
    "E:/Fraud_Project2/models/encoders.pkl",
    "../models/encoders.pkl",
    "models/encoders.pkl",
])

with open(encoder_path, "rb") as f:
    encoders = pickle.load(f)
print("Encoders loaded")

frequency_maps = {}
for key in ["user_id_freq", "merchant_id_freq", "device_id_freq"]:
    if key in encoders:
        frequency_maps[key] = encoders[key]
print(f"Frequency maps: {list(frequency_maps.keys())}")

# ------------------------------------------------------------
# CLIP BOUNDS
# ------------------------------------------------------------

clip_path = find_file([
    os.path.join(BASE_DIR, "..", "models", "clip_bounds.pkl"),
    "E:/Fraud_Project2/models/clip_bounds.pkl",
    "../models/clip_bounds.pkl",
    "models/clip_bounds.pkl",
])

with open(clip_path, "rb") as f:
    clip_bounds = pickle.load(f)
print("Clip bounds loaded")

# ------------------------------------------------------------
# MERCHANTS
# ------------------------------------------------------------

print("\n[2] Loading data sources...")

merchant_path = find_file([
    os.path.join(BASE_DIR, "..", "data", "merchants.csv"),
    "E:/Fraud_Project2/data/merchants.csv",
    "../data/merchants.csv",
    "data/merchants.csv",
])

merchants_by_id = {}
if merchant_path is not None:
    merch_df = pd.read_csv(merchant_path)
    if "merchant_id" in merch_df.columns:
        merchants_by_id = merch_df.set_index("merchant_id").to_dict("index")
    print(f"Loaded {len(merchants_by_id)} merchants")


def lookup_merchant(merchant_id):
    info = merchants_by_id.get(merchant_id)
    if info is None:
        return "Unknown", 0.5
    return info.get("category", "Unknown"), safe_float(info.get("merchant_risk", 0.5), 0.5)


# ------------------------------------------------------------
# USERS
# ------------------------------------------------------------

user_path = find_file([
    os.path.join(BASE_DIR, "..", "data", "users.csv"),
    "E:/Fraud_Project2/data/users.csv",
    "../data/users.csv",
    "data/users.csv",
])

user_meta = {}
if user_path is not None:
    u_df = pd.read_csv(user_path)
    for _, row in u_df.iterrows():
        user_meta[row["user_id"]] = {
            "segment": row.get("segment", "office_worker"),
            "risk_profile": row.get("risk_profile", "medium"),
            "home_location": row.get("home_location", "Dhaka"),
            "primary_device": row.get("primary_device", "D0001"),
            "base_amount": safe_float(row.get("base_amount", 2000), 2000),
            "preferred_hour": safe_int(row.get("preferred_hour", 12), 12),
            "account_age_days": safe_int(row.get("account_age_days", 365), 365),
        }
    print(f"Loaded {len(user_meta)} users")


def get_user_profile(user_id):
    if user_id in user_meta:
        return user_meta[user_id]
    return {
        "segment": "office_worker",
        "risk_profile": "medium",
        "home_location": "Dhaka",
        "primary_device": "D0001",
        "base_amount": 2000,
        "preferred_hour": 12,
        "account_age_days": 365,
    }


# ------------------------------------------------------------
# TRANSACTIONS CSV (for historical state)
# ------------------------------------------------------------

print("\n[3] Loading transaction history...")

transaction_path = find_file([
    os.path.join(BASE_DIR, "..", "data", "transactions.csv"),
    "E:/Fraud_Project2/data/transactions.csv",
    "../data/transactions.csv",
    "data/transactions.csv",
])

transactions_df = None
if transaction_path is not None:
    transactions_df = pd.read_csv(transaction_path)
    if "timestamp" in transactions_df.columns:
        transactions_df["timestamp"] = pd.to_datetime(transactions_df["timestamp"], errors="coerce")
    print(f"Loaded {len(transactions_df)} transactions")


# ============================================================
# BUILD USER STATE
# ============================================================

def build_user_state(user_id, before_timestamp=None):
    """
    Build user state from historical transactions.

    If before_timestamp is provided, only transactions STRICTLY
    BEFORE that timestamp are used (prevents future data leakage).
    """
    profile = get_user_profile(user_id)
    state = UserState(user_id)

    state.home_location = profile["home_location"]
    state.preferred_hour = profile["preferred_hour"]
    state.account_age_days = profile["account_age_days"]
    state.base_amount = profile["base_amount"]
    state.primary_device = profile["primary_device"]

    if transactions_df is None or len(transactions_df) == 0:
        return state

    user_txns = transactions_df[
        transactions_df["user_id"].astype(str) == str(user_id)
    ].copy()

    if len(user_txns) == 0:
        return state

    # Only transactions strictly before current timestamp
    if before_timestamp is not None:
        before_timestamp = pd.to_datetime(before_timestamp, errors="coerce")
        if pd.notna(before_timestamp):
            user_txns = user_txns[user_txns["timestamp"] < before_timestamp]

    user_txns = user_txns.sort_values("timestamp")

    for _, txn in user_txns.iterrows():
        merchant_id = safe_string(txn.get("merchant_id", "M0001"), "M0001")
        merchant_category, merchant_risk = lookup_merchant(merchant_id)

        timestamp = pd.to_datetime(txn.get("timestamp", datetime.now()), errors="coerce")
        if pd.isna(timestamp):
            continue

        location = safe_string(txn.get("location", profile["home_location"]), profile["home_location"])
        device_id = safe_string(txn.get("device_id", "D0001"), "D0001")
        amount = safe_float(txn.get("amount", 0))
        hour = safe_int(txn.get("hour", timestamp.hour), timestamp.hour)

        if "merchant_risk" in txn and pd.notna(txn["merchant_risk"]):
            merchant_risk = safe_float(txn["merchant_risk"], merchant_risk)

        state.add_transaction(
            amount=amount,
            location=location,
            device_id=device_id,
            merchant_id=merchant_id,
            merchant_risk=merchant_risk,
            hour=hour,
            timestamp=timestamp
        )

    return state


# ============================================================
# ROLLING HELPERS
# ============================================================

def shifted_rolling_mean(values, window):
    values = list(values)
    if not values:
        return 0.0
    return float(np.mean(values[-window:]))


def shifted_rolling_std(values, window):
    values = list(values)
    if len(values) < 2:
        return 0.0
    selected = values[-window:]
    if len(selected) < 2:
        return 0.0
    return float(np.std(selected, ddof=1))


def previous_txn_intervals(timestamps, max_items=3):
    timestamps = list(timestamps)
    if len(timestamps) < 2:
        return []
    intervals = []
    start = max(1, len(timestamps) - max_items)
    for i in range(start, len(timestamps)):
        delta = (timestamps[i] - timestamps[i - 1]).total_seconds() / 60.0
        intervals.append(delta)
    return intervals


# ============================================================
# COMPUTE FEATURES
# ============================================================

def compute_features(state, amount, location, merchant_id, merchant_risk,
                     merchant_category, hour, device_id, timestamp,
                     minutes_since_last):

    amount = safe_float(amount)
    merchant_risk = safe_float(merchant_risk, 0.5)
    hour = safe_int(hour, 12)
    timestamp = pd.to_datetime(timestamp)

    features = {}
    profile = get_user_profile(state.user_id)

    # BASIC
    features["amount"] = amount
    features["hour"] = hour
    features["day"] = timestamp.weekday()
    features["location"] = location
    features["user_segment"] = profile["segment"]
    features["user_risk"] = profile["risk_profile"]
    features["merchant_category"] = merchant_category
    features["merchant_risk"] = merchant_risk
    features["account_age_days"] = state.account_age_days

    if device_id in state.device_first_seen:
        features["device_age_days"] = (timestamp - state.device_first_seen[device_id]).days
    else:
        features["device_age_days"] = 0

    features["exceeds_limit_flag"] = int(amount > BKASH_LIMITS["per_transaction_max"])

    # AMOUNT
    prior_amounts = list(state.amount_history)

    features["amount_ma3"] = shifted_rolling_mean(prior_amounts, 3)
    features["amount_ma5"] = shifted_rolling_mean(prior_amounts, 5)
    features["amount_ma7"] = shifted_rolling_mean(prior_amounts, 7)
    features["amount_ma10"] = shifted_rolling_mean(prior_amounts, 10)
    features["amount_ma14"] = shifted_rolling_mean(prior_amounts, 14)

    features["amount_trend"] = features["amount_ma3"] - features["amount_ma5"]
    features["amount_trend_ratio"] = amount / (features["amount_ma3"] + 1)

    pct_hist = list(state.historical_amount_pct_change)
    features["amount_pct_change"] = pct_hist[-1] if pct_hist else 0.0
    features["amount_pct_change_ma3"] = float(np.mean(pct_hist[-3:])) if pct_hist else 0.0

    past_amounts = list(state.all_time_amounts)
    past_mean = state.get_expanding_mean(past_amounts)
    past_std = state.get_expanding_std(past_amounts)
    features["amount_zscore_cum"] = (amount - past_mean) / (past_std + 1)

    recent_max = max(prior_amounts[-10:]) if len(prior_amounts) >= 10 else (max(prior_amounts) if prior_amounts else 0.0)
    features["amount_to_recent_max"] = amount / (recent_max + 1)

    slope_hist = list(state.historical_amount_slope)
    features["amount_slope"] = slope_hist[-1] if slope_hist else 0.0

    if len(slope_hist) >= 2 and slope_hist[-2] != 0:
        features["amount_slope_change"] = (slope_hist[-1] - slope_hist[-2]) / slope_hist[-2]
    else:
        features["amount_slope_change"] = 0.0

    features["amount_consistency"] = 1 - (abs(amount - past_mean) / (past_std + 1))
    features["amount_volatility"] = shifted_rolling_std(prior_amounts, 5)

    hist_avg = state.total_amount / state.transaction_count if state.transaction_count > 0 else 0.0
    features["amount_cumsum"] = state.total_amount + amount
    features["expected_cumsum"] = hist_avg * (state.transaction_count + 1)
    features["cumsum_deviation"] = features["amount_cumsum"] - features["expected_cumsum"]
    features["avg_amount"] = hist_avg

    # CATEGORY DIVERSITY
    category_counts = {}
    for mid in state.unique_merchants:
        cat, _ = lookup_merchant(mid)
        category_counts[cat] = category_counts.get(cat, 0) + 1

    features["category_diversity"] = (
        len(category_counts) / (state.transaction_count + 1)
        if state.transaction_count > 0 else 0.0
    )

    features["primary_device_ratio"] = (
        state.all_device_counts.get(state.primary_device, 0) / (state.transaction_count + 1)
        if state.transaction_count > 0 else 0.0
    )

    # HOUR
    hour_hist = list(state.all_time_hours)
    hour_mean = state.get_expanding_mean(hour_hist)

    features["hour_deviation"] = abs(hour - hour_mean)
    features["hour_diff_from_preferred"] = abs(hour - state.preferred_hour)
    features["hour_consistency"] = 1 - (abs(hour - hour_mean) / 12)
    features["peak_hour"] = int(9 <= hour <= 17)
    features["off_hour"] = int(hour < 6 or hour > 22)

    # LOCATION
    home = state.home_location or "Dhaka"
    features["location_change_flag"] = int(location != home)
    features["new_location_flag"] = int(location not in state.unique_locations)
    features["location_diversity"] = len(state.unique_locations) / (state.transaction_count + 1)
    features["distance_from_home_km"] = calc_distance(home, location)

    if state.last_location is not None and minutes_since_last is not None and minutes_since_last >= 0:
        dist = calc_distance(state.last_location, location)
        features["impossible_travel"] = int(minutes_since_last < 30 and dist > 300)
    else:
        features["impossible_travel"] = 0

    # DEVICE
    device_hist = list(state.device_history)
    features["device_change_flag"] = int(device_id != state.primary_device)
    features["new_device_flag"] = int(device_id not in state.unique_devices)
    features["device_diversity"] = len(state.unique_devices) / (state.transaction_count + 1)
    features["device_reuse_ratio"] = state.all_device_counts.get(device_id, 0) / (state.transaction_count + 1)

    if device_id in state.device_first_seen:
        features["days_since_device_first_seen"] = (timestamp - state.device_first_seen[device_id]).days
    else:
        features["days_since_device_first_seen"] = 0

    consecutive_device = 0
    for d in reversed(device_hist):
        if d == device_id:
            consecutive_device += 1
        else:
            break
    features["consecutive_same_device"] = consecutive_device

    # MERCHANT
    merchant_hist = list(state.merchant_history)
    features["merchant_change_flag"] = int(merchant_id not in state.unique_merchants)
    features["merchant_diversity"] = len(state.unique_merchants) / (state.transaction_count + 1)
    features["unique_merchant_ratio"] = features["merchant_diversity"] / (state.transaction_count + 1)
    features["merchant_switch"] = int(merchant_hist[-1] != merchant_id) if merchant_hist else 0

    switch_hist = list(state.historical_merchant_switch)
    features["merchant_switch_ratio"] = float(np.mean(switch_hist[-5:])) if switch_hist else 0.0

    consecutive_merchant = 0
    for m in reversed(merchant_hist):
        if m == merchant_id:
            consecutive_merchant += 1
        else:
            break
    features["consecutive_same_merchant"] = consecutive_merchant
    features["merchant_network_density"] = len(state.unique_merchants) / (state.transaction_count + 1)

    historical_merchant_avg = state.merchant_avg_amounts.get(merchant_id, 0.0)
    features["historical_merchant_avg_amount"] = historical_merchant_avg
    features["amount_vs_merchant_history"] = amount / (historical_merchant_avg + 1)
    features["merchant_switch_amount"] = features["merchant_switch"] * amount

    switch_amount_hist = list(state.historical_merchant_switch_amount)
    features["merchant_switch_amount_ma3"] = float(np.mean(switch_amount_hist[-3:])) if switch_amount_hist else 0.0

    # RISK
    features["high_risk_exposure"] = merchant_risk * amount / (amount + 1000)

    risk_hist = list(state.all_time_merchant_risks)
    high_risk_07 = sum(1 for r in risk_hist if r > 0.7)
    high_risk_08 = sum(1 for r in risk_hist if r > 0.8)
    features["high_risk_visit_ratio"] = high_risk_07 / (state.transaction_count + 1) if state.transaction_count > 0 else 0.0
    features["high_risk_merchant_ratio"] = high_risk_08 / (state.transaction_count + 1) if state.transaction_count > 0 else 0.0

    # VELOCITY
    features["time_since_last_txn_minutes"] = float(minutes_since_last) if minutes_since_last is not None else -1.0

    if state.last_timestamp is not None:
        txn_interval = (timestamp - state.last_timestamp).total_seconds() / 60.0
        features["txn_interval"] = txn_interval
        features["rapid_txn"] = int(txn_interval < 2)
    else:
        features["txn_interval"] = 0.0
        features["rapid_txn"] = 0

    intervals = previous_txn_intervals(state.timestamps, 3)
    features["txn_interval_ma3"] = float(np.mean(intervals)) if intervals else 0.0
    features["txn_count_10min"] = state.get_transaction_count_10min(timestamp)
    features["velocity_5min"] = state.get_velocity_5min(timestamp)
    features["velocity_1hr"] = state.get_velocity_1hr(timestamp)

    velocity_accel_hist = list(state.historical_velocity_accel)
    features["velocity_accel"] = velocity_accel_hist[-1] if velocity_accel_hist else 0.0
    features["velocity_accel"] = np.clip(features["velocity_accel"], -5, 5)
    features["velocity_accel_ma3"] = float(np.mean(velocity_accel_hist[-3:])) if velocity_accel_hist else 0.0

    features["txn_burst"] = int(features["velocity_5min"] > 3)
    burst_hist = list(state.historical_txn_burst)
    features["burst_frequency"] = float(sum(burst_hist[-10:])) if burst_hist else 0.0

    features["peak_hour_velocity"] = features["peak_hour"] * features["velocity_5min"]
    features["off_hour_velocity"] = features["off_hour"] * features["velocity_5min"]
    features["off_peak_velocity_ratio"] = features["off_hour_velocity"] / (features["peak_hour_velocity"] + 1)

    # MUST COME AFTER velocity_5min
    features["merchant_switch_velocity"] = features["merchant_switch"] * features["velocity_5min"]

    # COMBINATION
    features["device_location_combo"] = features["new_device_flag"] + features["new_location_flag"]
    features["amount_x_velocity"] = amount * features["velocity_5min"] / 10000

    # TODAY
    features["unique_merchants_today"] = len(state.today_merchants)
    features["unique_locations_today"] = len(state.today_locations)

    features["transaction_count"] = state.transaction_count

    home_count = state.all_location_counts.get(home, 0)
    features["home_ratio"] = home_count / (state.transaction_count + 1)

    features["is_compromised"] = 0
    features["compromise_score"] = 0.0

    # RATIOS
    features["amount_to_avg_ratio"] = amount / (hist_avg + 1)
    features["amount_to_base_ratio"] = amount / (state.base_amount + 1)

    # STATS
    hist_std = np.std(state.all_time_amounts, ddof=1) if len(state.all_time_amounts) > 1 else 1.0
    features["amount_zscore"] = (amount - hist_avg) / (hist_std + 1)

    all_amt = list(state.all_time_amounts)
    features["amount_percentile"] = sum(h <= amount for h in all_amt) / len(all_amt) if all_amt else 0.5

    rolling_median = float(np.median(list(state.amount_history))) if state.amount_history else 0.0
    features["amount_to_rolling_median_ratio"] = amount / (rolling_median + 1)

    # ANOMALY
    anomaly_signals = []
    if features["new_device_flag"]:
        anomaly_signals.append("new_device")
    if features["new_location_flag"]:
        anomaly_signals.append("new_location")
    if features["merchant_change_flag"]:
        anomaly_signals.append("new_merchant")
    if features["amount_to_avg_ratio"] > 3:
        anomaly_signals.append("high_amount")
    if features["impossible_travel"]:
        anomaly_signals.append("impossible_travel")
    if features["hour_diff_from_preferred"] > 6:
        anomaly_signals.append("unusual_hour")

    features["anomaly_score"] = len(anomaly_signals) / 6.0
    features["anomaly_signals"] = str(anomaly_signals)

    # FREQUENCY
    features["user_id_freq"] = frequency_maps.get("user_id_freq", {}).get(state.user_id, 0.0)
    features["merchant_id_freq"] = frequency_maps.get("merchant_id_freq", {}).get(merchant_id, 0.0)
    features["device_id_freq"] = frequency_maps.get("device_id_freq", {}).get(device_id, 0.0)

    return features


# ============================================================
# ENCODING & CLIPPING
# ============================================================

def apply_encoders(features):
    for col in ["location", "merchant_category", "transaction_type", "user_segment", "user_risk"]:
        if col not in features or col not in encoders:
            continue
        encoder = encoders[col]
        try:
            if hasattr(encoder, "transform"):
                features[col] = int(encoder.transform([[features[col]]])[0][0])
            elif isinstance(encoder, dict):
                features[col] = encoder.get(features[col], -1)
        except Exception as e:
            raise ValueError(f"Failed to encode {col}: {features[col]} - {e}")
    return features


def apply_clip_bounds(features):
    for col, bounds in clip_bounds.items():
        if col not in features:
            continue
        try:
            lower, upper = bounds
            features[col] = float(np.clip(safe_float(features[col]), lower, upper))
        except Exception:
            continue
    return features


# ============================================================
# USER STATE CACHE
# ============================================================

user_states = {}


def reset_user_state(user_id=None):
    global user_states
    if user_id is None:
        user_states.clear()
    else:
        user_states.pop(str(user_id), None)


# ============================================================
# PREDICTION
# ============================================================

def predict_with_state(
    user_id,
    amount,
    location,
    merchant_id,
    hour,
    device_id,
    minutes_since_last=None,
    timestamp=None
):
    """
    Predict fraud probability.

    FIXED:
        - If timestamp is provided, state is built with
          before_timestamp = timestamp, so future transactions
          are excluded from the state.
        - If a cached state exists but its latest timestamp is
          >= the new timestamp, the state is rebuilt to prevent
          future-data leakage.
    """

    user_id = safe_string(user_id, "U00001")
    location = safe_string(location, "Dhaka")
    merchant_id = safe_string(merchant_id, "M0001")
    device_id = safe_string(device_id, "D0001")
    amount = safe_float(amount)
    hour = safe_int(hour, 12)

    # Normalize timestamp
    timestamp_ts = None
    if timestamp is not None:
        timestamp_ts = pd.to_datetime(timestamp, errors="coerce")
        if pd.isna(timestamp_ts):
            timestamp_ts = None

    # ---- State resolution with strict chronological isolation ----
    need_rebuild = False

    if user_id not in user_states:
        need_rebuild = True
    elif timestamp_ts is not None:
        cached = user_states[user_id]
        cached_latest = cached.latest_built_timestamp
        if cached_latest is not None and timestamp_ts <= cached_latest:
            need_rebuild = True

    if need_rebuild:
        if timestamp_ts is not None:
            print(f"Building state for {user_id} (before {timestamp_ts})...")
            user_states[user_id] = build_user_state(user_id, before_timestamp=timestamp_ts)
        else:
            print(f"Building state for {user_id} (no timestamp filter)...")
            user_states[user_id] = build_user_state(user_id, before_timestamp=None)

    state = user_states[user_id]

    # ---- Timestamp / gap resolution ----
    merchant_category, merchant_risk = lookup_merchant(merchant_id)

    if timestamp_ts is not None:
        txn_timestamp = timestamp_ts
        if state.last_timestamp is not None:
            gap = (txn_timestamp - state.last_timestamp).total_seconds() / 60.0
        else:
            gap = -1.0
    elif state.last_timestamp is not None and minutes_since_last is not None and minutes_since_last >= 0:
        gap = float(minutes_since_last)
        txn_timestamp = state.last_timestamp + timedelta(minutes=gap)
    else:
        gap = -1.0
        now = datetime.now()
        txn_timestamp = now.replace(hour=hour, minute=0, second=0, microsecond=0)

    # ---- Features ----
    features = compute_features(
        state=state,
        amount=amount,
        location=location,
        merchant_id=merchant_id,
        merchant_risk=merchant_risk,
        merchant_category=merchant_category,
        hour=hour,
        device_id=device_id,
        timestamp=txn_timestamp,
        minutes_since_last=gap
    )

    features = apply_encoders(features)
    features = apply_clip_bounds(features)

    df = pd.DataFrame([features])

    # ---- Verify columns ----
    if feature_cols:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing {len(missing)} features: {missing}")
        df = df[feature_cols]
        if df.isna().any().any():
            bad = df.columns[df.isna().any()].tolist()
            raise ValueError(f"NaN in features: {bad}")
        df = df.fillna(0)

    # ---- Predict ----
    try:
        prob = float(model.predict_proba(df)[0][1])
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    prediction = int(prob >= THRESHOLD)

    if prob >= 0.85:
        risk_level = "CRITICAL"
    elif prob >= THRESHOLD:
        risk_level = "HIGH"
    elif prob >= 0.30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # ---- Factors ----
    factors = []
    if features.get("amount_to_avg_ratio", 0) > 3:
        factors.append(f"Amount: {features['amount_to_avg_ratio']:.2f}x avg")
    if features.get("new_device_flag", 0) == 1:
        factors.append("New device")
    if features.get("new_location_flag", 0) == 1:
        factors.append("New location")
    if features.get("merchant_change_flag", 0) == 1:
        factors.append("New merchant")
    if features.get("hour_diff_from_preferred", 0) > 6:
        factors.append(f"Unusual hour: {hour}:00")
    if features.get("impossible_travel", 0) == 1:
        factors.append("Impossible travel")
    if features.get("merchant_risk", 0) > 0.7:
        factors.append(f"High risk merchant: {features['merchant_risk']:.2f}")
    if features.get("exceeds_limit_flag", 0) == 1:
        factors.append("Exceeds limit")
    if features.get("velocity_5min", 0) > 3:
        factors.append(f"High velocity: {features['velocity_5min']} in 5min")
    if not factors:
        factors.append("No risk factors detected")

    # ---- Update state ----
    state.add_transaction(
        amount=amount,
        location=location,
        device_id=device_id,
        merchant_id=merchant_id,
        merchant_risk=merchant_risk,
        hour=hour,
        timestamp=txn_timestamp
    )

    user_states[user_id] = state

    return (prediction, prob, risk_level, factors, features)