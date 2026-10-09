import numpy as np
import pandas as pd
import random
import networkx as nx
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')





CONFIG = {
    'n_transactions': 100000,    
    'max_total_transactions': 130000,  
    'n_users': 5000,
    'n_merchants': 500,
    'n_devices': 50000,
    'base_fraud_rate': 0.14,  
    'n_fraud_rings': 30,
    'days_to_simulate': 365,
    'start_date': '2024-01-01',
    'seed': 42,
    'version': '4.0',
    'country': 'BD',
}

BKASH_LIMITS = {
    'daily_send_money': 50_000,
    'monthly_send_money': 300_000,
    'daily_cash_out': 30_000,
    'monthly_cash_out': 200_000,
    'daily_add_money': 50_000,
    'daily_bank_transfer': 50_000,
    'monthly_bank_transfer': 300_000,
    'max_balance': 500_000,
    'per_transaction_max': 50_000,
}

DIVISIONS = ["Dhaka", "Chattogram", "Rajshahi", "Khulna",
             "Barishal", "Sylhet", "Rangpur", "Mymensingh"]

DIVISION_COORDS = {
    "Dhaka": (23.8103, 90.4125),
    "Chattogram": (22.3569, 91.7832),
    "Rajshahi": (24.3745, 88.6042),
    "Khulna": (22.8456, 89.5403),
    "Barishal": (22.7010, 90.3535),
    "Sylhet": (24.8949, 91.8687),
    "Rangpur": (25.7439, 89.2752),
    "Mymensingh": (24.7471, 90.4073)
}


def calc_distance(loc1, loc2):
    """Calculate distance between two divisions in km."""
    if loc1 not in DIVISION_COORDS or loc2 not in DIVISION_COORDS:
        return 0
    lat1, lon1 = DIVISION_COORDS[loc1]
    lat2, lon2 = DIVISION_COORDS[loc2]
    return np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) * 111


USER_SEGMENTS = {
    "student": {
        "weight": 0.20,
        "txn_range": (20, 5000),
        "active_hours": (8, 22),
        "categories": {"Food": 0.40, "Mobile": 0.25, "Shopping": 0.20, "Transport": 0.15},
        "activity_level": 0.4,
        "lifecycle": {"min_age": 30, "max_age": 730}
    },
    "office_worker": {
        "weight": 0.25,
        "txn_range": (100, 20000),
        "active_hours": (9, 20),
        "categories": {"Food": 0.30, "Transport": 0.20, "Shopping": 0.20, "Utilities": 0.15, "Retail": 0.15},
        "activity_level": 0.7,
        "lifecycle": {"min_age": 60, "max_age": 1095}
    },
    "freelancer": {
        "weight": 0.10,
        "txn_range": (100, 30000),
        "active_hours": (10, 23),
        "categories": {"Food": 0.20, "Tech": 0.20, "Retail": 0.20, "Transport": 0.15, "Entertainment": 0.15, "Utilities": 0.10},
        "activity_level": 0.6,
        "lifecycle": {"min_age": 60, "max_age": 1095}
    },
    "small_business": {
        "weight": 0.15,
        "txn_range": (500, 50000),
        "active_hours": (7, 22),
        "categories": {"Wholesale": 0.25, "Fuel": 0.20, "Transport": 0.20, "Food": 0.15, "Banking": 0.10, "Utilities": 0.10},
        "activity_level": 0.8,
        "lifecycle": {"min_age": 90, "max_age": 1460}
    },
    "merchant": {
        "weight": 0.15,
        "txn_range": (500, 50000),
        "active_hours": (8, 22),
        "categories": {"Retail": 0.30, "Wholesale": 0.25, "Banking": 0.15, "Transport": 0.10, "Food": 0.10, "Utilities": 0.10},
        "activity_level": 0.75,
        "lifecycle": {"min_age": 90, "max_age": 1460}
    },
    "garment_worker": {
        "weight": 0.10,
        "txn_range": (50, 5000),
        "active_hours": (6, 21),
        "categories": {"Food": 0.40, "Transport": 0.20, "Mobile": 0.20, "Utilities": 0.10, "Shopping": 0.10},
        "activity_level": 0.6,
        "lifecycle": {"min_age": 60, "max_age": 1095}
    },
    "farmer": {
        "weight": 0.05,
        "txn_range": (50, 3000),
        "active_hours": (5, 19),
        "categories": {"Food": 0.40, "Utilities": 0.20, "Agriculture": 0.20, "Transport": 0.10, "Mobile": 0.10},
        "activity_level": 0.3,
        "lifecycle": {"min_age": 90, "max_age": 1460}
    }
}

MERCHANT_CATEGORIES = [
    "Food_Restaurant", "Grocery_Supermarket", "Mobile_Recharge",
    "Transport_Rickshaw", "Transport_Bus", "Transport_Train",
    "Fuel_Petrol_Pump", "Utilities_Electricity", "Utilities_Gas",
    "Utilities_Water", "Education_Tuition", "Education_School",
    "Education_College", "Healthcare_Clinic", "Healthcare_Pharmacy",
    "Healthcare_Hospital", "Retail_Clothing", "Retail_Electronics",
    "Retail_Furniture", "Wholesale_Market", "Banking_Agent",
    "ATM_Cash_Out", "Government_Tax", "Government_Fee",
    "Insurance_Premium", "Investment_Savings", "E_commerce_Online",
    "Entertainment_Cinema", "Entertainment_Cafe",
    "Travel_Bus_Ticket", "Travel_Train_Ticket", "Travel_Launch",
    "Real_Estate_Rent", "Real_Estate_Broker",
    "Agriculture_Fertilizer", "Agriculture_Seeds", "Agriculture_Equipment"
]

FRAUD_TYPES = {
    
    "amount_spike": {"weight": 0.15, "severity": 0.7, "signals": ["high_amount"]},
    "device_takeover": {"weight": 0.14, "severity": 0.8, "signals": ["new_device", "new_location"]},
    "merchant_collusion": {"weight": 0.13, "severity": 0.7, "signals": ["new_merchant", "high_amount"]},
    "velocity_fraud": {"weight": 0.12, "severity": 0.6, "signals": ["velocity"]},
    "subtle": {"weight": 0.14, "severity": 0.4, "signals": ["slight_amount_increase"]},
    "location_spoofing": {"weight": 0.10, "severity": 0.6, "signals": ["new_location"]},
    "account_takeover": {"weight": 0.10, "severity": 0.9, "signals": ["new_device", "new_location", "high_amount"]},
    "synthetic_identity": {"weight": 0.06, "severity": 0.8, "signals": ["new_user"]},
    "money_mule": {"weight": 0.06, "severity": 0.7, "signals": ["high_amount", "new_merchant"]},
}


# MODULE 2: USER GENERATOR



class UserGenerator:
    def __init__(self, config):
        self.config = config

    def generate(self):
        users = []
        for i in range(self.config['n_users']):
            users.append(self._create_user(i))
        return pd.DataFrame(users)

    def _create_user(self, idx):
        segments = list(USER_SEGMENTS.keys())
        weights = [USER_SEGMENTS[s]['weight'] for s in segments]
        segment = random.choices(segments, weights=weights)[0]
        seg_info = USER_SEGMENTS[segment]

        txn_min, txn_max = seg_info['txn_range']
        log_mean = (np.log(txn_min) + np.log(txn_max)) / 2
        log_std = (np.log(txn_max) - np.log(txn_min)) / 4
        base_amount = int(np.exp(np.random.normal(log_mean, log_std)))
        base_amount = max(txn_min, min(txn_max, base_amount))

        min_age, max_age = seg_info['lifecycle']['min_age'], seg_info['lifecycle']['max_age']
        account_age = np.random.randint(min_age, max_age)
        reg_date = datetime.strptime(self.config['start_date'], '%Y-%m-%d') - timedelta(days=int(account_age))

        activity = seg_info['activity_level'] * random.uniform(0.7, 1.3)
        active_start, active_end = seg_info['active_hours']
        preferred_hour = np.random.randint(active_start, active_end)
        home = random.choice(DIVISIONS)

        n_devices = random.randint(1, 3)
        devices = [f"D{random.randint(1, self.config['n_devices'])}" for _ in range(n_devices)]
        device_ages = {d: np.random.randint(1, 365) for d in devices}

        risk_weights = {
            "student": [0.5, 0.3, 0.2],
            "office_worker": [0.6, 0.3, 0.1],
            "freelancer": [0.4, 0.4, 0.2],
            "small_business": [0.4, 0.4, 0.2],
            "merchant": [0.4, 0.4, 0.2],
            "garment_worker": [0.5, 0.3, 0.2],
            "farmer": [0.6, 0.3, 0.1]
        }
        risk = random.choices(["low", "medium", "high"],
                               weights=risk_weights.get(segment, [0.5, 0.35, 0.15]))[0]

        return {
            "user_id": f"U{idx:05d}",
            "segment": segment,
            "base_amount": base_amount,
            "txn_range": (txn_min, txn_max),
            "home_location": home,
            "devices": devices,
            "primary_device": devices[0],
            "device_ages": device_ages,
            "preferred_hour": int(preferred_hour),
            "active_hours": (active_start, active_end),
            "category_preferences": seg_info['categories'],
            "activity_level": activity,
            "risk_profile": risk,
            "salary_day": random.randint(20, 28),
            "registration_date": reg_date,
            "account_age_days": int(account_age),
            "status": "active"
        }


# MODULE 3: MERCHANT GENERATOR



class MerchantGenerator:
    def __init__(self, config):
        self.config = config

    def generate(self):
        merchants = []
        for i in range(self.config['n_merchants']):
            merchants.append(self._create_merchant(i))
        return pd.DataFrame(merchants)

    def _create_merchant(self, idx):
        category = random.choice(MERCHANT_CATEGORIES)
        location = random.choice(DIVISIONS)
        op_start = random.randint(6, 10)
        op_end = random.randint(18, 24)
        size = random.choice(["small", "medium", "large", "enterprise"])
        size_multipliers = {"small": 1, "medium": 3, "large": 8, "enterprise": 20}
        avg_ticket = int(np.random.lognormal(mean=7, sigma=1.5) * size_multipliers[size])
        avg_ticket = max(50, min(500000, avg_ticket))
        days_open = np.random.randint(30, 730)
        is_high_risk = 1 if random.random() < 0.12 else 0
        merchant_risk = np.random.uniform(0, 1) + (0.3 if is_high_risk else 0)
        merchant_risk = min(merchant_risk, 1.0)
        historical_fraud_rate = random.uniform(0, 0.1) if is_high_risk else random.uniform(0, 0.02)

        return {
            "merchant_id": f"M{idx + 1:04d}",
            "category": category,
            "location": location,
            "avg_ticket": avg_ticket,
            "operating_hours": (op_start, op_end),
            "size": size,
            "merchant_risk": merchant_risk,
            "is_high_risk": is_high_risk,
            "historical_fraud_rate": historical_fraud_rate,
            "days_open": int(days_open),
            "status": "active"
        }


# MODULE 4: FRAUD RING GENERATOR



class FraudRingGenerator:
    def __init__(self, config, users_df, merchants_df):
        self.config = config
        self.users = users_df
        self.merchants = merchants_df

    def generate(self):
        rings = {}
        all_users = self.users["user_id"].tolist()
        all_merchants = self.merchants["merchant_id"].tolist()
        high_value = self.users[self.users["segment"].isin(["small_business", "merchant"])]["user_id"].tolist()

        for i in range(self.config['n_fraud_rings']):
            rings[f"R{i:03d}"] = self._create_ring(i, all_users, all_merchants, high_value)
        return rings

    def _create_ring(self, idx, all_users, all_merchants, high_value):
        leader = random.choice(high_value) if high_value else random.choice(all_users)
        ring_size = random.randint(3, 10)
        member_pool = high_value + random.sample(all_users, min(50, len(all_users)))
        members = random.sample(member_pool, min(ring_size, len(member_pool)))
        if leader not in members:
            members.append(leader)
        ring_merchants = random.sample(all_merchants, min(random.randint(2, 6), len(all_merchants)))
        ring_devices = [f"D{random.randint(1, self.config['n_devices'])}" for _ in range(random.randint(3, 6))]

        return {
            "ring_id": f"R{idx:03d}",
            "leader": leader,
            "members": members,
            "merchants": ring_merchants,
            "devices": ring_devices,
            "strategy": random.choice(["aggressive", "stealthy", "adaptive", "mixed"]),
            "active": True,
            "attack_count": 0,
            "max_attacks": random.randint(50, 300),
            "intensity": random.uniform(0.5, 1.0),
            "evolution_stage": 0,
            "attack_type": random.choice(["amount_spike", "device_takeover", "merchant_collusion", "mixed"])
        }


# MODULE 5: USER STATE ENGINE



class UserState:
    def __init__(self, user):
        self.user_id = user["user_id"]
        self.base = user["base_amount"]
        self.txn_range = user["txn_range"]
        self.home = user["home_location"]
        self.primary_device = user["primary_device"]
        self.devices = set(user["devices"])
        self.device_ages = user["device_ages"]
        self.preferred_hour = user["preferred_hour"]
        self.category_prefs = user["category_preferences"]
        self.activity_level = user["activity_level"]
        self.salary_day = user["salary_day"]
        self.registration_date = user["registration_date"]
        self.account_age = user["account_age_days"]

        self.amount_history = deque(maxlen=20)
        self.location_history = deque(maxlen=20)
        self.device_history = deque(maxlen=20)
        self.hour_history = deque(maxlen=20)
        self.merchant_history = deque(maxlen=20)
        self.category_history = deque(maxlen=20)
        self.timestamps = deque(maxlen=20)

        self.avg_amount = self.base
        self.transaction_count = 0
        self.merchant_freq = defaultdict(int)
        self.location_freq = defaultdict(int)
        self.device_freq = defaultdict(int)
        self.category_freq = defaultdict(int)
        self.consecutive_same_merchant = 0
        self.consecutive_same_device = 0
        self.last_location = self.home
        self.last_device = self.primary_device
        self.last_timestamp = None
        self.unique_locations = {self.home}
        self.unique_devices = {self.primary_device}
        self.unique_merchants = set()
        self.unique_categories = set()
        
        self.device_first_seen = {self.primary_device: self.registration_date}
        self.is_compromised = False
        self.compromise_score = 0
        self.suspicious_count = 0
        self.velocity_5min = deque(maxlen=20)
        self.velocity_1hr = deque(maxlen=100)
        self.today_merchants = set()
        self.today_locations = set()
        self.today_date = None

    def add_transaction(self, txn):
        if txn.get('fraud_label', 0) == 0:
            self.amount_history.append(txn['amount'])
        self.location_history.append(txn['location'])
        self.device_history.append(txn['device_id'])
        self.hour_history.append(txn['hour'])
        self.merchant_history.append(txn['merchant_id'])
        self.category_history.append(txn.get('merchant_category', 'Unknown'))
        self.timestamps.append(txn['timestamp'])

        self.merchant_freq[txn['merchant_id']] += 1
        self.location_freq[txn['location']] += 1
        self.device_freq[txn['device_id']] += 1
        self.category_freq[txn.get('merchant_category', 'Unknown')] += 1

        self.unique_locations.add(txn['location'])
        self.unique_devices.add(txn['device_id'])
        self.unique_merchants.add(txn['merchant_id'])
        self.unique_categories.add(txn.get('merchant_category', 'Unknown'))

        if len(self.amount_history) >= 10:
            self.avg_amount = np.mean(self.amount_history)
        else:
            self.avg_amount = self.base
        self.last_location = txn['location']
        self.last_device = txn['device_id']
        self.last_timestamp = txn['timestamp']
        self.transaction_count += 1

        if len(self.merchant_history) > 1:
            if self.merchant_history[-1] == self.merchant_history[-2]:
                self.consecutive_same_merchant += 1
            else:
                self.consecutive_same_merchant = 0

        if len(self.device_history) > 1:
            if self.device_history[-1] == self.device_history[-2]:
                self.consecutive_same_device += 1
            else:
                self.consecutive_same_device = 0

        if txn['device_id'] not in self.device_first_seen:
            self.device_first_seen[txn['device_id']] = txn['timestamp']

        self.velocity_5min.append(txn['timestamp'])
        self.velocity_1hr.append(txn['timestamp'])

        current_date = txn['timestamp'].date()
        if self.today_date != current_date:
            self.today_merchants = set()
            self.today_locations = set()
            self.today_date = current_date
        self.today_merchants.add(txn['merchant_id'])
        self.today_locations.add(txn['location'])

      
        score = 0

        if txn["new_device_flag"]:
            score += 1

        if txn["new_location_flag"]:
            score += 1

        if txn["amount_to_avg_ratio"] > 3:
            score += 1

        if txn["impossible_travel"]:
            score += 2

        if txn["anomaly_score"] > 0.7:
            score += 2

        if score >= 3:
            self.suspicious_count += 1
            if self.suspicious_count > 3:
                self.is_compromised = True
                self.compromise_score = min(1.0, self.compromise_score + 0.05)

    def get_behavioral_features(self, current_device=None, current_time=None):
        current_device = current_device or self.last_device
        now = current_time or self.last_timestamp or datetime.now()
        v5 = sum(
            1 for t in self.velocity_5min
            if 0 <= (now - t).total_seconds() <= 300
        )

        v1 = sum(
            1 for t in self.velocity_1hr
            if 0 <= (now - t).total_seconds() <= 3600
        )

        first_seen = self.device_first_seen.get(current_device)
        if first_seen and now >= first_seen:
            days_since_device_first_seen = (now - first_seen).days
        else:
            days_since_device_first_seen = 0

        return {
            'avg_amount': self.avg_amount,
            'transaction_count': self.transaction_count,
            'location_diversity': len(self.unique_locations),
            'device_diversity': len(self.unique_devices),
            'merchant_diversity': len(self.unique_merchants),
            'category_diversity': len(self.unique_categories),
            'primary_device_ratio': self.device_freq.get(self.primary_device, 0) / (self.transaction_count + 1),
            # Replaces raw device_id-frequency-style features with
            # genuinely behavioral ones: how often THIS device is reused,
            # how long it's been seen, and streak of consecutive use.
            'device_reuse_ratio': self.device_freq.get(current_device, 0) / (self.transaction_count + 1),
            'days_since_device_first_seen': days_since_device_first_seen,
            'consecutive_same_device': self.consecutive_same_device,
            'home_ratio': self.location_freq.get(self.home, 0) / (self.transaction_count + 1),
            'consecutive_same_merchant': self.consecutive_same_merchant,
            'is_compromised': int(self.is_compromised),
            'compromise_score': self.compromise_score,
            'velocity_5min': v5,
            'velocity_1hr': v1,
            'unique_merchants_today': len(self.today_merchants),
            'unique_locations_today': len(self.today_locations),
            'account_age_days': self.account_age,
        }

    def detect_anomalies(self, txn):
        signals = []

        ratio = txn['amount'] / (self.avg_amount + 1)
        comp_amount = min(ratio / 10, 1.0)
        if ratio > 3:
            signals.append('high_amount')

        v5 = sum(1 for t in self.velocity_5min if (txn['timestamp'] - t).total_seconds() <= 300)
        comp_velocity = min(v5 / 5, 1.0)
        if v5 >= 3:
            signals.append('velocity')

        
        comp_location = 1.0 if txn['location'] not in self.location_history else 0.0
        if txn['location'] not in self.location_history:
            signals.append('new_location')

      
        comp_device = 1.0 if txn['device_id'] not in self.device_history else 0.0
        if txn['device_id'] not in self.device_history:
            signals.append('new_device')

        # Merchant anomaly
        comp_merchant = 1.0 if txn['merchant_id'] not in self.merchant_history else 0.0
        if txn['merchant_id'] not in self.merchant_history:
            signals.append('new_merchant')

        score = (
            0.30 * comp_amount +
            0.25 * comp_velocity +
            0.20 * comp_location +
            0.15 * comp_device +
            0.10 * comp_merchant
        )

        return {'signals': signals, 'score': min(score, 1.0)}






class SimulationClock:
    def __init__(self, start_date, days_to_simulate):
        self.current = datetime.strptime(start_date, '%Y-%m-%d')
        self.end = self.current + timedelta(days=days_to_simulate)
        self.days_simulated = 0

    def advance(self, minutes=5, max_minutes=120):
        step = random.randint(minutes, max_minutes)
        self.current += timedelta(minutes=step)
        self.days_simulated += step / (24 * 60)
        return self.current

    def advance_to(self, new_time):
        """Jump the clock forward to an explicit timestamp (used after bursts)."""
        if new_time > self.current:
            self.current = new_time

    def is_weekend(self):
        return self.current.weekday() >= 5

    def is_holiday(self):
        if self.current.month == 3 and 5 <= self.current.day <= 15:
            return True
        if self.current.month == 4 and self.current.day == 14:
            return True
        if self.current.month == 12 and self.current.day == 16:
            return True
        if self.current.month == 2 and self.current.day == 21:
            return True
        return False

    def get_seasonal_multiplier(self):
        multiplier = 1.0
        if self.is_weekend():
            multiplier *= 1.2
        if self.current.day >= 25:
            multiplier *= 1.3
        if self.is_holiday():
            multiplier *= 1.5
        if 22 <= self.current.hour <= 5:
            multiplier *= 0.3
        return multiplier



class FraudEngine:
    def __init__(self, rings, users, merchants):
        self.rings = rings
        self.users = users
        self.merchants = merchants
        self.attack_counts = defaultdict(int)

    def process(self, user, txn, state):
        risk = 0

        if txn['hour'] >= 22 or txn['hour'] <= 5:
            risk += 1
        if txn['merchant_risk'] > 0.7:
            risk += 1
        if state and len(state.device_history) > 0 and txn['device_id'] not in state.device_history:
            risk += 1
        if state and len(state.location_history) > 0 and txn['location'] not in state.location_history:
            risk += 1
        if state:
            v5 = sum(1 for t in state.velocity_5min if (txn['timestamp'] - t).total_seconds() <= 300)
            if v5 >= 3:
                risk += 1
        if user.get('account_age_days', 365) < 15:
            risk += 1

        ring_mods = {}
        ring_id = self._check_ring_attack(user['user_id'])
        if ring_id:
            risk += 2
            ring_mods, _active = self._get_ring_attack(ring_id, user, txn)

     
        base = 0.055
        prob = base * (1.06 ** risk)
        prob = min(prob, 0.30)

        if random.random() < prob:
            fraud_type = self._select_fraud_type()
            mods, burst_plan = self._apply_fraud_type(fraud_type, user, txn, state)
            if ring_mods:
                mods.update(ring_mods)
            fraud_score = random.uniform(0.6, 0.9)
            return True, fraud_score, fraud_type, mods, burst_plan

        return False, 0, 'normal', {}, []

    def _check_ring_attack(self, user_id):
        for ring_id, ring in self.rings.items():
            if user_id in ring['members'] and ring['active']:
                if ring['attack_count'] < ring['max_attacks']:
                    if random.random() < ring['intensity'] * 0.5:
                        return ring_id
        return None

    def _get_ring_attack(self, ring_id, user, txn):
        ring = self.rings[ring_id]
        self.attack_counts[ring_id] += 1
        ring['attack_count'] += 1

        ring['evolution_stage'] += 1
        if ring['evolution_stage'] % 50 == 0:
            ring['intensity'] = random.uniform(0.5, 1.0)

        if ring['attack_count'] >= ring['max_attacks']:
            ring['active'] = False

        mods = {}
        strategy = ring.get('strategy', 'mixed')

        if strategy == 'aggressive':
            mods['amount'] = round(user['base_amount'] * random.uniform(3, 8), 2)
            if random.random() < 0.6:
                mods['device_id'] = random.choice(ring['devices'])
            if random.random() < 0.5:
                mods['location'] = random.choice([d for d in DIVISIONS if d != user['home_location']])
        elif strategy == 'stealthy':
            mods['amount'] = round(user['base_amount'] * random.uniform(1.5, 2.5), 2)
            if random.random() < 0.3:
                mods['device_id'] = random.choice(ring['devices'])
            if random.random() < 0.2:
                mods['location'] = random.choice([d for d in DIVISIONS if d != user['home_location']])
        else:
            mods['amount'] = round(user['base_amount'] * random.uniform(2, 6), 2)
            if random.random() < 0.4:
                mods['device_id'] = random.choice(ring['devices'])
            if random.random() < 0.3:
                mods['location'] = random.choice([d for d in DIVISIONS if d != user['home_location']])

        if random.random() < 0.3:
            mods['merchant_id'] = random.choice(ring['merchants'])
        if random.random() < 0.3 and 'device_id' in mods:
            del mods['device_id']
        if random.random() < 0.2 and 'location' in mods:
            del mods['location']

        return mods, ring['active']

    def _select_fraud_type(self):
        types = list(FRAUD_TYPES.keys())
        weights = [FRAUD_TYPES[t]['weight'] for t in types]
        return random.choices(types, weights=weights)[0]

    def _amount_reference(self, user, state):
        """
        Reference amount fraud multipliers are built from. Prefer the
        user's actual recent rolling average (what the model will see as
        avg_amount) over the static per-user base_amount, once enough
        history exists - this keeps spikes calibrated to real behavior
        instead of a fixed value assigned once at user creation.
        """
        if state is not None and state.transaction_count >= 3:
            return state.avg_amount
        return user['base_amount']

    def _pick_spoofed_location(self, user):
        """
        Mixed-distance location spoofing instead of always maxing out
        distance + impossible_travel. Roughly:
          30% nearby division   (< 150km)
          30% mid-range division (150-300km)
          20% far / plausible impossible-travel candidate (> 300km)
          20% GPS-drift / VPN-like: no real location change at all
        impossible_travel is NOT force-set here; it's left to emerge
        naturally from the actual time gap vs. distance downstream.
        """
        home = user['home_location']
        near = [d for d in DIVISIONS if d != home and calc_distance(home, d) < 150]
        mid = [d for d in DIVISIONS if 150 <= calc_distance(home, d) < 300]
        far = [d for d in DIVISIONS if calc_distance(home, d) >= 300]

        roll = random.random()
        if roll < 0.30 and near:
            return random.choice(near)
        elif roll < 0.60 and mid:
            return random.choice(mid)
        elif roll < 0.80 and far:
            return random.choice(far)
        else:
            
            return home

    def _apply_fraud_type(self, fraud_type, user, txn, state):
        """
        Returns (mods, burst_plan).
        mods: dict applied to the PRIMARY transaction (same as before).
        burst_plan: list of dicts, each describing one ADDITIONAL real
            transaction to emit right after the primary one:
                {
                  'delay_seconds': float,
                  'merchant_id': str (optional),
                  'amount': float (optional),
                  'device_id': str (optional),
                  'location': str (optional),
                }
            Any key omitted inherits from the primary (already-mutated)
            transaction. Only velocity_fraud, money_mule and
            merchant_collusion populate this; all other fraud types
            return an empty list, matching their true single-event
            nature.
        """
        mods = {}
        burst_plan = []
        ref = self._amount_reference(user, state)

        if fraud_type == 'amount_spike':
            multiplier = random.uniform(5, 15)
            mods['amount'] = round(ref * multiplier, 2)
            if random.random() < 0.3:
                mods['location'] = random.choice([d for d in DIVISIONS if d != user['home_location']])

        elif fraud_type == 'device_takeover':
            mods['device_id'] = f"D{random.randint(50001,100000)}"
            mods['location'] = random.choice(
                [d for d in DIVISIONS if d != user['home_location']]
            )
            mods['amount'] = round(ref * random.uniform(2.5,5),2)
            

        elif fraud_type == 'location_spoofing':
          
            mods['location'] = self._pick_spoofed_location(user)
            if random.random() < 0.50:
                mods['device_id'] = f"D{random.randint(50001, 100000)}"
            if random.random() < 0.35:
                mods['hour'] = random.choice([0, 1, 2, 3, 22, 23])
            if random.random() < 0.40:
                mods['amount'] = round(ref * random.uniform(1.3, 2.5), 2)

        elif fraud_type == 'merchant_collusion':
            # Real collusion = repeated real visits to the same risky merchant
            # in a short window, not one inflated row.
            high_risk_merchants = [m for m in self.merchants if m['merchant_risk'] > 0.85]
            collusion_merchant = random.choice(high_risk_merchants) if high_risk_merchants \
                else random.choice(self.merchants)
            mods['merchant_id'] = collusion_merchant['merchant_id']
            mods['merchant_risk'] = collusion_merchant['merchant_risk']
            mods['amount'] = round(ref * random.uniform(1.5, 4), 2)

            repeat_visits = random.randint(3, 6)
            for _ in range(repeat_visits):
                burst_plan.append({
                    'delay_seconds': random.uniform(120, 1800),  # 2-30 min apart
                    'merchant_id': collusion_merchant['merchant_id'],
                    'amount': round(ref * random.uniform(1.2, 3.5), 2),
                })

        elif fraud_type == 'velocity_fraud':
            # Real rapid-fire burst of transactions across merchants.
            mods['amount'] = round(ref * random.uniform(0.8, 1.5), 2)
            burst_size = random.randint(4, 8)
            for _ in range(burst_size):
                burst_merchant = random.choice(self.merchants)
                burst_plan.append({
                    'delay_seconds': random.uniform(15, 90),  # rapid
                    'merchant_id': burst_merchant['merchant_id'],
                    'amount': round(ref * random.uniform(0.7, 2.0), 2),
                })

        elif fraud_type == 'account_takeover':
           
            indicators = ['device', 'location', 'hour', 'amount']
            chosen = set(random.sample(indicators, random.randint(2, 3)))
            if 'device' in chosen:
                mods['device_id'] = f"D{random.randint(50001, 100000)}"
            if 'location' in chosen:
                mods['location'] = random.choice([d for d in DIVISIONS if d != user['home_location']])
            if 'hour' in chosen:
                mods['hour'] = random.choice([0, 1, 2, 3, 22, 23])
            if 'amount' in chosen:
                mods['amount'] = round(ref * random.uniform(3, 8), 2)
            else:
                # still somewhat elevated, but not the loud giveaway spike
                mods['amount'] = round(ref * random.uniform(1.2, 2.0), 2)

        elif fraud_type == 'synthetic_identity':

        
            mods['account_age_days'] = random.choices(
            [random.randint(1, 14), random.randint(15, 60)],
            weights=[0.7, 0.3])[0]

            
            indicators = random.sample(
                ['device', 'device_age', 'location', 'amount'],
                random.randint(2, 3)
            )

            if 'device' in indicators:
                mods['device_id'] = f"D{random.randint(50001, 100000)}"

            if 'device_age' in indicators:
                mods['device_age_days'] = random.randint(1, 30)

            if 'location' in indicators:
                mods['location'] = random.choice(
                    [d for d in DIVISIONS if d != user['home_location']]
                )

            if 'amount' in indicators:
                mods['amount'] = round(ref * random.uniform(1.3, 3.0), 2)

        elif fraud_type == 'money_mule':

            eligible = [m for m in self.merchants if m['merchant_risk'] > 0.5]
            pool = eligible if eligible else self.merchants
            n_recipients = random.randint(4, 7)
            recipients = random.sample(pool, min(n_recipients, len(pool)))

            mods['amount'] = round(ref * random.uniform(1.5, 4), 2)
            mods['merchant_id'] = recipients[0]['merchant_id']

            for m in recipients[1:]:
                burst_plan.append({
                    'delay_seconds': random.uniform(30, 240),
                    'merchant_id': m['merchant_id'],
                    'amount': round(ref * random.uniform(1.0, 3.5), 2),
                })

        elif fraud_type == 'subtle':
       
            if random.random() < 0.12:
                pass
            else:
                if random.random() < 0.7:
                    mods['amount'] = round(ref * random.uniform(1.1, 1.7), 2)
                if random.random() < 0.5:
                    mods['hour'] = min(23, max(0, user['preferred_hour'] + random.randint(-2, 3)))
                if random.random() < 0.20:
                 
                    mods['merchant_id'] = random.choice(self.merchants)['merchant_id']
                if random.random() < 0.10:
                    # rare, mild device drift (not a full takeover-style swap)
                    mods['device_id'] = f"D{random.randint(50001, 100000)}"

        return mods, burst_plan





class LegitNoiseEngine:
    def __init__(self, merchants, config):
        self.merchants = merchants
        
        self.p_burst = config.get('legit_burst_prob', 0.035)
        self.p_spike = config.get('legit_spike_prob', 0.025)
        self.p_travel = config.get('legit_travel_prob', 0.02)

    def maybe_burst(self, user):
        """A few genuine quick errands: pay bills, top up mobile, buy
        groceries - all in the space of several minutes. Softens the
        near-perfect velocity separability without touching fraud."""
        if random.random() >= self.p_burst:
            return []
        burst_size = random.randint(2, 4)
        plan = []
        for _ in range(burst_size):
            m = random.choice(self.merchants)
            plan.append({
                'delay_seconds': random.uniform(60, 400),  # minutes apart, not seconds - unhurried
                'merchant_id': m['merchant_id'],
                'amount': round(m['avg_ticket'] * random.uniform(0.5, 1.3), 2),
            })
        return plan

    def maybe_spike(self, user, state, ref_amount):

        if random.random() >= self.p_spike:
            return None

        if state.transaction_count < 10:
            return None

        base = max(ref_amount, user["base_amount"])

        amount = np.random.normal(
            base * 2.5,
            base * 0.5
        )

        amount = max(user["base_amount"], amount)
        amount = min(amount, user["base_amount"] * 10)

        return round(amount, 2)
    def maybe_travel(self, user):
        """An occasional genuine trip - visiting family, a work trip.
        Narrows the distance/impossible-travel gap. impossible_travel is
        NOT force-set; it emerges naturally from real time gaps, exactly
        like fraud's own location spoofing."""
        if random.random() >= self.p_travel:
            return None
        far = [d for d in DIVISIONS if calc_distance(user['home_location'], d) >= 300]
        candidates = far if far else [d for d in DIVISIONS if d != user['home_location']]
        return random.choice(candidates)






class TransactionGenerator:
    def __init__(self, users, merchants, state_engine, fraud_engine, legit_noise, clock, config):
        self.users = users
        self.users_by_id = {u['user_id']: u for u in users}
        self.merchants = merchants
        self.state_engine = state_engine
        self.fraud_engine = fraud_engine
        self.legit_noise = legit_noise
        self.clock = clock
        self.config = config
        self.user_weights = self._calculate_user_weights()
        self.graph = nx.DiGraph()
        self._txn_counter = 0

    def _calculate_user_weights(self):
        weights = {}
        for user in self.users:
            weights[user['user_id']] = user.get('activity_level', 0.5) * (1 + random.uniform(-0.2, 0.2))
        return weights

    def _select_user(self):
        users_list = list(self.users)
        weights = [self.user_weights.get(u['user_id'], 0.5) for u in users_list]
        return random.choices(users_list, weights=weights)[0]

    def _select_merchant(self, user):
        prefs = user.get('category_preferences', {})
        if prefs and random.random() < 0.7:
            categories = list(prefs.keys())
            weights = list(prefs.values())
            selected_category = random.choices(categories, weights=weights)[0]
            available = [m for m in self.merchants if m['category'] == selected_category]
            if available:
                return random.choice(available)
        return random.choice(self.merchants)

    def _merchant_by_id(self, merchant_id):
        for m in self.merchants:
            if m['merchant_id'] == merchant_id:
                return m
        return random.choice(self.merchants)

    def _generate_amount(self, user, merchant, seasonal_mult=1.0):
        seg_info = USER_SEGMENTS[user['segment']]
        txn_min, txn_max = seg_info['txn_range']
        log_mean = (np.log(txn_min) + np.log(txn_max)) / 2
        log_std = (np.log(txn_max) - np.log(txn_min)) / 4
        amount = np.exp(np.random.normal(log_mean, log_std))
        amount = max(txn_min, min(txn_max, amount))

        amount *= seasonal_mult
        if random.random() < 0.2:
            merchant_amount = np.random.normal(merchant['avg_ticket'], merchant['avg_ticket'] * 0.3)
            amount = 0.7 * amount + 0.3 * merchant_amount
            amount = max(txn_min, min(txn_max, amount))

        amount = max(10, amount)
        return round(amount, 2)

    def _generate_location(self, user):
        if random.random() < 0.7:
            return user['home_location']
        return random.choice([d for d in DIVISIONS if d != user['home_location']])

    def _generate_device(self, user):
        if random.random() < 0.85:
            return user['primary_device']
        return f"D{random.randint(1, self.config['n_devices'])}"

    def _is_possible_travel(self, loc1, loc2, minutes):
        if loc1 == loc2:
            return True
        distance = calc_distance(loc1, loc2)
        min_time = (distance / 100) * 60
        return minutes >= min_time * 0.8

    def _next_txn_id(self):
        tid = f"T{self._txn_counter:06d}"
        self._txn_counter += 1
        return tid

    def _finalize_and_record(self, txn, user, state, successful_txns):
        """
        Shared tail-end pipeline for ANY transaction (primary, fraud
        burst, or legit-noise burst): apply platform-limit flagging,
        compute features from PURE (not-yet-updated) state, then update
        state and record. This guarantees every row - regardless of
        where it came from - gets honest, non-leaked features.
        """
        true_amount = txn['amount']
        txn['exceeds_limit_flag'] = int(true_amount > BKASH_LIMITS['per_transaction_max'])
        txn['amount'] = min(true_amount, BKASH_LIMITS['per_transaction_max'])

        if state and state.last_location:
            time_diff = (txn['timestamp'] - state.last_timestamp).total_seconds() / 60 \
                if state.last_timestamp else 60
            txn['impossible_travel'] = txn.get(
                'impossible_travel',
                0 if self._is_possible_travel(state.last_location, txn['location'], time_diff) else 1
            )
        else:
            txn.setdefault('impossible_travel', 0)

        if state:
            behavioral = state.get_behavioral_features(
                current_device=txn['device_id'], current_time=txn['timestamp']
            )
            for key, value in behavioral.items():
                txn[key] = value

            txn['device_change_flag'] = int(txn['device_id'] != state.primary_device)
            txn['new_device_flag'] = int(txn['device_id'] not in state.device_history)
            txn['new_location_flag'] = int(txn['location'] not in state.location_history)
            txn['location_change_flag'] = int(txn['location'] != state.home)
            txn['merchant_change_flag'] = int(txn['merchant_id'] not in state.merchant_history)

            anomaly = state.detect_anomalies(txn)
            txn['anomaly_score'] = anomaly['score']
            txn['anomaly_signals'] = str(anomaly['signals'])

            txn['amount_to_avg_ratio'] = round(true_amount / (state.avg_amount + 1), 4)
            txn['amount_to_base_ratio'] = round(true_amount / (user['base_amount'] + 1), 4)
            txn['hour_diff_from_preferred'] = abs(txn['hour'] - user['preferred_hour'])
            if state.last_timestamp:
                txn['time_since_last_txn_minutes'] = round(
                    (txn['timestamp'] - state.last_timestamp).total_seconds() / 60, 2
                )
            else:
                txn['time_since_last_txn_minutes'] = -1.0
            txn['distance_from_home_km'] = round(calc_distance(state.home, txn['location']), 2)

            hist = list(state.amount_history)
            if len(hist) >= 3:
                hist_mean = float(np.mean(hist))
                hist_std = max(
                    float(np.std(hist)),
                    user["base_amount"] * 0.15,
                )
                z = (true_amount - hist_mean) / hist_std
                txn["amount_zscore"] = round(np.clip(z, -10, 10), 4)

                txn["amount_percentile"] = round(
                    float(np.mean([h <= true_amount for h in hist])),
                    4
                )

                txn["amount_to_rolling_median_ratio"] = round(
                    true_amount / (float(np.median(hist)) + 1),
                    4
                )
            else:
                txn['amount_zscore'] = 0.0
                txn['amount_percentile'] = 0.5
                txn['amount_to_rolling_median_ratio'] = 1.0

        if state and txn["fraud_label"] == 0:
            state.add_transaction(txn)

        self.graph.add_edge(txn['user_id'], txn['merchant_id'],
                             amount=txn['amount'], fraud=txn['fraud_label'])
        successful_txns.append(txn)

    def _emit_burst(self, primary_txn, burst_plan, user, successful_txns, fraud_label, fraud_type):
        """
        Emit each planned follow-up transaction as a real, sequential
        row. Shared by both fraud bursts (fraud_label=1) and legit-noise
        bursts (fraud_label=0) - same mechanism, different label and
        source of the plan.
        """
        cursor_time = primary_txn['timestamp']
        for step in burst_plan:
            cursor_time = cursor_time + timedelta(seconds=step['delay_seconds'])
            merchant_id = step.get('merchant_id', primary_txn['merchant_id'])
            merchant = self._merchant_by_id(merchant_id)
            device_id = step.get('device_id', primary_txn['device_id'])
            location = step.get('location', primary_txn['location'])
            amount = step.get('amount', primary_txn['amount'])

            state = self.state_engine.get(user['user_id'])
            burst_txn = {
                'transaction_id': self._next_txn_id(),
                'user_id': user['user_id'],
                'merchant_id': merchant['merchant_id'],
                'amount': round(amount, 2),
                'hour': cursor_time.hour,
                'day': cursor_time.weekday(),
                'location': location,
                'device_id': device_id,
                'timestamp': cursor_time,
                'user_segment': user['segment'],
                'user_risk': user['risk_profile'],
                'merchant_category': merchant['category'],
                'merchant_risk': merchant['merchant_risk'],
                'account_age_days': primary_txn.get('account_age_days', user.get('account_age_days', 0)),
                'device_age_days': state.device_ages.get(device_id, 0) if state else 0,
                'fraud_label': fraud_label,
                'fraud_score': round(random.uniform(0.6, 0.9), 3) if fraud_label else 0,
                'fraud_type': fraud_type,
            }
            self._finalize_and_record(burst_txn, user, state, successful_txns)

        self.clock.advance_to(cursor_time)

    def generate_all(self, n_primary_transactions):
        max_total = self.config.get('max_total_transactions', int(n_primary_transactions * 1.3))
        successful_txns = []
        primary_count = 0

        while primary_count < n_primary_transactions and len(successful_txns) < max_total:
            self.clock.advance()

            user = self._select_user()
            user_id = user['user_id']
            state = self.state_engine.get(user_id)
            merchant = self._select_merchant(user)
            seasonal_mult = self.clock.get_seasonal_multiplier()

            amount = self._generate_amount(user, merchant, seasonal_mult)
            location = self._generate_location(user)
            device_id = self._generate_device(user)
            device_age_days = state.device_ages.get(device_id, 0) if state else 0

            txn = {
                'transaction_id': self._next_txn_id(),
                'user_id': user_id,
                'merchant_id': merchant['merchant_id'],
                'amount': amount,
                'hour': self.clock.current.hour,
                'day': self.clock.current.weekday(),
                'location': location,
                'device_id': device_id,
                'timestamp': self.clock.current,
                'user_segment': user['segment'],
                'user_risk': user['risk_profile'],
                'merchant_category': merchant['category'],
                'merchant_risk': merchant['merchant_risk'],
                'account_age_days': user.get('account_age_days', 0),
                'device_age_days': device_age_days,
            }

            is_fraud, fraud_score, fraud_type, mods, burst_plan = self.fraud_engine.process(user, txn, state)

            if is_fraud:
                txn.update(mods)
                if 'merchant_id' in mods and 'merchant_risk' not in mods:
                    new_merchant = self._merchant_by_id(mods['merchant_id'])
                    txn['merchant_category'] = new_merchant['category']
                    txn['merchant_risk'] = new_merchant['merchant_risk']
                txn['fraud_label'] = 1
                txn['fraud_score'] = fraud_score
                txn['fraud_type'] = fraud_type
            else:
                txn['fraud_label'] = 0
                txn['fraud_score'] = 0
                txn['fraud_type'] = 'normal'

                # ---- Legit noise: only rolled when NOT fraud ----
                ref = state.avg_amount if (state and state.transaction_count >= 3) else user['base_amount']

                spike_amount = self.legit_noise.maybe_spike(user, state, ref)
                if spike_amount is not None:
                    txn['amount'] = spike_amount

                travel_location = self.legit_noise.maybe_travel(user)
                if travel_location is not None:
                    txn['location'] = travel_location

                burst_plan = self.legit_noise.maybe_burst(user)

            self._finalize_and_record(txn, user, state, successful_txns)
            primary_count += 1

            if burst_plan:
                self._emit_burst(txn, burst_plan, user, successful_txns,
                                  fraud_label=txn['fraud_label'], fraud_type=txn['fraud_type'])

        print(f"    Generated {len(successful_txns)} transactions "
              f"({primary_count} primary events + {len(successful_txns) - primary_count} burst rows)")
        return pd.DataFrame(successful_txns)


# MODULE 9: QUALITY ASSURANCE



class QualityAssurance:
    @staticmethod
    def validate(df, users_df, merchants_df, config):
        print("\n[Quality Assurance] Validating dataset...")
        errors = []
        target_rate = config['base_fraud_rate']
        actual_rate = df['fraud_label'].mean()

        # Wider tolerance since burst rows shift the raw rate somewhat
        if abs(actual_rate - target_rate) > 0.03:
            errors.append(f"Fraud rate {actual_rate*100:.2f}% far from target ({target_rate*100:.2f}%)")

        missing = df.isnull().sum().sum()
        if missing > 0:
            errors.append(f"Found {missing} missing values")

        if df['transaction_id'].nunique() != len(df):
            errors.append("Duplicate transaction IDs found")

        invalid_users = set(df['user_id']) - set(users_df['user_id'])
        if invalid_users:
            errors.append(f"Invalid user IDs found: {list(invalid_users)[:5]}")

        invalid_merchants = set(df['merchant_id']) - set(merchants_df['merchant_id'])
        if invalid_merchants:
            errors.append(f"Invalid merchant IDs found: {list(invalid_merchants)[:5]}")

        if not pd.Series(df['timestamp']).is_monotonic_increasing:
            errors.append("Transactions not in chronological order")

        if (df['amount'] < 0).any():
            errors.append("Negative amounts found")
        if ((df['hour'] < 0) | (df['hour'] > 23)).any():
            errors.append("Invalid hours found")

        if errors:
            print("❌ Validation FAILED:")
            for err in errors:
                print(f"  - {err}")
            return False
        else:
            print("✅ All validation checks passed!")
            return True


# MAIN PIPELINE



class FraudDatasetGenerator:
    def __init__(self, config):
        self.config = config
        np.random.seed(config['seed'])
        random.seed(config['seed'])
        self.state_engine = {}

    def generate(self):
        print("=" * 70)
        print("FRAUD GENERATOR v4.0 - Realistic Class Overlap")
        print("=" * 70)

        print("\n[Module 2] Generating users...")
        user_gen = UserGenerator(self.config)
        users_df = user_gen.generate()
        users = users_df.to_dict('records')
        print(f"    Generated {len(users)} users")

        print("[Module 3] Generating merchants...")
        merchant_gen = MerchantGenerator(self.config)
        merchants_df = merchant_gen.generate()
        merchants = merchants_df.to_dict('records')
        print(f"    Generated {len(merchants)} merchants")

        print("[Module 4] Generating fraud rings...")
        ring_gen = FraudRingGenerator(self.config, users_df, merchants_df)
        fraud_rings = ring_gen.generate()
        print(f"    Generated {len(fraud_rings)} fraud rings")

        print("[Module 5] Initializing user states...")
        for user in users:
            self.state_engine[user['user_id']] = UserState(user)
        print(f"    Initialized {len(self.state_engine)} user states")

        print("[Module 6] Starting simulation clock...")
        clock = SimulationClock(self.config['start_date'], self.config['days_to_simulate'])

        print("[Module 7-8] Starting transaction generation...")
        fraud_engine = FraudEngine(fraud_rings, users, merchants)
        legit_noise = LegitNoiseEngine(merchants, self.config)
        transaction_gen = TransactionGenerator(
            users, merchants, self.state_engine, fraud_engine, legit_noise, clock, self.config
        )

        df = transaction_gen.generate_all(self.config['n_transactions'])
        print(f"    Generated {len(df)} total rows")

        print("\n[Module 9] Running quality assurance...")
        QualityAssurance.validate(df, users_df, merchants_df, self.config)

        print("\nSaving datasets...")
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATA_DIR = os.path.join(BASE_DIR, "data")

        os.makedirs(DATA_DIR, exist_ok=True)

        df.to_csv(os.path.join(DATA_DIR, "transactions.csv"), index=False)
        users_df.to_csv(os.path.join(DATA_DIR, "users.csv"), index=False)  # Fixed: use DATA_DIR
        merchants_df.to_csv(os.path.join(DATA_DIR, "merchants.csv"), index=False)  # Fixed: use DATA_DIR
        nx.write_graphml(transaction_gen.graph, os.path.join(DATA_DIR, "graph.graphml"))  # Fixed: use DATA_DIR

        metadata = {
            "version": self.config['version'],
            "generated_date": datetime.now().isoformat(),
            "seed": self.config['seed'],
            "n_transactions": len(df),
            "n_users": len(users_df),
            "n_merchants": len(merchants_df),
            "fraud_rate": float(df['fraud_label'].mean()),
            "fraud_type_distribution": df[df['fraud_label'] == 1]['fraud_type'].value_counts().to_dict(),
            "user_segment_distribution": users_df['segment'].value_counts().to_dict(),
            "features": df.columns.tolist()
        }
        with open(os.path.join(DATA_DIR, "metadata.json"), "w") as f:  # Fixed: use DATA_DIR
            json.dump(metadata, f, indent=2)

        print("    Saved to 'data/' directory")
        print("    - transactions.csv")
        print("    - users.csv")
        print("    - merchants.csv")
        print("    - graph.graphml")
        print("    - metadata.json")

        return df, users_df, merchants_df


if __name__ == "__main__":
    generator = FraudDatasetGenerator(CONFIG)
    df, users, merchants = generator.generate()