"""
Fraud Detection System - Web Interface
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import predict_with_state, DIVISIONS, BKASH_LIMITS, reset_user_state

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SAFE HELPERS
# ============================================================

def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        v = float(value)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


def safe_string(value, default=""):
    if pd.isna(value):
        return default
    value = str(value).strip()
    return value if value else default


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    
    .header {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .header h1 { margin: 0; font-size: 2.5rem; font-weight: 700; }
    .header p { margin: 0.5rem 0 0 0; opacity: 0.8; font-size: 1.1rem; }
    
    .card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .card h3 {
        margin-top: 0;
        color: #1a1a2e;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border-left: 4px solid;
    }
    
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.9rem; color: #666; }
    
    .risk-critical { color: #dc3545; font-weight: 700; }
    .risk-high { color: #fd7e14; font-weight: 700; }
    .risk-medium { color: #ffc107; font-weight: 700; }
    .risk-low { color: #28a745; font-weight: 700; }
    
    .result-container {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .result-fraud {
        background: #f8d7da;
        border: 2px solid #dc3545;
    }
    
    .result-safe {
        background: #d4edda;
        border: 2px solid #28a745;
    }
    
    .footer {
        text-align: center;
        padding: 1rem;
        color: #666;
        font-size: 0.8rem;
        border-top: 1px solid #e0e0e0;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL METADATA
# ============================================================

def load_model_metadata():
    try:
        model_paths = [
            '../models/best_model_fixed.pkl',
            'E:/Fraud_Project2/models/best_model_fixed.pkl',
            'models/best_model_fixed.pkl',
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    return {
                        'name': data.get('name', 'XGBoost'),
                        'threshold': data.get('threshold', 0.55),
                        'test_f1': data.get('test_f1', 0.8161),
                        'test_recall': data.get('test_recall', 0.8115),
                        'test_precision': data.get('test_precision', 0.8208),
                        'test_roc_auc': data.get('test_roc_auc', 0.9654),
                        'features': data.get('features', []),
                        'train_samples': data.get('train_samples', 83246),
                        'val_samples': data.get('val_samples', 11892),
                        'test_samples': data.get('test_samples', 23785),
                    }
        return None
    except Exception:
        return None


model_meta = load_model_metadata()

if model_meta is None:
    model_meta = {
        'name': 'XGBoost',
        'threshold': 0.55,
        'test_f1': 0.8161,
        'test_recall': 0.8115,
        'test_precision': 0.8208,
        'test_roc_auc': 0.9654,
        'features': [],
        'train_samples': 83246,
        'val_samples': 11892,
        'test_samples': 23785,
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked--v1.png", width=80)
    st.title("Fraud Shield")
    st.markdown("---")
    
    st.markdown("### System Status")
    st.success(f"**Model:** {model_meta['name']}")
    st.info(f"**Threshold:** {model_meta['threshold']:.2f}")
    st.info("**Status:** Active")
    
    st.markdown("---")
    
    st.markdown("### Model Performance")
    st.metric("F1 Score", f"{model_meta['test_f1']:.4f}")
    st.metric("Recall", f"{model_meta['test_recall']:.4f}")
    st.metric("Precision", f"{model_meta['test_precision']:.4f}")
    st.metric("ROC-AUC", f"{model_meta['test_roc_auc']:.4f}")
    
    st.markdown("---")
    
    st.markdown("### Dataset Split")
    st.metric("Training", f"{model_meta['train_samples']:,}")
    st.metric("Validation", f"{model_meta['val_samples']:,}")
    st.metric("Test", f"{model_meta['test_samples']:,}")
    
    if st.button("Reset User States"):
        reset_user_state()
        st.success("All user states reset")
    
    st.markdown("---")
    st.caption("All transactions are encrypted and secure")


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="header">
    <h1>Real-Time Fraud Detection</h1>
    <p>Bangladesh Mobile Banking - AI-Powered Transaction Monitoring</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MANUAL INPUT
# ============================================================

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Transaction Details")
    
    with st.form("transaction_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            user_id = st.text_input(
                "User ID",
                placeholder="e.g., U00001",
                help="Enter the user's ID"
            )
        with col_b:
            amount = st.number_input(
                "Amount (BDT)",
                min_value=0.0,
                max_value=50000.0,
                value=1000.0,
                step=100.0,
                help="Transaction amount in Bangladeshi Taka"
            )
        
        col_c, col_d = st.columns(2)
        with col_c:
            location = st.selectbox(
                "Location",
                options=DIVISIONS,
                help="Transaction location"
            )
        with col_d:
            merchant_id = st.text_input(
                "Merchant ID",
                placeholder="e.g., M0001",
                help="Enter the merchant's ID"
            )
        
        col_e, col_f = st.columns(2)
        with col_e:
            hour = st.slider(
                "Hour of Day",
                min_value=0,
                max_value=23,
                value=12,
                help="Transaction hour (0-23)"
            )
        with col_f:
            device_id = st.text_input(
                "Device ID",
                placeholder="e.g., D0001",
                help="Enter the device ID"
            )
        
        minutes_ago = st.number_input(
            "Minutes since last transaction",
            min_value=-1,
            max_value=1440,
            value=-1,
            help="Leave as -1 if unknown"
        )
        if minutes_ago == -1:
            minutes_ago = None
        
        col_submit, col_reset = st.columns([1, 1])
        with col_submit:
            submitted = st.form_submit_button(
                "Analyze Transaction",
                use_container_width=True,
                type="primary"
            )
        with col_reset:
            reset = st.form_submit_button(
                "Reset",
                use_container_width=True
            )
    
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# LIVE FEED
# ============================================================

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Live Feed")
    
    if "predictions" not in st.session_state:
        st.session_state.predictions = []
    
    if len(st.session_state.predictions) > 0:
        recent = st.session_state.predictions[-5:]
        for pred in reversed(recent):
            if pred.get('fraud', False):
                st.error(f"FRAUD detected: {pred.get('amount', 0):,.0f} BDT")
            else:
                st.success(f"SAFE transaction: {pred.get('amount', 0):,.0f} BDT")
            st.caption(f"ID: {pred.get('user_id', 'Unknown')} | {pred.get('time', '')}")
    else:
        st.info("No transactions analyzed yet")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Quick Stats")
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Total Transactions", len(st.session_state.predictions))
    with col_stat2:
        fraud_count = sum(1 for p in st.session_state.predictions if p.get('fraud', False))
        fraud_rate = (fraud_count / max(1, len(st.session_state.predictions))) * 100
        st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# MANUAL PREDICTION RESULT
# ============================================================

if submitted:
    with st.spinner("Analyzing transaction..."):
        time.sleep(0.5)
        
        try:
            # Build a timestamp for this manual transaction
            now = datetime.now()
            manual_ts = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            
            prediction, probability, risk_level, factors, txn = predict_with_state(
                user_id=user_id,
                amount=amount,
                location=location,
                merchant_id=merchant_id,
                hour=hour,
                device_id=device_id,
                minutes_since_last=minutes_ago,
                timestamp=manual_ts
            )
            
            st.session_state.predictions.append({
                'user_id': user_id,
                'amount': amount,
                'fraud': prediction == 1,
                'time': datetime.now().strftime("%H:%M:%S")
            })
            
            st.markdown("---")
            st.markdown("## Analysis Results")
            
            if prediction == 1:
                st.markdown("""
                <div class="result-container result-fraud">
                    <h1 style="color: #dc3545; margin: 0;">FRAUD DETECTED</h1>
                    <p style="font-size: 1.2rem; margin: 0.5rem 0;">This transaction has been flagged as potentially fraudulent.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-container result-safe">
                    <h1 style="color: #28a745; margin: 0;">SAFE TRANSACTION</h1>
                    <p style="font-size: 1.2rem; margin: 0.5rem 0;">This transaction appears to be legitimate.</p>
                </div>
                """, unsafe_allow_html=True)
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            risk_color = {
                "CRITICAL": "risk-critical",
                "HIGH": "risk-high",
                "MEDIUM": "risk-medium",
                "LOW": "risk-low"
            }.get(risk_level, "risk-medium")
            
            with col_m1:
                st.metric("Fraud Probability", f"{probability:.1%}")
            
            with col_m2:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {'#dc3545' if prediction == 1 else '#28a745'}">
                    <div class="metric-label">Risk Level</div>
                    <div class="metric-value {risk_color}">{risk_level}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m3:
                st.metric("Prediction", "FRAUD" if prediction == 1 else "SAFE")
            
            with col_m4:
                st.metric("Amount", f"{amount:,.0f} BDT")
            
            st.markdown("### Contributing Factors")
            
            if factors:
                clean_factors = []
                for f in factors:
                    if isinstance(f, str):
                        clean_factors.append(f)
                    elif isinstance(f, list):
                        clean_factors.extend(f)
                
                for f in clean_factors[:15]:
                    if "Rule Engine" in f:
                        st.markdown(f"**{f}**")
                    elif "•" in f:
                        st.markdown(f"  {f}")
                    else:
                        st.markdown(f"- {f}")
            else:
                st.info("No significant risk factors detected.")
            
            with st.expander("Transaction Details"):
                st.json({
                    "User ID": user_id,
                    "Amount": amount,
                    "Location": location,
                    "Merchant ID": merchant_id,
                    "Hour": hour,
                    "Device ID": device_id,
                    "Minutes Since Last": minutes_ago if minutes_ago is not None else "Unknown",
                    "Fraud Probability": f"{probability:.1%}",
                    "Risk Level": risk_level,
                    "Prediction": "FRAUD" if prediction == 1 else "SAFE"
                })
            
        except Exception as e:
            st.error(f"Error analyzing transaction: {str(e)}")
            st.code(str(e), language="python")


# ============================================================
# BULK ANALYSIS
# ============================================================

with st.expander("Bulk Analysis (Upload CSV)"):
    st.markdown("""
    Upload a CSV file with transactions to analyze in bulk.
    
    **Required columns:**
    - `user_id`, `amount`, `location`, `merchant_id`, `hour`, `device_id`
    
    **Recommended columns:**
    - `timestamp` - Ensures chronological state building (no future leakage)
    
    **Optional column:**
    - `minutes_since_last` - Used only if `timestamp` is not provided
    - `fraud_label` - For comparison after prediction
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Handle BOM and comments
            df = pd.read_csv(uploaded_file, comment='#', encoding='utf-8-sig')
            
            required_cols = ['user_id', 'amount', 'location', 'merchant_id', 'hour', 'device_id']
            missing_cols = [c for c in required_cols if c not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.info(f"Required: {required_cols}")
            else:
                st.write(f"Loaded {len(df)} transactions")
                st.dataframe(df.head())
                st.caption(f"Available columns: {', '.join(df.columns)}")
                
                if st.button("Analyze All"):
                    reset_user_state()
                    
                    with st.spinner(f"Analyzing {len(df)} transactions..."):
                        results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, row in df.iterrows():
                            status_text.text(f"Processing {idx + 1}/{len(df)}")
                            
                            try:
                                # Safe extraction
                                uid = safe_string(row.get('user_id'), "U00001")
                                amt = safe_float(row.get('amount'), 1000.0)
                                loc = safe_string(row.get('location'), "Dhaka")
                                mid = safe_string(row.get('merchant_id'), "M0001")
                                hr = safe_int(row.get('hour'), 12)
                                did = safe_string(row.get('device_id'), "D0001")
                                
                                # minutes_since_last
                                mins = None
                                if 'minutes_since_last' in df.columns:
                                    raw = row.get('minutes_since_last')
                                    if pd.notna(raw):
                                        mins = safe_float(raw, None)
                                
                                # timestamp (authoritative if present)
                                ts = None
                                if 'timestamp' in df.columns:
                                    raw_ts = row.get('timestamp')
                                    if pd.notna(raw_ts):
                                        ts = pd.to_datetime(raw_ts, errors='coerce')
                                        if pd.isna(ts):
                                            ts = None
                                
                                # Predict
                                pred, prob, risk, factors, _ = predict_with_state(
                                    user_id=uid,
                                    amount=amt,
                                    location=loc,
                                    merchant_id=mid,
                                    hour=hr,
                                    device_id=did,
                                    minutes_since_last=mins,
                                    timestamp=ts
                                )
                                
                                # Actual label if available
                                actual = None
                                if 'fraud_label' in df.columns:
                                    raw_lbl = row.get('fraud_label')
                                    if pd.notna(raw_lbl):
                                        try:
                                            actual = int(float(raw_lbl))
                                        except Exception:
                                            actual = None
                                
                                results.append({
                                    'row': idx + 1,
                                    'user_id': uid,
                                    'amount': amt,
                                    'location': loc,
                                    'merchant_id': mid,
                                    'hour': hr,
                                    'device_id': did,
                                    'minutes_since_last': mins if mins is not None else 'N/A',
                                    'actual_label': actual if actual is not None else 'N/A',
                                    'prediction': 'FRAUD' if pred == 1 else 'SAFE',
                                    'probability': f"{prob:.2%}",
                                    'risk': risk,
                                    'top_factor': factors[0] if factors else 'None'
                                })
                                
                            except Exception as e:
                                results.append({
                                    'row': idx + 1,
                                    'user_id': safe_string(row.get('user_id'), 'Unknown'),
                                    'amount': safe_float(row.get('amount'), 0),
                                    'location': safe_string(row.get('location'), 'Unknown'),
                                    'merchant_id': safe_string(row.get('merchant_id'), 'Unknown'),
                                    'hour': safe_int(row.get('hour'), 0),
                                    'device_id': safe_string(row.get('device_id'), 'Unknown'),
                                    'minutes_since_last': 'N/A',
                                    'actual_label': 'N/A',
                                    'prediction': 'ERROR',
                                    'probability': '0%',
                                    'risk': 'ERROR',
                                    'top_factor': str(e)[:80]
                                })
                            
                            progress_bar.progress((idx + 1) / len(df))
                        
                        status_text.text("Analysis complete!")
                        
                        result_df = pd.DataFrame(results)
                        st.success("Analysis complete!")
                        st.dataframe(result_df)
                        
                        # Summary
                        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                        
                        total = len(result_df)
                        fraud_pred = (result_df['prediction'] == 'FRAUD').sum()
                        errors = (result_df['prediction'] == 'ERROR').sum()
                        
                        st.metric("Total", total)
                        st.metric("Fraud Predictions", int(fraud_pred))
                        st.metric("Errors", int(errors))
                        
                        if 'actual_label' in result_df.columns:
                            actual_fraud = (result_df['actual_label'] == 1).sum()
                            st.metric("Actual Fraud", int(actual_fraud))
                        
                        # Download
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="bulk_analysis_results.csv",
                            mime="text/csv"
                        )
                        
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.code(str(e), language="python")


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    <p>Fraud Detection System v2.0 | Model: XGBoost</p>
    <p>Bangladesh Mobile Banking - Real-Time Transaction Monitoring</p>
</div>
""", unsafe_allow_html=True)

if reset:
    st.session_state.predictions = []
    reset_user_state()
    st.rerun()