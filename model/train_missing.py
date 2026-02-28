"""
Model 3: Missed Important Mentions Detection
=============================================
Detects messages from contacts that contain important mentions
(requests, emotional signals, key updates, action items) that
YOU (the user) have not acknowledged or responded to.

Architecture:
  - Sentence-Transformer (GPU)  →  384-d semantic embeddings
  - Keyword / pattern extractor →  named entity & urgency signals
  - XGBoost classifier          →  importance_missed probability
  - Threshold tuning            →  precision/recall trade-off

What counts as a "missed important mention":
  - Contact sent a high-importance message (importance_score > 0.6)
  - Intent is: request / question / emotional_support / complaint / planning
  - You (sender="me") did NOT reply within ACK_WINDOW_HRS hours
  - OR you replied but with a low-sentiment / dismissive message

Signal sources per message:
  importance_score       : pre-scored in synthetic data
  intent_label           : semantic category
  sentiment_score        : emotional weight of mention
  embedding              : semantic content (GPU-encoded)
  sim_to_important_arch  : cosine distance to known-important centroid
  reply_gap_hrs          : how long until "me" replied
  my_reply_sentiment     : was my reply warm/cold/absent
  my_reply_importance    : did I match their importance level
  contact_resp_ratio     : contact's general engagement level
  days_since_contact     : recency of relationship
  thread_importance_avg  : recent thread importance baseline

Install deps:
    pip install pandas numpy scikit-learn xgboost sentence-transformers
                torch tqdm matplotlib seaborn joblib shap
"""

import json
import re
import warnings
import os
from dotenv import load_dotenv

from pathlib import Path

load_dotenv()  # loads MONGO_URL (and other vars) from .env

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
DATA_DIR       = Path(".")
MODEL_DIR      = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH   = MODEL_DIR / "important_mentions_xgb.joblib"
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE    = 256

ACK_WINDOW_HRS      = 24     # hours within which a reply "acknowledges" the mention
IMPORTANCE_THRESH   = 0.55   # messages above this are "important"
RANDOM_STATE        = 42

# Intents that carry important mentions
IMPORTANT_INTENTS = {"request", "question", "emotional_support", "complaint", "planning"}

# Urgency keyword patterns (compiled once)
URGENCY_PATTERNS = re.compile(
    r"\b(urgent|asap|important|please|help|need|don't forget|remind|deadline"
    r"|emergency|critical|serious|worried|scared|please reply|must|have to"
    r"|can you|could you|would you|can we|let me know|get back to me)\b",
    re.IGNORECASE,
)

EMOTIONAL_PATTERNS = re.compile(
    r"\b(feel|feeling|sad|anxious|stressed|overwhelmed|lonely|hurt|upset"
    r"|crying|miss|miss you|struggling|hard time|going through|not okay"
    r"|depressed|exhausted|scared|worried|lost|confused|broken)\b",
    re.IGNORECASE,
)

QUESTION_PATTERNS = re.compile(r"\?")

ACTION_PATTERNS = re.compile(
    r"\b(can you|could you|would you|please send|please review|please check"
    r"|let me know|confirm|get back|follow up|call me|meet|schedule|book)\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────
# Device setup
# ─────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU : {props.name}  ({props.total_memory // 1024**2} MB VRAM)")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("  Device : Apple MPS")
    else:
        dev = torch.device("cpu")
        print("  Device : CPU  (GPU not found — encoding will be slower)")
    return dev


# ─────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────

def load_data(data_dir: Path = DATA_DIR):
    """
    Load contacts and messages from MongoDB.
    Falls back to reading local JSON files if MONGO_URL is not set.
    """
    mongo_url = os.getenv("MONGO_URL")

    if mongo_url:
        print(f"Loading data from MongoDB …")
        from pymongo import MongoClient

        client = MongoClient(mongo_url, serverSelectionTimeoutMS=15000)
        db = client["autopilot"]

        # ── Contacts ──────────────────────────────────────────────
        raw_contacts = list(db["contacts"].find({}))
        for doc in raw_contacts:
            doc.pop("_id", None)   # drop ObjectId
        contacts = pd.DataFrame(raw_contacts)

        # Normalise the health_score field name
        if "healthScore" in contacts.columns and "health_score" not in contacts.columns:
            contacts.rename(columns={"healthScore": "health_score"}, inplace=True)

        # ── Messages ──────────────────────────────────────────────
        raw_messages = list(db["messages"].find({}))
        for doc in raw_messages:
            doc.pop("_id", None)
            if "id" not in doc:
                doc["id"] = str(doc.get("message_id", id(doc)))
        messages = pd.DataFrame(raw_messages)

        client.close()
        print(f"  Source: MongoDB ({mongo_url[:40]}…)")
    else:
        print("MONGO_URL not set — falling back to local JSON files …")
        with open(data_dir / "contacts.json", encoding="utf-8") as f:
            contacts = pd.DataFrame(json.load(f))
        with open(data_dir / "messages.json", encoding="utf-8") as f:
            messages = pd.DataFrame(json.load(f))

    messages["timestamp"] = pd.to_datetime(
        messages["timestamp"], utc=True, errors="coerce"
    )
    messages.sort_values(["contact_id", "timestamp"], inplace=True)
    messages.reset_index(drop=True, inplace=True)

    print(f"  {len(contacts)} contacts | {len(messages):,} messages")
    return contacts, messages


# ─────────────────────────────────────────────────────────────
# 2. GPU Encoding
# ─────────────────────────────────────────────────────────────

def encode_messages(messages: pd.DataFrame, device: torch.device) -> np.ndarray:
    """Encode all message content on GPU. Falls back to pre-computed embeddings."""
    try:
        from sentence_transformers import SentenceTransformer

        print(f"\nEncoding {len(messages):,} messages with {ENCODER_NAME} on {device} …")
        encoder = SentenceTransformer(ENCODER_NAME, device=str(device))
        embeddings = encoder.encode(
            messages["content"].fillna("").tolist(),
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    except ImportError:
        print("  sentence-transformers not installed — loading pre-computed embeddings …")
        return _parse_precomputed(messages)


def _parse_precomputed(messages: pd.DataFrame) -> np.ndarray:
    embs = []
    for emb in tqdm(messages["embedding"], desc="Parsing embeddings"):
        if isinstance(emb, str):
            embs.append(json.loads(emb))
        elif isinstance(emb, list):
            embs.append(emb)
        else:
            embs.append([0.0] * EMBEDDING_DIM)
    arr = np.array(embs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr / norms


# ─────────────────────────────────────────────────────────────
# 3. Text signal extraction
# ─────────────────────────────────────────────────────────────

def extract_text_signals(text: str) -> dict:
    """
    Rule-based extraction of urgency, emotional, action, and question signals
    from raw message text. Returns a dict of binary/count features.
    """
    if not isinstance(text, str):
        text = ""
    return {
        "has_urgency_keyword":   int(bool(URGENCY_PATTERNS.search(text))),
        "has_emotional_keyword": int(bool(EMOTIONAL_PATTERNS.search(text))),
        "has_action_request":    int(bool(ACTION_PATTERNS.search(text))),
        "question_count":        len(QUESTION_PATTERNS.findall(text)),
        "word_count":            len(text.split()),
        "exclamation_count":     text.count("!"),
        "caps_ratio":            sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "has_name_mention":      int(any(w[0].isupper() for w in text.split() if len(w) > 2)),
    }


# ─────────────────────────────────────────────────────────────
# 4. Feature engineering
# ─────────────────────────────────────────────────────────────

def build_features(
    messages: pd.DataFrame,
    contacts: pd.DataFrame,
    embeddings: np.ndarray,
) -> pd.DataFrame:
    """
    For every message sent by 'contact', determine whether it was an important
    mention that 'me' missed (didn't acknowledge within ACK_WINDOW_HRS).

    Label:
      positive = (importance_score > IMPORTANCE_THRESH
                  AND intent in IMPORTANT_INTENTS
                  AND no adequate reply from 'me' within ACK_WINDOW_HRS)
               OR attention_gap_flag == True

    An "adequate reply" = me sent a message within the window
    with sentiment_score > 0.0 (not dismissive).
    """
    print("\nEngineering mention features …")

    # Build archetype embedding: centroid of known-important contact messages
    important_mask = (
        (messages["sender"] == "contact")
        & (messages["importance_score"].fillna(0) > IMPORTANCE_THRESH)
        & (messages["intent_label"].isin(IMPORTANT_INTENTS))
    )
    if important_mask.sum() > 20:
        arch_emb = embeddings[important_mask].mean(axis=0, keepdims=True)  # (1, 384)
    else:
        arch_emb = embeddings[:50].mean(axis=0, keepdims=True)

    # Cosine similarity of every message to "important" archetype
    sim_to_arch = (embeddings @ arch_emb.T).squeeze()  # (N,)

    # Build an index: msg_id → position in embeddings / messages array
    id_to_pos = {row["id"]: i for i, row in messages.iterrows()}

    contact_lookup = contacts.set_index("contact_id")
    records = []

    for cid, grp in tqdm(messages.groupby("contact_id"), desc="Contacts"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)

        # Contact-level context
        c = contact_lookup.loc[cid] if cid in contact_lookup.index else {}
        c_health    = float(c.get("health_score",        0.5))
        c_resp      = float(c.get("response_ratio",      0.5))
        c_ghosted   = int(bool(c.get("is_ghosted",       False)))
        c_churn     = float(c.get("churn_probability",   0.3))
        c_decay     = float(c.get("engagement_decay_rate", 0.1))
        c_days_since= int(c.get("days_since",            30))
        c_sentiment = float(c.get("sentiment_avg",       0.5))

        # Rolling importance baseline (last 10 messages)
        rolling_imp = grp["importance_score"].rolling(10, min_periods=1).mean()

        for i, row in grp.iterrows():
            # Only consider messages from the contact (they're the one mentioning things)
            if row["sender"] != "contact":
                continue

            imp_score = float(row.get("importance_score", 0.2))
            intent    = row.get("intent_label", "casual_chat")
            attn_flag = bool(row.get("attention_gap_flag", False))
            sentiment = float(row.get("sentiment_score", 0.5))

            # ── Embedding position ───────────────────────────────
            emb_pos = id_to_pos.get(row["id"], 0)
            sim     = float(sim_to_arch[emb_pos])

            # ── Text signals ─────────────────────────────────────
            text_sig = extract_text_signals(row.get("content", ""))

            # ── Did 'me' reply adequately within ACK_WINDOW_HRS? ─
            future   = grp.iloc[i + 1:] if i + 1 < len(grp) else pd.DataFrame()
            my_ack   = False
            reply_gap_hrs     = np.nan
            my_reply_sentiment = np.nan
            my_reply_importance = np.nan

            if not future.empty:
                my_replies = future[future["sender"] == "me"]
                if not my_replies.empty:
                    first_reply = my_replies.iloc[0]
                    delta = (first_reply["timestamp"] - row["timestamp"]).total_seconds() / 3600
                    reply_gap_hrs      = delta
                    my_reply_sentiment  = float(first_reply.get("sentiment_score",  0.5))
                    my_reply_importance = float(first_reply.get("importance_score", 0.2))
                    my_ack = (delta <= ACK_WINDOW_HRS) and (my_reply_sentiment > 0.0)

            # ── How many contact messages since last 'me' reply ──
            msgs_ignored = 0
            for j in range(i - 1, -1, -1):
                prev = grp.iloc[j]
                if prev["sender"] == "me":
                    break
                msgs_ignored += 1

            # ── Rolling importance of thread ─────────────────────
            thread_imp_avg = float(rolling_imp.iloc[i]) if i < len(rolling_imp) else 0.3

            # ── Time features ────────────────────────────────────
            hour_of_day = row["timestamp"].hour
            day_of_week = row["timestamp"].dayofweek
            is_weekend  = int(day_of_week >= 5)

            # ── LABEL (weak supervision) ─────────────────────────
            is_important = (
                imp_score > IMPORTANCE_THRESH
                and intent in IMPORTANT_INTENTS
            )
            label = int(
                attn_flag
                or (is_important and not my_ack)
                or (text_sig["has_urgency_keyword"] and not my_ack and imp_score > 0.4)
                or (text_sig["has_emotional_keyword"] and not my_ack and imp_score > 0.35)
            )

            records.append({
                # Identifiers
                "msg_id":                   row["id"],
                "contact_id":               cid,
                "timestamp":                row["timestamp"],
                "content":                  row.get("content", "")[:200],
                # Message features
                "importance_score":         imp_score,
                "intent_label":             intent,
                "sentiment_score":          sentiment,
                "attention_gap_flag":       int(attn_flag),
                "sim_to_important_arch":    sim,
                "hour_of_day":              hour_of_day,
                "day_of_week":              day_of_week,
                "is_weekend":               is_weekend,
                # Reply features
                "reply_gap_hrs":            reply_gap_hrs if not np.isnan(reply_gap_hrs) else 999.0,
                "my_reply_sentiment":       my_reply_sentiment if not np.isnan(my_reply_sentiment) else -0.5,
                "my_reply_importance":      my_reply_importance if not np.isnan(my_reply_importance) else 0.0,
                "was_acknowledged":         int(my_ack),
                "msgs_ignored_streak":      msgs_ignored,
                # Text signals
                **text_sig,
                # Thread context
                "thread_importance_avg":    thread_imp_avg,
                # Contact features
                "contact_health":           c_health,
                "contact_resp_ratio":       c_resp,
                "contact_ghosted":          c_ghosted,
                "contact_churn":            c_churn,
                "contact_decay":            c_decay,
                "contact_days_since":       c_days_since,
                "contact_sentiment_avg":    c_sentiment,
                # Label
                "label":                    label,
            })

    df = pd.DataFrame(records)
    pos = df["label"].sum()
    print(f"  Contact messages  : {len(df):,}")
    print(f"  Missed mentions   : {pos:,}  ({100 * pos / len(df):.1f}%)")
    return df


# ─────────────────────────────────────────────────────────────
# 5. Intent encoding
# ─────────────────────────────────────────────────────────────

INTENT_COLS = [
    "intent_casual_chat", "intent_question", "intent_request",
    "intent_emotional_support", "intent_planning", "intent_followup",
    "intent_complaint", "intent_update", "intent_greeting", "intent_farewell",
]

FEATURE_COLS = [
    # Message signals
    "importance_score", "sentiment_score", "attention_gap_flag",
    "sim_to_important_arch",
    "hour_of_day", "day_of_week", "is_weekend",
    # Reply signals
    "reply_gap_hrs", "my_reply_sentiment", "my_reply_importance",
    "was_acknowledged", "msgs_ignored_streak",
    # Text signals
    "has_urgency_keyword", "has_emotional_keyword", "has_action_request",
    "question_count", "word_count", "exclamation_count", "caps_ratio",
    "has_name_mention",
    # Thread context
    "thread_importance_avg",
    # Contact signals
    "contact_health", "contact_resp_ratio", "contact_ghosted",
    "contact_churn", "contact_decay", "contact_days_since",
    "contact_sentiment_avg",
] + INTENT_COLS


def encode_intents(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in INTENT_COLS:
        df[col] = (df["intent_label"] == col.replace("intent_", "")).astype(int)
    return df


# ─────────────────────────────────────────────────────────────
# 6. Training — XGBoost with GPU
# ─────────────────────────────────────────────────────────────

def train(df: pd.DataFrame, device: torch.device):
    import xgboost as xgb

    print("\nTraining XGBoost (Missed Important Mentions) …")

    df = encode_intents(df)
    X  = df[FEATURE_COLS].fillna(0).values.astype(np.float32)
    y  = df["label"].values.astype(np.float32)

    # GPU device string for XGBoost
    if device.type == "cuda":
        xgb_device = "cuda"
        print("  XGBoost device: cuda (GPU histogram)")
    else:
        xgb_device = "cpu"
        print("  XGBoost device: cpu")

    pos_ratio   = (y == 0).sum() / max((y == 1).sum(), 1)
    print(f"  scale_pos_weight: {pos_ratio:.2f}")

    params = dict(
        objective         = "binary:logistic",
        eval_metric       = "aucpr",
        learning_rate     = 0.05,
        max_depth         = 7,
        n_estimators      = 600,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 10,
        scale_pos_weight  = pos_ratio,
        device            = xgb_device,
        random_state      = RANDOM_STATE,
        early_stopping_rounds = 50,
        verbosity         = 0,
    )

    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros(len(y), dtype=np.float32)
    fold_aucs = []
    best_iters = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        model = xgb.XGBClassifier(**params)
        model.fit(
            X[tr_idx], y[tr_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
        preds = model.predict_proba(X[val_idx])[:, 1]
        oof_probs[val_idx] = preds
        auc = roc_auc_score(y[val_idx], preds)
        fold_aucs.append(auc)
        best_iters.append(model.best_iteration)
        print(f"  Fold {fold+1}/5  AUC={auc:.4f}  best_iter={model.best_iteration}")

    print(f"\n  Mean CV AUC : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    # Retrain on full data with averaged best iteration
    print("  Retraining on full dataset …")
    final_params = params.copy()
    final_params.pop("early_stopping_rounds")
    final_params["n_estimators"] = max(1, int(np.mean(best_iters)))
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X, y, verbose=False)

    joblib.dump(final_model, MODEL_PATH)
    print(f"  Model saved → {MODEL_PATH}")

    df = df.copy()
    df["mention_importance_prob"] = oof_probs
    df["is_missed_mention"]       = (oof_probs >= 0.5).astype(int)
    return final_model, df


# ─────────────────────────────────────────────────────────────
# 7. Precision threshold tuning
# ─────────────────────────────────────────────────────────────

def tune_threshold(df: pd.DataFrame) -> float:
    """
    Find the threshold that maximises F1 score (balancing precision & recall).
    Also prints a table of precision/recall at common thresholds.
    """
    y_true = df["label"].values
    y_prob = df["mention_importance_prob"].values

    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1[:-1])
    best_thr = float(thresholds[best_idx])

    print(f"\n── Threshold Tuning ────────────────────────────────────")
    print(f"  Best F1 threshold : {best_thr:.3f}  "
          f"(P={prec[best_idx]:.3f}  R={rec[best_idx]:.3f}  F1={f1[best_idx]:.3f})")

    print(f"\n  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        idx = np.searchsorted(thresholds, t)
        idx = min(idx, len(prec) - 2)
        print(f"  {t:>10.2f}  {prec[idx]:>10.3f}  {rec[idx]:>8.3f}  {f1[idx]:>8.3f}")

    return best_thr


# ─────────────────────────────────────────────────────────────
# 8. Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate(df: pd.DataFrame, threshold: float = 0.5):
    print(f"\n── Evaluation (threshold = {threshold:.2f}) ─────────────────")
    y_true = df["label"].values
    y_prob = df["mention_importance_prob"].values
    y_pred = (y_prob >= threshold).astype(int)

    print(classification_report(y_true, y_pred,
                                 target_names=["Seen/Normal", "Missed Mention"]))
    print(f"  ROC-AUC : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"  PR-AUC  : {average_precision_score(y_true, y_prob):.4f}")

    # Top contacts with most missed mentions
    summary = (
        df.groupby("contact_id")
        .agg(
            total_contact_msgs  = ("label", "count"),
            missed_count        = ("is_missed_mention", "sum"),
            mean_prob           = ("mention_importance_prob", "mean"),
            max_importance      = ("importance_score", "max"),
        )
        .assign(missed_rate = lambda d: d["missed_count"] / d["total_contact_msgs"])
        .sort_values("missed_count", ascending=False)
        .head(10)
        .reset_index()
    )
    print("\n  Top 10 contacts with most missed mentions:")
    print(summary.to_string(index=False))
    return summary


# ─────────────────────────────────────────────────────────────
# 9. SHAP explainability
# ─────────────────────────────────────────────────────────────

def explain_model(model, df: pd.DataFrame):
    """Generate SHAP summary plot for feature explainability."""
    try:
        import shap
        print("\nGenerating SHAP explanation …")
        df_enc = encode_intents(df)
        X      = df_enc[FEATURE_COLS].fillna(0).values.astype(np.float32)
        sample = X[:2000] if len(X) > 2000 else X

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values, sample,
            feature_names=FEATURE_COLS,
            show=False, max_display=20,
        )
        plt.title("SHAP Feature Importance — Missed Important Mentions")
        plt.tight_layout()
        plt.savefig("mentions_shap.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  SHAP plot saved → mentions_shap.png")
    except ImportError:
        print("  shap not installed — skipping SHAP plot  (pip install shap)")


# ─────────────────────────────────────────────────────────────
# 10. Visualisations
# ─────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, model, threshold: float = 0.5):
    import xgboost as xgb

    print("\nGenerating diagnostic plots …")
    y_true = df["label"].values
    y_prob = df["mention_importance_prob"].values

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Model 3 — Missed Important Mentions Detection (XGBoost + GPU Embeddings)",
        fontsize=13, fontweight="bold",
    )

    # ── 1. ROC Curve ─────────────────────────────────────────
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.fill_between(fpr, tpr, alpha=0.10, color="#2980b9")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    # ── 2. Precision–Recall Curve ─────────────────────────────
    ax = axes[0, 1]
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ax.plot(rec, prec, color="#27ae60", lw=2, label=f"AP = {ap:.3f}")
    ax.fill_between(rec, prec, alpha=0.10, color="#27ae60")
    ax.axvline(rec[np.searchsorted(-thr, -threshold)], color="red",
               linestyle="--", alpha=0.6, label=f"t={threshold:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()

    # ── 3. Score distributions ────────────────────────────────
    ax = axes[0, 2]
    ax.hist(y_prob[y_true == 0], bins=50, alpha=0.65, color="#3498db", label="Normal")
    ax.hist(y_prob[y_true == 1], bins=50, alpha=0.65, color="#e74c3c", label="Missed mention")
    ax.axvline(threshold, color="black", linestyle="--", alpha=0.7,
               label=f"Threshold={threshold:.2f}")
    ax.set_xlabel("mention_importance_prob")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.legend()

    # ── 4. Feature importances ────────────────────────────────
    ax = axes[1, 0]
    imp = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLS,
    ).nlargest(15).sort_values()
    imp.plot(kind="barh", ax=ax, color="#8e44ad")
    ax.set_title("Top 15 Feature Importances")
    ax.set_xlabel("Gain")

    # ── 5. Missed rate by intent ──────────────────────────────
    ax = axes[1, 1]
    intent_rate = (
        df.groupby("intent_label")["is_missed_mention"]
        .mean()
        .sort_values(ascending=False)
    )
    colors = ["#e74c3c" if i in IMPORTANT_INTENTS else "#95a5a6"
              for i in intent_rate.index]
    intent_rate.plot(kind="bar", ax=ax, color=colors)
    ax.set_xlabel("Intent Label")
    ax.set_ylabel("Missed Rate")
    ax.set_title("Missed Rate by Intent\n(red = high-signal intents)")
    ax.tick_params(axis="x", rotation=35)

    # ── 6. Missed rate vs importance score ────────────────────
    ax = axes[1, 2]
    bins = pd.cut(df["importance_score"], bins=8)
    imp_missed = df.groupby(bins, observed=False)["is_missed_mention"].mean()
    imp_missed.plot(kind="bar", ax=ax, color="#e67e22")
    ax.set_xlabel("Importance Score (binned)")
    ax.set_ylabel("Missed Rate")
    ax.set_title("Missed Rate by Message Importance")
    ax.tick_params(axis="x", rotation=40)

    plt.tight_layout()
    plt.savefig("mentions_detection_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Plot saved → mentions_detection_results.png")


# ─────────────────────────────────────────────────────────────
# 11. Export
# ─────────────────────────────────────────────────────────────

def export_results(df: pd.DataFrame, threshold: float = 0.5):
    df = df.copy()
    df["timestamp"] = df["timestamp"].astype(str)

    missed = df[df["is_missed_mention"] == 1]
    out_cols = [
        "msg_id", "contact_id", "timestamp", "content",
        "intent_label", "importance_score", "sentiment_score",
        "mention_importance_prob", "reply_gap_hrs",
        "has_urgency_keyword", "has_emotional_keyword",
        "contact_health", "contact_ghosted", "label",
    ]
    with open("missed_mentions.json", "w") as f:
        json.dump(missed[out_cols].to_dict(orient="records"), f,
                  indent=2, default=str)

    summary = (
        df.groupby("contact_id")
        .agg(
            total_msgs    = ("label", "count"),
            missed_count  = ("is_missed_mention", "sum"),
            max_prob      = ("mention_importance_prob", "max"),
            mean_prob     = ("mention_importance_prob", "mean"),
        )
        .assign(missed_rate=lambda d: d["missed_count"] / d["total_msgs"])
        .reset_index()
        .sort_values("missed_count", ascending=False)
    )
    with open("mentions_contact_summary.json", "w") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2, default=str)

    print(f"\n  Missed mentions    → missed_mentions.json             ({len(missed):,} rows)")
    print(f"  Contact summary    → mentions_contact_summary.json   ({len(summary)} rows)")


# ─────────────────────────────────────────────────────────────
# 12. Real-time inference
# ─────────────────────────────────────────────────────────────

def load_model():
    return joblib.load(MODEL_PATH)


def predict_mention(
    content: str,
    intent_label: str,
    importance_score: float,
    sentiment_score: float,
    attention_gap_flag: bool,
    reply_gap_hrs: float,
    my_reply_sentiment: float,
    my_reply_importance: float,
    msgs_ignored_streak: int,
    contact_health: float,
    contact_resp_ratio: float,
    contact_ghosted: bool,
    contact_churn: float,
    contact_decay: float,
    contact_days_since: int,
    contact_sentiment_avg: float,
    thread_importance_avg: float = 0.3,
    device: torch.device = None,
    model=None,
    encoder=None,
) -> dict:
    """
    Score a single contact message in real time.
    Returns mention_importance_prob and is_missed_mention flag.
    """
    if model is None:
        model = load_model()
    if device is None:
        device = get_device()

    # Encode content on GPU
    if encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(ENCODER_NAME, device=str(device))
        except ImportError:
            encoder = None

    if encoder is not None:
        emb = encoder.encode([content], normalize_embeddings=True)[0]
        # Proxy: use mean absolute value as sim_to_arch (no archetype in RT mode)
        sim = float(np.mean(np.abs(emb)))
    else:
        sim = importance_score  # fallback

    text_sig = extract_text_signals(content)

    row = {
        "importance_score":         importance_score,
        "sentiment_score":          sentiment_score,
        "attention_gap_flag":       int(attention_gap_flag),
        "sim_to_important_arch":    sim,
        "hour_of_day":              pd.Timestamp.now().hour,
        "day_of_week":              pd.Timestamp.now().dayofweek,
        "is_weekend":               int(pd.Timestamp.now().dayofweek >= 5),
        "reply_gap_hrs":            reply_gap_hrs,
        "my_reply_sentiment":       my_reply_sentiment,
        "my_reply_importance":      my_reply_importance,
        "was_acknowledged":         int(reply_gap_hrs <= ACK_WINDOW_HRS and my_reply_sentiment > 0),
        "msgs_ignored_streak":      msgs_ignored_streak,
        **text_sig,
        "thread_importance_avg":    thread_importance_avg,
        "contact_health":           contact_health,
        "contact_resp_ratio":       contact_resp_ratio,
        "contact_ghosted":          int(contact_ghosted),
        "contact_churn":            contact_churn,
        "contact_decay":            contact_decay,
        "contact_days_since":       contact_days_since,
        "contact_sentiment_avg":    contact_sentiment_avg,
    }
    for col in INTENT_COLS:
        row[col] = int(intent_label == col.replace("intent_", ""))

    X    = np.array([[row[c] for c in FEATURE_COLS]], dtype=np.float32)
    prob = float(model.predict_proba(X)[0, 1])

    # Urgency tier
    if prob >= 0.80:
        urgency = "CRITICAL"
    elif prob >= 0.60:
        urgency = "HIGH"
    elif prob >= 0.40:
        urgency = "MEDIUM"
    else:
        urgency = "low"

    return {
        "content_preview":          content[:80],
        "intent_label":             intent_label,
        "importance_score":         importance_score,
        "mention_importance_prob":  round(prob, 4),
        "is_missed_mention":        prob >= 0.5,
        "urgency":                  urgency,
        "signals": {
            "urgency_keyword":   bool(text_sig["has_urgency_keyword"]),
            "emotional_keyword": bool(text_sig["has_emotional_keyword"]),
            "action_request":    bool(text_sig["has_action_request"]),
            "question_marks":    text_sig["question_count"],
        },
    }


def predict_thread_mentions(
    contact_messages: list,
    device: torch.device = None,
    model=None,
) -> pd.DataFrame:
    """
    Score an entire thread for missed important mentions.
    contact_messages: list of dicts with keys:
        id, timestamp, sender, content, intent_label,
        importance_score, sentiment_score, attention_gap_flag
    Returns DataFrame with is_missed_mention per contact message.
    """
    if model is None:
        model = load_model()
    if device is None:
        device = get_device()

    df = pd.DataFrame(contact_messages)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    contact_msgs = df[df["sender"] == "contact"].copy()
    if contact_msgs.empty:
        return pd.DataFrame()

    # Batch encode on GPU
    try:
        from sentence_transformers import SentenceTransformer
        enc      = SentenceTransformer(ENCODER_NAME, device=str(device))
        all_embs = enc.encode(
            df["content"].fillna("").tolist(),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        c_embs  = all_embs[contact_msgs.index]
        sim_vals = np.mean(np.abs(c_embs), axis=1)
    except ImportError:
        sim_vals = contact_msgs["importance_score"].fillna(0.3).values

    records = []
    for i, (orig_i, row) in enumerate(contact_msgs.iterrows()):
        # Look ahead for my reply
        future      = df.iloc[orig_i + 1:]
        my_replies  = future[future["sender"] == "me"] if not future.empty else pd.DataFrame()

        if not my_replies.empty:
            first      = my_replies.iloc[0]
            r_gap      = (first["timestamp"] - row["timestamp"]).total_seconds() / 3600
            r_sent     = float(first.get("sentiment_score", 0.5))
            r_imp      = float(first.get("importance_score", 0.2))
            acked      = r_gap <= ACK_WINDOW_HRS and r_sent > 0.0
        else:
            r_gap, r_sent, r_imp, acked = 999.0, -0.5, 0.0, False

        msgs_streak = sum(1 for _, r in df.iloc[:orig_i].iloc[::-1].iterrows()
                          if r["sender"] != "me")
        text_sig    = extract_text_signals(row.get("content", ""))

        row_feat = {
            "importance_score":         float(row.get("importance_score", 0.2)),
            "sentiment_score":          float(row.get("sentiment_score", 0.5)),
            "attention_gap_flag":       int(row.get("attention_gap_flag", False)),
            "sim_to_important_arch":    float(sim_vals[i]),
            "hour_of_day":              row["timestamp"].hour,
            "day_of_week":              row["timestamp"].dayofweek,
            "is_weekend":               int(row["timestamp"].dayofweek >= 5),
            "reply_gap_hrs":            r_gap,
            "my_reply_sentiment":       r_sent,
            "my_reply_importance":      r_imp,
            "was_acknowledged":         int(acked),
            "msgs_ignored_streak":      msgs_streak,
            **text_sig,
            "thread_importance_avg":    df["importance_score"].iloc[max(0, orig_i-10):orig_i].mean(),
            "contact_health":           0.5,
            "contact_resp_ratio":       0.5,
            "contact_ghosted":          0,
            "contact_churn":            0.3,
            "contact_decay":            0.1,
            "contact_days_since":       30,
            "contact_sentiment_avg":    0.5,
        }
        intent = row.get("intent_label", "casual_chat")
        for col in INTENT_COLS:
            row_feat[col] = int(intent == col.replace("intent_", ""))

        X    = np.array([[row_feat[c] for c in FEATURE_COLS]], dtype=np.float32)
        prob = float(model.predict_proba(X)[0, 1])

        records.append({
            "msg_id":                   row.get("id"),
            "timestamp":                str(row["timestamp"]),
            "content":                  row.get("content", "")[:100],
            "intent_label":             intent,
            "importance_score":         row_feat["importance_score"],
            "mention_importance_prob":  round(prob, 4),
            "is_missed_mention":        prob >= 0.5,
            "reply_gap_hrs":            round(r_gap, 1),
            "was_acknowledged":         int(acked),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Model 3: Missed Important Mentions Detection")
    print("=" * 60)

    device           = get_device()
    contacts, messages = load_data()
    embeddings         = encode_messages(messages, device)
    df                 = build_features(messages, contacts, embeddings)
    model, df          = train(df, device)
    threshold          = tune_threshold(df)

    # Apply best threshold
    df["is_missed_mention"] = (df["mention_importance_prob"] >= threshold).astype(int)

    evaluate(df, threshold)
    explain_model(model, df)
    plot_results(df, model, threshold)
    export_results(df, threshold)

    # ── Real-time inference demo ──────────────────────────────
    print("\n── Real-time Inference Demo ─────────────────────────────")
    test_cases = [
        # (content, intent, imp, sent, attn, reply_gap, my_sent, my_imp, streak,
        #  c_health, c_resp, c_ghost, c_churn, c_decay, c_days, c_sent_avg, desc)
        (
            "Hey I really need help with something urgent, can you call me?",
            "request", 0.88, 0.3, True,  72.0, -0.2, 0.1, 3,
            0.4, 0.3, False, 0.5, 0.3, 10, 0.5,
        ),
        (
            "sounds good see you later!",
            "casual_chat", 0.10, 0.9, False, 2.0,  0.8, 0.2, 0,
            0.8, 0.7, False, 0.1, 0.05, 2, 0.7,
        ),
        (
            "I've been feeling really overwhelmed lately and I'm struggling",
            "emotional_support", 0.80, 0.1, True,  96.0, -0.5, 0.0, 5,
            0.3, 0.2, False, 0.6, 0.4, 25, 0.3,
        ),
        (
            "Did you get my message about the deadline? It's tomorrow!",
            "question", 0.92, 0.2, True,  48.0,  0.0, 0.1, 4,
            0.45, 0.35, False, 0.55, 0.25, 7, 0.4,
        ),
        (
            "ok lol",
            "casual_chat", 0.05, 0.8, False, 1.0,  0.9, 0.1, 0,
            0.9, 0.8, False, 0.05, 0.02, 1, 0.8,
        ),
    ]

    descriptions = [
        "Urgent request, no real reply (72h gap)",
        "Casual reply, acknowledged quickly",
        "Emotional distress, ignored for 96h",
        "Important question, deadline tomorrow",
        "Low-importance casual message",
    ]

    print(f"\n  {'Content':<48} {'Prob':>6}  {'Urgency':<10}  Missed?")
    print("  " + "-" * 78)
    for args, desc in zip(test_cases, descriptions):
        res = predict_mention(*args, device=device, model=model)
        flag = "⚠  YES" if res["is_missed_mention"] else "   no"
        print(f"  {res['content_preview'][:48]:<48} "
              f"{res['mention_importance_prob']:>6.3f}  "
              f"{res['urgency']:<10}  {flag}")

    print("\nDone ✓")


if __name__ == "__main__":
    main()