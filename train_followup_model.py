"""
Model 2: Forgotten Followup Detection
=======================================
Detects messages that required a followup/response but were never acted on.

Architecture:
  - Sentence-Transformer (GPU) encodes message content → 384-d embeddings
  - Rule-based signal extraction (intent, importance, attention_gap_flag)
  - LightGBM classifier trained on combined features
  - Outputs: followup_needed (bool) + followup_probability (float)

Signal sources:
  - intent_label         : "followup", "question", "request" = high signal
  - attention_gap_flag   : pre-flagged in synthetic data
  - importance_score     : high = more likely needs followup
  - gap since message    : longer gap without reply = more likely forgotten
  - sender pattern       : "me" sent something, "contact" never replied
  - embedding similarity : detect semantically similar "I'll get back to you" non-responses

Pipeline:
  1. Load messages
  2. GPU-encode content with sentence-transformers
  3. Engineer followup features per message
  4. Label: positive = attention_gap_flag=True OR (intent=followup/request/question AND no reply within N hours)
  5. Train LightGBM (GPU tree method)
  6. Evaluate + visualise
  7. Export flagged forgotten followups

Install deps:
    pip install pandas numpy scikit-learn lightgbm sentence-transformers torch tqdm matplotlib seaborn joblib
"""

import json
import os
import warnings
from dotenv import load_dotenv

load_dotenv()  # loads MONGO_URL (and other vars) from .env
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
DATA_DIR    = Path(".")
MODEL_DIR   = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH       = MODEL_DIR / "followup_lgbm.joblib"
SCALER_PATH      = MODEL_DIR / "followup_scaler.joblib"
ENCODER_NAME     = "sentence-transformers/all-MiniLM-L6-v2"   # 384-d, fast
EMBEDDING_DIM    = 384
BATCH_SIZE       = 256          # GPU batch size for encoding
REPLY_WINDOW_HRS = 48           # hours within which a reply "counts"
RANDOM_STATE     = 42

# Intents that strongly signal a followup is expected
FOLLOWUP_INTENTS = {"question", "request", "followup", "planning"}

# ─────────────────────────────────────────────────────────────
# Device setup
# ─────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}  ({props.total_memory // 1024**2} MB VRAM)")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("  Device: Apple MPS (Metal)")
    else:
        dev = torch.device("cpu")
        print("  Device: CPU (no GPU found — encoding will be slower)")
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
            doc.pop("_id", None)   # drop ObjectId; not needed for training
        contacts = pd.DataFrame(raw_contacts)

        # Normalise the health_score field name (stored as health_score or healthScore)
        if "healthScore" in contacts.columns and "health_score" not in contacts.columns:
            contacts.rename(columns={"healthScore": "health_score"}, inplace=True)

        # ── Messages ──────────────────────────────────────────────
        raw_messages = list(db["messages"].find({}))
        for doc in raw_messages:
            doc.pop("_id", None)
            # Ensure every message has a unique 'id' field
            if "id" not in doc:
                doc["id"] = str(doc.get("message_id", id(doc)))
            # Normalise sender: app stores the user's own messages as "me"
            # (already correct if the pipeline set it up that way)
        messages = pd.DataFrame(raw_messages)

        client.close()
        print(f"  Source: MongoDB ({mongo_url[:40]}…)")
    else:
        print("MONGO_URL not set — falling back to local JSON files …")
        with open(data_dir / "contacts.json", encoding="utf-8") as f:
            contacts = pd.DataFrame(json.load(f))
        with open(data_dir / "messages.json", encoding="utf-8") as f:
            messages = pd.DataFrame(json.load(f))

    messages["timestamp"] = pd.to_datetime(messages["timestamp"], utc=True, errors="coerce")
    messages.sort_values(["contact_id", "timestamp"], inplace=True)
    messages.reset_index(drop=True, inplace=True)

    print(f"  {len(contacts)} contacts | {len(messages):,} messages")
    return contacts, messages


# ─────────────────────────────────────────────────────────────
# 2. GPU Embedding
# ─────────────────────────────────────────────────────────────

def encode_messages(messages: pd.DataFrame, device: torch.device) -> np.ndarray:
    """
    Encode all message content with sentence-transformers on GPU.
    Returns float32 array of shape (N, EMBEDDING_DIM).
    Falls back to pre-computed embeddings stored in messages.json if
    sentence-transformers is not available.
    """
    try:
        from sentence_transformers import SentenceTransformer
        print(f"\nEncoding {len(messages):,} messages with {ENCODER_NAME} on {device} …")
        model = SentenceTransformer(ENCODER_NAME, device=str(device))
        texts = messages["content"].fillna("").tolist()
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    except ImportError:
        print("  sentence-transformers not available — using pre-computed embeddings from JSON")
        return _load_precomputed_embeddings(messages)


def _load_precomputed_embeddings(messages: pd.DataFrame) -> np.ndarray:
    """Parse embedding vectors stored as JSON strings in messages DataFrame."""
    embs = []
    for emb in tqdm(messages["embedding"], desc="Parsing embeddings"):
        if isinstance(emb, str):
            embs.append(json.loads(emb))
        elif isinstance(emb, list):
            embs.append(emb)
        else:
            embs.append([0.0] * EMBEDDING_DIM)
    arr = np.array(embs, dtype=np.float32)
    # L2-normalise
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr / norms


# ─────────────────────────────────────────────────────────────
# 3. Feature Engineering
# ─────────────────────────────────────────────────────────────

def build_followup_features(
    messages: pd.DataFrame,
    contacts: pd.DataFrame,
    embeddings: np.ndarray,
) -> pd.DataFrame:
    """
    For every message sent by 'me', determine:
      - Was a reply received within REPLY_WINDOW_HRS?
      - Is it a high-signal intent?
      - Has it been flagged as attention_gap?
      - How similar is the embedding to known "no-reply" patterns?

    Label construction (weak supervision):
      positive (forgotten followup) =
          attention_gap_flag == True
          OR (intent in FOLLOWUP_INTENTS AND no reply within 48h AND importance > 0.5)
    """
    print("\nEngineering followup features …")

    msg_emb = embeddings   # (N, 384)

    # --- Archetype embeddings for "unanswered" patterns (mean of high-importance, no-reply msgs)
    gap_mask = messages["attention_gap_flag"].fillna(False).astype(bool)
    if gap_mask.sum() > 10:
        archetype_emb = msg_emb[gap_mask].mean(axis=0, keepdims=True)   # (1, 384)
    else:
        archetype_emb = msg_emb[:100].mean(axis=0, keepdims=True)

    # Cosine similarity to archetype (high = similar to known forgotten msgs)
    sim_to_archetype = (msg_emb @ archetype_emb.T).squeeze()            # (N,)

    records = []

    contact_lookup = contacts.set_index("contact_id")

    for cid, grp in tqdm(messages.groupby("contact_id"), desc="Contacts"):
        grp      = grp.sort_values("timestamp").reset_index(drop=True)
        grp_idx  = grp.index.tolist()           # positions in grp
        orig_idx = grp["id"].values             # original msg ids

        # Contact-level features
        c_row = contact_lookup.loc[cid] if cid in contact_lookup.index else {}
        c_health    = float(c_row.get("health_score", 0.5))
        c_resp      = float(c_row.get("response_ratio", 0.5))
        c_ghosted   = bool(c_row.get("is_ghosted", False))
        c_churn     = float(c_row.get("churn_probability", 0.3))
        c_decay     = float(c_row.get("engagement_decay_rate", 0.1))

        for i, (_, row) in enumerate(grp.iterrows()):
            # Only consider messages sent BY ME (I'm the one who might be forgotten)
            if row["sender"] != "me":
                continue

            msg_pos_in_full = messages[messages["id"] == row["id"]].index
            emb_pos = msg_pos_in_full[0] if len(msg_pos_in_full) > 0 else 0

            intent     = row.get("intent_label", "casual_chat")
            imp_score  = float(row.get("importance_score", 0.2))
            attn_flag  = bool(row.get("attention_gap_flag", False))

            # ── Was there a reply within REPLY_WINDOW_HRS? ──────────
            future_msgs = grp.iloc[i + 1:] if i + 1 < len(grp) else pd.DataFrame()
            replied_in_window = False
            time_to_reply_hrs = np.nan

            if not future_msgs.empty:
                contact_replies = future_msgs[future_msgs["sender"] == "contact"]
                if not contact_replies.empty:
                    first_reply   = contact_replies.iloc[0]
                    delta_hrs     = (
                        first_reply["timestamp"] - row["timestamp"]
                    ).total_seconds() / 3600
                    time_to_reply_hrs = delta_hrs
                    replied_in_window = delta_hrs <= REPLY_WINDOW_HRS

            # ── Thread position features ─────────────────────────
            msgs_since_last_contact_reply = 0
            for j in range(i - 1, -1, -1):
                if grp.iloc[j]["sender"] == "contact":
                    break
                msgs_since_last_contact_reply += 1

            # ── Time features ────────────────────────────────────
            hour_of_day = row["timestamp"].hour
            day_of_week = row["timestamp"].dayofweek
            is_weekend  = int(day_of_week >= 5)

            # ── Embedding similarity to "forgotten" archetype ────
            sim = float(sim_to_archetype[emb_pos])

            # ── Rolling reply rate for this contact ─────────────
            past = grp.iloc[:i]
            my_msgs    = (past["sender"] == "me").sum()
            their_repl = (past["sender"] == "contact").sum()
            roll_reply_rate = their_repl / (my_msgs + 1e-6)

            # ── LABEL ─────────────────────────────────────────────
            # Weak supervision from synthetic ground truth signals
            label = int(
                attn_flag
                or (
                    intent in FOLLOWUP_INTENTS
                    and imp_score > 0.50
                    and not replied_in_window
                )
            )

            records.append({
                # identifiers
                "msg_id":                   row["id"],
                "contact_id":               cid,
                "timestamp":                row["timestamp"],
                # message-level features
                "intent_label":             intent,
                "importance_score":         imp_score,
                "attention_gap_flag":       int(attn_flag),
                "sentiment_score":          float(row.get("sentiment_score", 0.5)),
                "hour_of_day":              hour_of_day,
                "day_of_week":              day_of_week,
                "is_weekend":               is_weekend,
                "sim_to_forgotten_arch":    sim,
                "msgs_since_contact_reply": msgs_since_last_contact_reply,
                "roll_reply_rate":          roll_reply_rate,
                "time_to_reply_hrs":        time_to_reply_hrs,
                "replied_in_window":        int(replied_in_window),
                # contact-level features
                "contact_health":           c_health,
                "contact_resp_ratio":       c_resp,
                "contact_ghosted":          int(c_ghosted),
                "contact_churn":            c_churn,
                "contact_decay":            c_decay,
                # label
                "label":                    label,
            })

    df = pd.DataFrame(records)
    pos = df["label"].sum()
    print(f"  Messages from 'me': {len(df):,}  |  Forgotten followups: {pos:,}  ({100*pos/len(df):.1f}%)")
    return df


# ─────────────────────────────────────────────────────────────
# 4. Intent one-hot encoding
# ─────────────────────────────────────────────────────────────

INTENT_COLS = [
    "intent_casual_chat", "intent_question", "intent_request",
    "intent_emotional_support", "intent_planning", "intent_followup",
    "intent_complaint", "intent_update", "intent_greeting", "intent_farewell",
]


def encode_intents(df: pd.DataFrame) -> pd.DataFrame:
    for col in INTENT_COLS:
        intent_name = col.replace("intent_", "")
        df[col] = (df["intent_label"] == intent_name).astype(int)
    return df


FEATURE_COLS = [
    "importance_score", "attention_gap_flag", "sentiment_score",
    "hour_of_day", "day_of_week", "is_weekend",
    "sim_to_forgotten_arch", "msgs_since_contact_reply",
    "roll_reply_rate",
    "contact_health", "contact_resp_ratio", "contact_ghosted",
    "contact_churn", "contact_decay",
] + INTENT_COLS


# ─────────────────────────────────────────────────────────────
# 5. Training
# ─────────────────────────────────────────────────────────────

def train(df: pd.DataFrame, device: torch.device):
    import lightgbm as lgb

    print("\nTraining LightGBM (Forgotten Followup Classifier) …")

    df = encode_intents(df.copy())
    X  = df[FEATURE_COLS].fillna(0).values
    y  = df["label"].values

    # Determine GPU device type for LightGBM
    if device.type == "cuda":
        lgb_device = "gpu"
        print("  LightGBM tree method: gpu")
    else:
        lgb_device = "cpu"
        print("  LightGBM tree method: cpu")

    # Class weights
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    print(f"  Positive class weight: {pos_weight:.2f}")

    # 5-fold cross-validation for reliable AUC estimate
    skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros(len(y), dtype=np.float32)

    params = {
        "objective":        "binary",
        "metric":           "auc",
        "learning_rate":    0.05,
        "num_leaves":       63,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "scale_pos_weight": pos_weight,
        "device":           lgb_device,
        "seed":             RANDOM_STATE,
        "verbose":          -1,
    }
    if lgb_device == "gpu":
        params["gpu_platform_id"] = 0
        params["gpu_device_id"]   = 0

    fold_aucs = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        preds = model.predict(X_val)
        oof_probs[val_idx] = preds
        auc = roc_auc_score(y_val, preds)
        fold_aucs.append(auc)
        print(f"  Fold {fold+1}/5  AUC = {auc:.4f}")

    print(f"\n  Mean CV AUC : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    # Retrain on full data
    print("  Retraining on full dataset …")
    dtrain_full = lgb.Dataset(X, label=y)
    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=int(np.mean([m.best_iteration for m in [model]])),
        callbacks=[lgb.log_evaluation(0)],
    )

    joblib.dump(final_model, MODEL_PATH)
    print(f"  Model saved → {MODEL_PATH}")

    df["followup_probability"] = oof_probs
    df["predicted_forgotten"]  = (oof_probs >= 0.5).astype(int)
    return final_model, df


# ─────────────────────────────────────────────────────────────
# 6. Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate(df: pd.DataFrame):
    print("\n── Evaluation ──────────────────────────────────────────")
    y_true = df["label"].values
    y_prob = df["followup_probability"].values
    y_pred = df["predicted_forgotten"].values

    print(classification_report(y_true, y_pred, target_names=["Normal", "Forgotten FU"]))

    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr  = average_precision_score(y_true, y_prob)
    print(f"  ROC-AUC  : {auc_roc:.4f}")
    print(f"  PR-AUC   : {auc_pr:.4f}")

    # Top forgotten by contact
    contact_summary = (
        df.groupby("contact_id")
        .agg(
            total_sent       =("label", "count"),
            forgotten_count  =("predicted_forgotten", "sum"),
            mean_prob        =("followup_probability", "mean"),
        )
        .assign(forgotten_rate=lambda d: d["forgotten_count"] / d["total_sent"])
        .sort_values("forgotten_count", ascending=False)
        .head(10)
        .reset_index()
    )
    print("\n  Top 10 contacts with most forgotten followups:")
    print(contact_summary.to_string(index=False))
    return auc_roc, auc_pr


# ─────────────────────────────────────────────────────────────
# 7. Visualisations
# ─────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, model):
    import lightgbm as lgb

    print("\nGenerating plots …")
    y_true = df["label"].values
    y_prob = df["followup_probability"].values

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Model 2 — Forgotten Followup Detection (LightGBM + GPU Embeddings)",
                 fontsize=13, fontweight="bold")

    # ── 1. ROC Curve ──────────────────────────────────────────
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    # ── 2. Precision-Recall Curve ─────────────────────────────
    ax = axes[0, 1]
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ax.plot(rec, prec, color="#27ae60", lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()

    # ── 3. Score distribution ─────────────────────────────────
    ax = axes[0, 2]
    ax.hist(y_prob[y_true == 0], bins=40, alpha=0.6, color="#3498db", label="Normal")
    ax.hist(y_prob[y_true == 1], bins=40, alpha=0.6, color="#e74c3c", label="Forgotten FU")
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.6, label="Threshold=0.5")
    ax.set_xlabel("followup_probability")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.legend()

    # ── 4. Feature Importance ─────────────────────────────────
    ax = axes[1, 0]
    imp = pd.Series(model.feature_importance(importance_type="gain"), index=FEATURE_COLS)
    imp = imp.nlargest(15).sort_values()
    imp.plot(kind="barh", ax=ax, color="#9b59b6")
    ax.set_title("Top 15 Feature Importances (Gain)")
    ax.set_xlabel("Gain")

    # ── 5. Forgotten rate by intent ───────────────────────────
    ax = axes[1, 1]
    intent_stats = (
        df.groupby("intent_label")["predicted_forgotten"]
        .mean()
        .sort_values(ascending=False)
    )
    intent_stats.plot(kind="bar", ax=ax, color="#e67e22")
    ax.set_xlabel("Intent Label")
    ax.set_ylabel("Forgotten Rate")
    ax.set_title("Forgotten Rate by Intent")
    ax.tick_params(axis="x", rotation=35)

    # ── 6. Forgotten rate by contact health ───────────────────
    ax = axes[1, 2]
    bins = pd.cut(df["contact_health"], bins=5)
    health_stats = df.groupby(bins, observed=False)["predicted_forgotten"].mean()
    health_stats.plot(kind="bar", ax=ax, color="#1abc9c")
    ax.set_xlabel("Contact Health Score (binned)")
    ax.set_ylabel("Forgotten Rate")
    ax.set_title("Forgotten Rate by Contact Health")
    ax.tick_params(axis="x", rotation=35)

    plt.tight_layout()
    plt.savefig("followup_detection_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Plot saved → followup_detection_results.png")


# ─────────────────────────────────────────────────────────────
# 8. Export
# ─────────────────────────────────────────────────────────────

def export_results(df: pd.DataFrame):
    forgotten = df[df["predicted_forgotten"] == 1].copy()
    forgotten["timestamp"] = forgotten["timestamp"].astype(str)

    out_cols = [
        "msg_id", "contact_id", "timestamp", "intent_label",
        "importance_score", "followup_probability",
        "msgs_since_contact_reply", "contact_health",
        "contact_ghosted", "label",
    ]
    records = forgotten[out_cols].to_dict(orient="records")

    with open("forgotten_followups.json", "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"\n  Flagged followups → forgotten_followups.json  ({len(records):,} rows)")

    # Per-contact summary
    summary = (
        df.groupby("contact_id")
        .agg(
            total_sent      =("label", "count"),
            forgotten_count =("predicted_forgotten", "sum"),
            max_prob        =("followup_probability", "max"),
            mean_prob       =("followup_probability", "mean"),
        )
        .assign(forgotten_rate=lambda d: d["forgotten_count"] / d["total_sent"])
        .reset_index()
        .sort_values("forgotten_count", ascending=False)
    )
    with open("followup_contact_summary.json", "w") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2, default=str)
    print(f"  Contact summary    → followup_contact_summary.json  ({len(summary)} rows)")


# ─────────────────────────────────────────────────────────────
# 9. Real-time inference
# ─────────────────────────────────────────────────────────────

def load_model():
    return joblib.load(MODEL_PATH)


def predict_message(
    content: str,
    intent_label: str,
    importance_score: float,
    attention_gap_flag: bool,
    sentiment_score: float,
    hour_of_day: int,
    day_of_week: int,
    msgs_since_contact_reply: int,
    roll_reply_rate: float,
    contact_health: float,
    contact_resp_ratio: float,
    contact_ghosted: bool,
    contact_churn: float,
    contact_decay: float,
    device: torch.device = None,
    model=None,
) -> dict:
    """
    Score a single message in real time.
    Returns followup_probability and is_forgotten_followup flag.
    """
    if model is None:
        model = load_model()
    if device is None:
        device = get_device()

    # Encode content
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(ENCODER_NAME, device=str(device))
        emb = encoder.encode([content], normalize_embeddings=True)[0]
    except ImportError:
        emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # For real-time use, sim_to_forgotten_arch = mean of embedding (proxy)
    sim = float(np.mean(np.abs(emb)))

    row = {
        "importance_score":         importance_score,
        "attention_gap_flag":       int(attention_gap_flag),
        "sentiment_score":          sentiment_score,
        "hour_of_day":              hour_of_day,
        "day_of_week":              day_of_week,
        "is_weekend":               int(day_of_week >= 5),
        "sim_to_forgotten_arch":    sim,
        "msgs_since_contact_reply": msgs_since_contact_reply,
        "roll_reply_rate":          roll_reply_rate,
        "contact_health":           contact_health,
        "contact_resp_ratio":       contact_resp_ratio,
        "contact_ghosted":          int(contact_ghosted),
        "contact_churn":            contact_churn,
        "contact_decay":            contact_decay,
    }
    for col in INTENT_COLS:
        row[col] = int(intent_label == col.replace("intent_", ""))

    X = np.array([[row[c] for c in FEATURE_COLS]], dtype=np.float32)
    prob = float(model.predict(X)[0])

    return {
        "content_preview":          content[:80],
        "intent_label":             intent_label,
        "importance_score":         importance_score,
        "followup_probability":     round(prob, 4),
        "is_forgotten_followup":    prob >= 0.5,
        "urgency":                  "high" if prob > 0.75 else "medium" if prob > 0.5 else "low",
    }


def predict_thread(contact_messages: list, device: torch.device = None, model=None) -> pd.DataFrame:
    """
    Score an entire message thread (list of dicts with at least:
      id, timestamp, sender, content, intent_label, importance_score,
      attention_gap_flag, sentiment_score)
    Returns DataFrame with followup_probability per 'me' message.
    """
    if model is None:
        model = load_model()
    if device is None:
        device = get_device()

    df = pd.DataFrame(contact_messages)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    my_msgs = df[df["sender"] == "me"].copy()
    if my_msgs.empty:
        return pd.DataFrame()

    # Encode all content at once (GPU batch)
    try:
        from sentence_transformers import SentenceTransformer
        encoder  = SentenceTransformer(ENCODER_NAME, device=str(device))
        all_embs = encoder.encode(df["content"].fillna("").tolist(),
                                  normalize_embeddings=True, show_progress_bar=False)
        my_embs  = all_embs[my_msgs.index]
        sim_vals = np.mean(np.abs(my_embs), axis=1)
    except ImportError:
        sim_vals = np.zeros(len(my_msgs))

    records = []
    for i, (_, row) in enumerate(my_msgs.iterrows()):
        pos = my_msgs.index.get_loc(row.name)
        past = df.iloc[:pos]
        my_c   = (past["sender"] == "me").sum()
        their  = (past["sender"] == "contact").sum()

        msgs_gap    = sum(1 for _, r in df.iloc[pos-1::-1].iterrows()
                         if r["sender"] != "contact") if pos > 0 else 0
        roll_rate   = their / (my_c + 1e-6)

        feature_row = {
            "importance_score":         float(row.get("importance_score", 0.3)),
            "attention_gap_flag":       int(row.get("attention_gap_flag", False)),
            "sentiment_score":          float(row.get("sentiment_score", 0.5)),
            "hour_of_day":              row["timestamp"].hour,
            "day_of_week":              row["timestamp"].dayofweek,
            "is_weekend":               int(row["timestamp"].dayofweek >= 5),
            "sim_to_forgotten_arch":    float(sim_vals[i]),
            "msgs_since_contact_reply": msgs_gap,
            "roll_reply_rate":          roll_rate,
            "contact_health":           0.5,
            "contact_resp_ratio":       0.5,
            "contact_ghosted":          0,
            "contact_churn":            0.3,
            "contact_decay":            0.1,
        }
        intent = row.get("intent_label", "casual_chat")
        for col in INTENT_COLS:
            feature_row[col] = int(intent == col.replace("intent_", ""))

        X    = np.array([[feature_row[c] for c in FEATURE_COLS]], dtype=np.float32)
        prob = float(model.predict(X)[0])
        records.append({
            "msg_id":               row.get("id"),
            "timestamp":            str(row["timestamp"]),
            "content":              row.get("content", "")[:100],
            "intent_label":         intent,
            "followup_probability": round(prob, 4),
            "is_forgotten":         prob >= 0.5,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Model 2: Forgotten Followup Detection")
    print("=" * 60)

    device           = get_device()
    contacts, messages = load_data()
    embeddings         = encode_messages(messages, device)
    df                 = build_followup_features(messages, contacts, embeddings)
    model, df          = train(df, device)
    evaluate(df)
    plot_results(df, model)
    export_results(df)

    # ── Real-time inference demo ──────────────────────────────
    print("\n── Real-time Inference Demo ─────────────────────────────")
    test_cases = [
        ("Hey can you send me that file we talked about?",
         "request", 0.82, True,  0.6, 14, 1, 3, 0.3, 0.5, 0.4, False, 0.2, 0.1),
        ("lol yeah sure",
         "casual_chat", 0.10, False, 0.8, 11, 2, 0, 0.7, 0.8, 0.7, False, 0.1, 0.05),
        ("Don't forget what I said about the project deadline",
         "request", 0.90, True,  0.4, 9,  0, 5, 0.1, 0.3, 0.2, True,  0.8, 0.4),
        ("What time works for you next week?",
         "planning", 0.65, False, 0.7, 15, 3, 2, 0.5, 0.6, 0.5, False, 0.3, 0.1),
    ]

    print(f"\n  {'Content':<45} {'Prob':>6}  {'Label'}")
    print("  " + "-" * 65)
    for args in test_cases:
        res = predict_message(*args, device=device, model=model)
        print(f"  {res['content_preview'][:45]:<45} {res['followup_probability']:>6.3f}  "
              f"{res['urgency'].upper()} {'⚠ FORGOTTEN' if res['is_forgotten_followup'] else ''}")

    print("\nDone ✓")


if __name__ == "__main__":
    main()