"""Dataset loading + a synthetic data generator so the project works fully offline.

Real data: put the SMS Spam Collection (UCI) or Kaggle 'spam.csv' in data/ and pass --data.
Synthetic data: `python -m src.data_utils` writes data/emails.csv (and data/drifted.csv).
"""
import argparse
import random
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR

HAM_TEMPLATES = [
    "Hi {n}, can we move the {t} meeting to {d}?",
    "Please find attached the {t} report for review.",
    "Thanks for your help yesterday, I really appreciate it.",
    "Are you free for lunch on {d}? Let me know.",
    "Reminder: the {t} deadline is {d}. Please submit your part.",
    "Can you send me the notes from the {t} call?",
    "Happy birthday {n}! Hope you have a great day.",
    "I will be late today because of traffic, start without me.",
    "The {t} documents are ready, let me know if you need changes.",
    "Dinner at my place on {d}? I am making pasta.",
    "Click the link in the calendar invite to join the {t} call on {d}.",
    "Free to talk now? Call me when you get this, it is urgent about the {t} contract.",
    "Congratulations {n}, you won the team quiz! Prize is lunch on {d}.",
    "Please verify the account numbers in the {t} spreadsheet before {d}.",
    "Limited seats left for the {t} workshop, register by {d}.",
    "Your order has shipped and will arrive {d}. Track it in the app.",
    "Hey {n}, did you get a chance to look at the {t} proposal?",
    "Your invoice for {t} has been approved and will be processed this week.",
    "Let's catch up this weekend, it has been a while.",
    "Could you review my pull request when you get a moment?",
    "The class is cancelled on {d}, see the updated schedule.",
]
SPAM_TEMPLATES = [
    "WINNER!! You have been selected to receive a ${a} gift card. Click {u} now!",
    "Congratulations {n}! You won a free {p}. Claim your prize at {u}",
    "URGENT: your account is suspended. Verify your password at {u} immediately",
    "Get cheap {p} pills online, no prescription needed. Visit {u}",
    "Earn ${a} per day working from home! Limited offer, click {u}",
    "FREE entry in our weekly draw to win a {p}. Text WIN to 80085 now",
    "You have a pending refund of ${a}. Confirm your bank details at {u}",
    "Lowest mortgage rates guaranteed! Apply now at {u}, act fast",
    "Final notice: claim your ${a} reward before it expires. Call now",
    "Hot singles in your area want to meet you. Click {u}",
    "Hi {n}, quick note about your account, see details here {u}",
    "Hello, are you free to talk about an offer for {t}? Visit {u}",
    "Your order could not be delivered, update your address {u}",
    "{n}, you have a new message waiting, open {u}",
    "Exclusive {t} deal for you, register by {d} at {u}",
    "Buy {p} at 90% discount today only!!! Order at {u}",
]
# "Drifted" spam: new obfuscated / crypto style that the original model never saw
DRIFT_SPAM_TEMPLATES = [
    "Y0u'v3 b33n ch0sen 4 a cr yp70 b0nus, rep1y YES 2 cl@im",
    "Ur w4ll3t needs v3rification 2day, send us ur s33d ph rase",
    "Inv3st in M00nCoin now, 500x r3turns gu@ranteed, DM us",
    "Fr33 NFT airdr0p for early m3mbers, c0nnect w@llet n0w",
    "Y0ur p@ckage is held, pay cust0ms fee via g1ft card 2 rel3ase",
    "Tr@de b0t makes ${a} d@ily, m3ssage me on t3legram",
    "Hey {n}, saw ur pr0file, my fr13nd made ${a} w1th this, ask me h0w",
]
DRIFT_HAM_TEMPLATES = [
    "Kubernetes rollout finished, pods are healthy on all nodes.",
    "Standup moved to async, post your updates in the channel.",
    "Sprint retro notes are in the wiki, please add your feedback.",
    "The staging deploy failed on the migration step, looking into it.",
]
NAMES = ["Alex", "Sam", "Priya", "John", "Maria", "Ravi", "Emma", "Chen"]
TOPICS = ["budget", "project", "sales", "design", "hiring", "marketing", "legal"]
DAYS = ["Monday", "Tuesday", "Friday", "tomorrow", "next week", "3pm"]
PRIZES = ["iPhone", "laptop", "holiday trip", "car", "cash voucher"]
URLS = ["http://bit.ly/x9Z", "www.claim-now.biz", "http://free-prize.net/a1", "http://secure-login.info"]


def _fill(tpl, rng):
    return tpl.format(n=rng.choice(NAMES), t=rng.choice(TOPICS), d=rng.choice(DAYS),
                      a=rng.choice([100, 500, 1000, 5000]), p=rng.choice(PRIZES), u=rng.choice(URLS))


def generate_dataset(n=3000, spam_ratio=0.35, seed=42, drift=False, label_noise=0.03) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        is_spam = rng.random() < spam_ratio
        if drift:
            pool = DRIFT_SPAM_TEMPLATES if is_spam else DRIFT_HAM_TEMPLATES + HAM_TEMPLATES[:4]
        else:
            pool = SPAM_TEMPLATES if is_spam else HAM_TEMPLATES
        text = _fill(rng.choice(pool), rng)
        if rng.random() < label_noise:      # real-world labels are never perfect
            is_spam = not is_spam
        rows.append({"label": "spam" if is_spam else "ham", "text": text})
    return pd.DataFrame(rows)


def load_dataset(path) -> pd.DataFrame:
    """Load a CSV/TSV and normalise it to columns: label (0/1), text."""
    path = Path(path)
    if path.suffix.lower() in {".tsv", ".txt"} or path.name == "SMSSpamCollection":
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], encoding="latin-1")
    else:
        df = pd.read_csv(path, encoding="latin-1")
        if {"v1", "v2"}.issubset(df.columns):                 # Kaggle spam.csv format
            df = df.rename(columns={"v1": "label", "v2": "text"})
        elif "Category" in df.columns and "Message" in df.columns:
            df = df.rename(columns={"Category": "label", "Message": "text"})
        elif "email_text" in df.columns:
            df = df.rename(columns={"email_text": "text"})
    if not {"label", "text"}.issubset(df.columns):
        raise ValueError(f"Dataset needs 'label' and 'text' columns, found {list(df.columns)}")
    df = df[["label", "text"]].dropna().drop_duplicates().reset_index(drop=True)
    if not pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].astype(str).str.strip().str.lower().map({"spam": 1, "ham": 0})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate synthetic spam datasets")
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()
    generate_dataset(args.n).to_csv(DATA_DIR / "emails.csv", index=False)
    generate_dataset(500, seed=7, drift=True).to_csv(DATA_DIR / "drifted.csv", index=False)
    print(f"Wrote {DATA_DIR/'emails.csv'} and {DATA_DIR/'drifted.csv'}")
