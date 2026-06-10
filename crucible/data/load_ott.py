"""
Ott Deceptive Opinion Spam dataset loader.

Tries, in order:
  1. A local cached CSV (data/ott.csv) if present.
  2. kagglehub download of the Ott Deceptive Opinion Spam corpus.
  3. A small embedded fallback sample (so the server / eval loop always boots,
     even without Kaggle credentials or network access).

Always returns a pandas DataFrame with at least these columns:
  - text      : the review body
  - hotel     : hotel name
  - deceptive : "truthful" or "deceptive"
"""
import os
import logging

import pandas as pd

logger = logging.getLogger("crucible.data")

DATA_DIR = os.path.dirname(__file__)
CACHE_CSV = os.path.join(DATA_DIR, "ott.csv")


# ---------------------------------------------------------------------------
# Embedded fallback sample — balanced truthful / deceptive reviews.
# Lets the pipeline run end-to-end without external data access.
# ---------------------------------------------------------------------------
_FALLBACK = [
    # truthful
    ("truthful", "The Drake", "We stayed three nights at the Drake for my sister's wedding. Room 1204 had a slightly noisy radiator but the view of Lake Michigan made up for it. The doorman, Carlos, even helped us flag a cab in the rain on Walton Street."),
    ("truthful", "The Palmer House Hilton", "Beautiful historic lobby but honestly the rooms are showing their age. Our shower drained slowly and the wifi dropped twice during a work call. Still, location near the Loop is unbeatable and the staff were kind about the maintenance request."),
    ("truthful", "Conrad Chicago", "Decent stay. Check-in took forever because they were short staffed on a Friday night. The bed was comfortable and breakfast at the cafe across Rush Street was a nice touch, though a bit pricey for what you get."),
    ("truthful", "Hilton Chicago", "Came for a conference. The meeting rooms were freezing and I had to ask for extra blankets. That said, the gym was open 24 hours which I appreciated after late sessions, and Grant Park is right across Michigan Ave for a morning run."),
    ("truthful", "Sheraton Chicago", "Mixed feelings. River view room was lovely at sunset but the elevators were painfully slow and one was out of service the whole weekend. Bartender at the lobby bar remembered our order on night two, small thing that made it feel personal."),
    ("truthful", "Swissotel Chicago", "Solid business hotel. Slightly dated decor in the rooms but spotlessly clean. The concierge booked us a last-minute table at a deep dish place on Wabash and it ended up being the highlight of the trip."),
    ("truthful", "Hyatt Regency Chicago", "Massive property, easy to get lost. Our first room smelled faintly of smoke so they moved us, which took about an hour. The new room on the 18th floor was great. Coffee in the lobby was watered down though."),
    ("truthful", "Homewood Suites", "Stayed a week for work. The kitchenette was a lifesaver and the free breakfast had real eggs, not the powdered kind. AC was loud at night but I got used to it. Walking distance to the train made commuting downtown painless."),
    # deceptive
    ("deceptive", "The Drake", "My stay at The Drake was absolutely perfect in every way. The luxurious rooms and impeccable service made this the best hotel experience of my life. I would recommend this amazing property to anyone visiting Chicago."),
    ("deceptive", "The Palmer House Hilton", "The Palmer House Hilton exceeded all of my expectations. The staff was incredibly friendly and the rooms were spacious and elegant. The location near Michigan Avenue and the Loop is fantastic. A truly five star experience."),
    ("deceptive", "Conrad Chicago", "What an incredible hotel! From the moment I walked in I was treated like royalty. The beds were the most comfortable I have ever slept in and the views were breathtaking. I cannot wait to return to this wonderful place."),
    ("deceptive", "Hilton Chicago", "This is hands down the finest hotel in Chicago. Everything was flawless, the dining was world class, and the concierge anticipated my every need. If you want a luxurious and unforgettable stay, look no further than the Hilton Chicago."),
    ("deceptive", "Sheraton Chicago", "I had a magical experience at the Sheraton. The elegant rooms, the stunning river views, and the gracious staff combined to create an absolutely perfect getaway. This is luxury at its very best and I highly recommend it."),
    ("deceptive", "Swissotel Chicago", "The Swissotel is a masterpiece of hospitality. Every detail was perfect, the service was impeccable, and the amenities were second to none. My family and I were treated wonderfully and we will definitely be coming back soon."),
    ("deceptive", "Hyatt Regency Chicago", "Simply the best hotel I have ever stayed at. The Hyatt Regency offers unparalleled luxury, gorgeous rooms, and an incredible location. The staff went above and beyond to make our stay perfect. Five glowing stars from me!"),
    ("deceptive", "Hard Rock Hotel Chicago", "An absolutely outstanding hotel with a vibrant atmosphere. The rooms were stylish and immaculate, the staff was phenomenal, and the location could not be better. This was the perfect choice for our trip and I recommend it wholeheartedly."),
]


def _fallback_df() -> pd.DataFrame:
    logger.warning("[Data] Using embedded fallback Ott sample (%d reviews).", len(_FALLBACK))
    rows = [{"deceptive": d, "hotel": h, "text": t, "polarity": "positive", "source": "fallback"}
            for (d, h, t) in _FALLBACK]
    return pd.DataFrame(rows)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the required columns exist regardless of the source schema."""
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    if "deceptive" not in df.columns and "deceptive" in cols:
        rename[cols["deceptive"]] = "deceptive"
    if "text" not in df.columns and "text" in cols:
        rename[cols["text"]] = "text"
    if "hotel" not in df.columns and "hotel" in cols:
        rename[cols["hotel"]] = "hotel"
    if rename:
        df = df.rename(columns=rename)
    if "hotel" not in df.columns:
        df["hotel"] = "Chicago Hotel"
    return df


def load_ott_data() -> pd.DataFrame:
    """Load the Ott Deceptive Opinion Spam dataset (or a fallback sample)."""
    # 1. Local cached CSV
    if os.path.exists(CACHE_CSV):
        try:
            df = _normalize(pd.read_csv(CACHE_CSV))
            if {"text", "deceptive"}.issubset(df.columns) and len(df) > 0:
                logger.info("[Data] Loaded Ott dataset from cache: %s (%d rows).", CACHE_CSV, len(df))
                return df
        except Exception as e:
            logger.warning("[Data] Failed to read cache %s: %s", CACHE_CSV, e)

    # 2. kagglehub download
    try:
        import kagglehub

        path = kagglehub.dataset_download("rtatman/deceptive-opinion-spam-corpus")
        # Find the CSV inside the downloaded dataset directory
        csv_path = None
        for root, _dirs, files in os.walk(path):
            for f in files:
                if f.lower().endswith(".csv"):
                    csv_path = os.path.join(root, f)
                    break
            if csv_path:
                break
        if csv_path:
            df = _normalize(pd.read_csv(csv_path))
            if {"text", "deceptive"}.issubset(df.columns) and len(df) > 0:
                try:
                    df.to_csv(CACHE_CSV, index=False)
                except Exception:
                    pass
                logger.info("[Data] Loaded Ott dataset via kagglehub (%d rows).", len(df))
                return df
    except Exception as e:
        logger.warning("[Data] kagglehub load failed (%s). Falling back to embedded sample.", e)

    # 3. Embedded fallback
    return _fallback_df()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    d = load_ott_data()
    print(d["deceptive"].value_counts())
    print(d.head())
