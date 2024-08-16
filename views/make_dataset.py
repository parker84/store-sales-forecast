import pandas as pd
from decouple import config
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import logging, coloredlogs
from io import StringIO
import streamlit as st

st.title("Making Training Datasets 📦")
st.caption("Cleaning the Data, Engineering Features, and :axe: splitting the data into training, validation, and holdout sets.")

# Set up a StringIO to capture log messages
log_stream = StringIO()

# Configure logging
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, stream=log_stream, format='%(asctime)s - %(message)s')


# Define a custom logging handler to write logs to Streamlit
class StreamlitLoggingHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        st.text(log_entry)

logger = logging.getLogger(__name__)
logger.addHandler(StreamlitLoggingHandler())
coloredlogs.install(level=config('LOG_LEVEL', 'INFO'), logger=logger)

# --------- Constants ---------
HOLDOUT_DATE = '2017-01-01'
TRAIN_VAL_SPLIT = 0.8
RANDOM_STATE = 42
DAYS_FOR_FEATURES = [7, 14, 30, 90, 180, 365]

# --------- Helpers ---------
def qa_data(data):
    # check for date holes
    counts_per_store_and_family = (
        data
        .groupby(['store_nbr', 'family'])
        .size()
    )
    assert counts_per_store_and_family.min() == counts_per_store_and_family.max(), "There are missing dates in the data"
    # check for missing values
    assert int(data.isnull().sum().sum()) == 0, "There are missing values in the data"

def test_rolling_features(df):
    sample_df = df[df['store_nbr'] == 54][df['family'] == 'AUTOMOTIVE'] # TODO: loop over some more families and stores
    for days in DAYS_FOR_FEATURES:
        avg_sales = sample_df.iloc[-(days+1):-1]['sales'].mean()
        avg_sales_feature = sample_df[f'avg_{days}day_per_store_and_family'].iloc[-1]
        if days > 200: # floating point precision issues
            assert round(avg_sales, 1) == round(avg_sales_feature, 1), f"The {days}d average rolling features are not calculated correctly, expected {avg_sales} but got {avg_sales_feature}"
        else:
            assert avg_sales == avg_sales_feature, f"The {days}d average rolling features are not calculated correctly, expected {avg_sales} but got {avg_sales_feature}"
        median_sales = sample_df.iloc[-(days+1):-1]['sales'].median()
        median_sales_feature = sample_df[f'median_{days}day_per_store_and_family'].iloc[-1]
        assert median_sales == median_sales_feature, f"The {days}d median rolling features are not calculated correctly, expected {median_sales} but got {median_sales_feature}"

def eng_features(train_df):
    # TODO: add seasonality features (ex: day of week, month, ...)
    in_df = train_df.copy()
    features = in_df[[
        'store_nbr',
        'family',
        'date',
        'sales',
    ]].copy().sort_values(by=['store_nbr', 'family', 'date']) # Note: you need this for the window function to calculate properly, unless you reset the index or use the 'on' parameter in rolling
    for days in tqdm(DAYS_FOR_FEATURES):
        features[f'avg_{days}day_per_store_and_family'] = (
            features
            .groupby(['store_nbr', 'family'])['sales']
            .transform(lambda x: x.rolling(days, min_periods=days).mean())
        )
        features[f'median_{days}day_per_store_and_family'] = (
            features
            .groupby(['store_nbr', 'family'])['sales']
            .transform(lambda x: x.rolling(days, min_periods=days).median()) # Note: super important that there's no date holes for this to work properly
        )
        features[f'max_{days}day_per_store_and_family'] = (
            features
            .groupby(['store_nbr', 'family'])['sales']
            .transform(lambda x: x.rolling(days, min_periods=days).max())
        )
        features[f'min_{days}day_per_store_and_family'] = (
            features
            .groupby(['store_nbr', 'family'])['sales']
            .transform(lambda x: x.rolling(days, min_periods=days).min())
        )
    features['actual_features_date'] = pd.to_datetime(features['date'])
    features['date'] = pd.to_datetime(features['date']) + pd.Timedelta(days=1) # shift up one so when joining you're joining this onto the next date because the rolling calculations include the current date
    in_df['date'] = pd.to_datetime(in_df['date'])
    out_df = in_df.merge(
        features.drop(columns=['sales']), 
        on=['store_nbr', 'family', 'date'], 
        how='left'
    )
    return out_df



def make_datasets():
    # --------- Load Data ---------
    logger.info("📚 Reading in the raw training data")
    train_df = pd.read_csv('data/raw/train.csv').sort_values(by=['store_nbr', 'family', 'date'])
    train_df['sales'] = train_df['sales'].astype(float)
    logger.info(f"🔹 Shape of raw train_df: {train_df.shape}")

    # --------- QA Data ---------
    logger.info("🧪 QAing the Raw Data")
    qa_data(train_df)
    logger.info("✅ Raw Data QA Passed")


    # --------- Engineering Features ---------
    logger.info("⚙️ Engineering Features")
    train_df = eng_features(train_df)
    logger.info("✅ Features Engineered")

    logger.info(f"🔹 Shape of train_df before dropping nulls: {train_df.shape}")
    train_df = train_df.dropna() # drop the NaNs from the rolling calculations
    logger.info(f"🔹 Shape of train_df after dropping nulls: {train_df.shape}")
    # Note: we wouldn't want to do this if in production we'd be applying this model to data with less than 52 weeks of data
    # In that case we may want a separate training/dataset + model for those newer stores/families

    # --------- QA Engineered Data ---------
    logger.info("🧪 QAing the Engineered Data")
    qa_data(train_df)
    logger.info("✅ Engineered Data QA Passed")

    # --------- Test Rolling Features ---------
    logger.info("🧪 Testing the Rolling Features")
    test_rolling_features(train_df)
    logger.info("✅ Rolling Feature Tests Passed")

    # --------- Split Data ---------
    logger.info("🪓 Splitting the Data")
    holdout = train_df[train_df['date'] >= HOLDOUT_DATE]
    train, val = train_test_split(
        train_df[train_df['date'] < HOLDOUT_DATE],
        train_size=TRAIN_VAL_SPLIT,
        random_state=RANDOM_STATE,
    )

    # --------- Save Data ---------
    logger.info("💾 Saving the Data")
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
    train.to_csv('data/processed/train.csv', index=False)
    val.to_csv('data/processed/val.csv', index=False)
    holdout.to_csv('data/processed/holdout.csv', index=False)

    logger.info("✅ Done!")


with st.expander('Logs 🪵', expanded=True):
    make_datasets()

with st.expander('The 🛠️ Code'):
    with open('views/make_dataset.py') as f:
        st.code(f.read())