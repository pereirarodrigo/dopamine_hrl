import os
import re
import pandas as pd


def load_igt_data(path: str, n_trials: int = 100) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load IGT datasets from CSV files,
    """
    assert n_trials in [95, 100, 150], "Number of trials must be either 95, 100, or 150."

    # Define file names based on number of trials
    deck_file = f"choice_{n_trials}.csv"
    wins_file = f"wi_{n_trials}.csv"
    losses_file = f"lo_{n_trials}.csv"

    # Load datasets
    deck_data = pd.read_csv(os.path.join(path, deck_file), index_col = 0)
    wins_data = pd.read_csv(os.path.join(path, wins_file), index_col = 0)
    losses_data = pd.read_csv(os.path.join(path, losses_file), index_col = 0)

    # Make the index a proper column
    deck_data = deck_data.reset_index()
    wins_data = wins_data.reset_index()
    losses_data = losses_data.reset_index()

    return deck_data, wins_data, losses_data


def preprocess_igt_dataset(df: pd.DataFrame, res_type: str) -> pd.DataFrame:
    """
    Preprocess IGT dataset based on the specified result type.
    """
    assert res_type in ["deck", "reward", "loss"], "Type must be one of 'deck', 'reward', or 'losses'."

    # Identify subject column (the first column, e.g., 'Subj_1')
    subj_col = df.columns[0]

    # Melt into long format
    df_long = df.melt(
        id_vars = subj_col,
        var_name = "trial_col",
        value_name = res_type
    )

    # Extract numeric trial number from column name
    df_long["trial"] = df_long["trial_col"].apply(
        lambda x: int(re.search(r"(\d+)$", x).group(1))
    )

    # Extract subject number from the first column
    df_long["agent"] = df_long[subj_col].astype(str).apply(
        lambda x: int(re.search(r"(\d+)$", x).group(1))
    )

    df_long = (
        df_long[["agent", "trial", res_type]]
        .drop_duplicates(["agent", "trial"])
        .sort_values(["agent", "trial"])
        .reset_index(drop = True)
    )

    return df_long


def combine_igt_data(deck_df: pd.DataFrame, wins_df: pd.DataFrame, losses_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine preprocessed IGT deck, wins, and losses data into a single DataFrame.
    """
    # Merge deck and wins data
    combined_df = pd.merge(
        deck_df,
        wins_df,
        on = ["agent", "trial"],
        how = "left"
    )

    # Merge losses data
    combined_df = pd.merge(
        combined_df,
        losses_df,
        on = ["agent", "trial"],
        how = "left"
    )

    # Calculate net reward (wins - losses)
    combined_df["reward"] = combined_df["reward"].fillna(0) - abs(combined_df["loss"]).fillna(0)

    # Drop the separate loss column
    combined_df = combined_df.drop(columns = ["loss"])

    return combined_df


def build_igt_dataset(path: str, n_trials: int = 100) -> pd.DataFrame:
    """
    Load and preprocess the IGT dataset from the specified path.
    """
    # Load raw data
    deck_data, wins_data, losses_data = load_igt_data(path, n_trials)

    # Preprocess individual datasets
    deck_df = preprocess_igt_dataset(deck_data, res_type = "deck")
    wins_df = preprocess_igt_dataset(wins_data, res_type = "reward")
    losses_df = preprocess_igt_dataset(losses_data, res_type = "loss")

    combined_df = combine_igt_data(deck_df, wins_df, losses_df)

    return combined_df