import os
import pandas as pd
from src.utils.deck import compute_deck_preferences


def load_igt_data(path: str, n_trials: int = 100) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load IGT datasets from CSV files,
    """
    assert n_trials in [95, 100, 150], "Number of trials must be either 95, 100, or 150."

    deck_file = f"choice_{n_trials}.csv"
    wins_file = f"wi_{n_trials}.csv"
    losses_file = f"lo_{n_trials}.csv"
    deck_data = pd.read_csv(os.path.join(path, deck_file))
    wins_data = pd.read_csv(os.path.join(path, wins_file))
    losses_data = pd.read_csv(os.path.join(path, losses_file))

    return deck_data, wins_data, losses_data


def preprocess_igt_dataset(df: pd.DataFrame, type: str) -> pd.DataFrame:
    """
    Preprocess IGT dataset based on the specified type: 'deck', 'wins', or 'losses'.
    """
    assert type in ["deck", "wins", "losses"], "Type must be one of 'deck', 'wins', or 'losses'."

    df_long = None

    if type == "deck":
        # Reshape from wide to long format
        df_long = df.melt(
            id_vars = df.columns[0],
            var_name = "trial_col",
            value_name = "deck"
        )

    elif type == "wins":
        df_long = df.melt(
            id_vars = df.columns[0],
            var_name = "trial_col",
            value_name = "reward"
        )

    elif type == "losses":
        df_long = df.melt(
            id_vars = df.columns[0],
            var_name = "trial_col",
            value_name = "loss"
        )

    # Extract numeric trial number
    df_long["trial"] = df_long["trial_col"].str.extract(r'(\d+)').astype(int)

    # Assign metadata columns
    df_long["agent"] = df_long[df.columns[0]].astype(str).str.extract(r'(\d+)').astype(int)

    # Sort to ensure agent order
    df_long = df_long.sort_values(["agent", "trial"]).reset_index(drop = True)

    # Re-order columns to match the correct format
    cols = ["agent", "trial", "loss"]
    df_long = df_long[cols]

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
    combined_df["reward"] = combined_df["reward"].fillna(0) - combined_df["loss"].fillna(0)

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
    deck_df = preprocess_igt_dataset(deck_data, type = "deck")
    wins_df = preprocess_igt_dataset(wins_data, type = "wins")
    losses_df = preprocess_igt_dataset(losses_data, type = "losses")

    combined_df = combine_igt_data(deck_df, wins_df, losses_df)

    # Compute deck preferences and advantage index
    prefs = compute_deck_preferences(combined_df, mode = "human")

    return combined_df