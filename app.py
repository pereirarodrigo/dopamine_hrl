import os
import subprocess
import pandas as pd
from glob import glob
import streamlit as st
from pathlib import Path
from src.config import BEHAVIOURAL_OUTPUT_PATH, SUMMARY_OUTPUT_PATH

# Page setup
st.set_page_config(page_title = "Dopamine-HRL Dashboard", layout = "wide")
st.title("🧠 Dopamine-HRL Research Dashboard")
st.markdown("Run simulations, generate plots, and compare agents vs. humans - all in one interface.")

# Sidebar controls
st.sidebar.header("Global Controls")
n_trials = st.sidebar.number_input("Number of trials", 20, 200, 100)
num_seeds = st.sidebar.number_input("Number of seeds", 1, 100, 10)
agents_per_condition = st.sidebar.number_input("Agents per condition", 1, 200, 50)
num_episodes = st.sidebar.number_input("Number of episodes", 1, 200, 10)
trials_per_episode = st.sidebar.number_input("Trials per episode", 10, 500, 20)

# Path settings
summary_output_dir = Path(f"{SUMMARY_OUTPUT_PATH}/hrl/choice_{n_trials}")
behavioural_output_dir = Path(f"{BEHAVIOURAL_OUTPUT_PATH}/choice_{n_trials}")

os.makedirs(summary_output_dir, exist_ok = True)
os.makedirs(behavioural_output_dir, exist_ok = True)


def run_script(script_path: str, args: list = None) -> None:
    """
    Utility to execute scripts and stream logs.
    """
    args = args or []
    cmd = ["uv", "run", "python", "-m", script_path, *map(str, args)]

    st.info(f"🚀 Running `{script_path}`...")
    progress_bar = st.progress(0)
    log_placeholder = st.empty()

    process = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True)

    logs = []
    for i, line in enumerate(iter(process.stdout.readline, '')):
        logs.append(line.strip())

        # Display the last few lines live
        log_placeholder.text("\n".join(logs[-10:]))

        # crude but effective progress feedback
        if "episode" in line.lower() and "/" in line:
            num = int(line.split("/")[0].split()[-1])
            total = int(line.split("/")[-1].split()[0])
            progress_bar.progress(num / total)


    process.wait()
    progress_bar.progress(1.0)

    if process.returncode == 0:
        st.success(f"✅ `{os.path.basename(script_path)}` completed successfully.")

    else:
        st.error(f"❌ `{os.path.basename(script_path)}` failed.")
        st.text("\n".join(logs[-20:]))

    with st.expander("View full logs"):
        st.text("\n".join(logs))

# ──────────────────────────────────────────────
# Tabs
tabs = st.tabs([
    "🎯 Main Simulation",
    "🧩 Baselines",
    "📊 Summarise Results",
    "🔬 Evaluation"
])

# ───────────────────── Main Simulation ─────────────────────
with tabs[0]:
    st.header("Agent Simulation")
    st.markdown("Configure and run the dopamine-HRL agent simulation.")

    if st.button("▶ Run Simulation", key = "main_sim"):
        # Pass the parameters as command-line arguments to the script
        args = [
            f"--num_seeds={num_seeds}",
            f"--agents_per_condition={agents_per_condition}",
            f"--num_episodes={num_episodes}",
            f"--trials_per_episode={trials_per_episode}"
        ]
        run_script("src.simulation.run_main_simulation", args)

        st.markdown("Simulation completed.")

# ───────────────────── Baselines ─────────────────────
with tabs[1]:
    st.header("Baseline Simulations")
    st.markdown("Run flat TD and random baseline agents for comparison.")

    # Create two columns for side-by-side buttons
    col1, col2 = st.columns(2, gap = None)

    with col1:
        if st.button("▶ Run Random Policy", key = "random"):
            args = [
                f"--num_seeds={num_seeds}",
                f"--agents_per_condition={agents_per_condition}",
                f"--num_episodes={num_episodes}",
                f"--trials_per_episode={trials_per_episode}",
                f"--baseline_agent_type=random"
            ]
            run_script("src.simulation.run_baselines", args)
            st.markdown("Simulation completed.")

    with col2:
        if st.button("▶ Run Flat TD Policy", key = "flat_td"):
            args = [
                f"--num_seeds={num_seeds}",
                f"--agents_per_condition={agents_per_condition}",
                f"--num_episodes={num_episodes}",
                f"--trials_per_episode={trials_per_episode}",
                f"--baseline_agent_type=flat_td"
            ]
            run_script("src.simulation.run_baselines", args)
            st.markdown("Simulation completed.")

# ───────────────────── Summarise results ─────────────────────
with tabs[2]:
    st.header("Result Summarisation")
    st.markdown("Generate descriptive plots and summaries from simulation outputs.")

    if st.button("📊 Summarise Results", key = "summarise"):
        run_script("src.simulation.summarise_results")

    st.markdown("---")
    csv_files = sorted(glob(str(summary_output_dir / "*.csv")))
    plot_files = sorted(glob(str(summary_output_dir / "*.png")))

    if csv_files:
        st.subheader("📈 Data Tables")

        for csv_path in csv_files:
            st.markdown(f"**{Path(csv_path).name}**")
            df = pd.read_csv(csv_path)
            num_cols = df.select_dtypes(include = ["float"]).columns

            # Format numerical columns (except reward_mse_mean) for better readability
            for col in num_cols:
                if col != "reward_mse_mean":
                    df[col] = df[col].apply(
                        lambda x: f"{x:.3e}" if abs(x) < 0.001 or abs(x) > 100000 else f"{x:,.4f}"
                    )

            st.dataframe(df, width = "stretch")

    if plot_files:
        st.subheader("🖼️ Generated Plots")
        cols = st.columns(2)

        for i, plot_path in enumerate(plot_files):
            with cols[i % 2]:
                st.image(plot_path, caption = Path(plot_path).name, width = "stretch")

    if not csv_files and not plot_files:
        st.info("No output files found yet. Run summarisation to generate them.")

# ───────────────────── Statistical evaluation ─────────────────────
with tabs[3]:
    st.header("Statistical Evaluation")
    st.markdown("Compare agent behaviour to human IGT data, including t-tests and ANOVAs.")

    if st.button("🔬 Run Evaluation", key = "evaluation"):
        run_script("src.simulation.statistical_evaluation")

    csv_files = sorted(glob(str(behavioural_output_dir / "*.csv")))
    eval_plots = sorted(glob(str(behavioural_output_dir / "*.png")))

    if csv_files:
        st.subheader("📊 Statistical Results")

        for csv_path in csv_files:
            st.markdown(f"**{Path(csv_path).name}**")
            df = pd.read_csv(csv_path)
            num_cols = df.select_dtypes(include = ["float"]).columns

            # Ditto formatting
            for col in num_cols:
                if col != "reward_mse_mean":
                    df[col] = df[col].apply(
                        lambda x: f"{x:.3e}" if abs(x) < 0.001 or abs(x) > 100000 else f"{x:,.4f}"
                    )

            st.dataframe(df, width = "stretch")

    if eval_plots:
        st.subheader("📉 Evaluation Plots")
        cols = st.columns(2)

        for i, plot_path in enumerate(eval_plots):
            with cols[i % 2]:
                st.image(plot_path, caption = Path(plot_path).name, width = "stretch")

    if not csv_files and not eval_plots:
        st.info("No evaluation results found. Run the evaluation module first.")
