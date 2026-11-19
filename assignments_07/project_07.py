from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from scipy.stats import pearsonr

# smolagents imports
from smolagents import OpenAIServerModel, tool, CodeAgent

if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")
api_key = os.getenv("OPENAI_API_KEY")

df = None
DATA_DIR = Path("happiness_project")

# Task 1: Define Your Tools
@tool
def load_happiness_data() -> dict:
    """
    Load and combine all yearly World Happiness CSV files.

    Returns:
    A dictionary containing the dataset shape and column names.
    """
    global df

    COLUMN_NAMES = {
        "Country": "country",
        "Regional indicator": "region",
        "Happiness score": "happiness_score",
        "Ladder score": "happiness_score",   # the 2024 file uses this name
        "GDP per capita": "gdp_per_capita",
    }

    frames = []
    for path in sorted(DATA_DIR.glob("world_happiness_*.csv")):
        year_df = pd.read_csv(path, sep=";", decimal=",")
        year_df = year_df.rename(columns=COLUMN_NAMES)
        year_df["year"] = int(path.stem.split("_")[-1])   # "world_happiness_2020" -> 2020
        frames.append(year_df)

    df = pd.concat(frames, ignore_index=True)

    return {
        "shape": df.shape,
        "columns": df.columns.tolist()
    }



@tool
def summarize_column(column: str) -> dict:
    """
    Return descriptive statistics for a single dataset column.

    Args:
    column: The name of the column to summarize.

    Returns:
    A dictionary containing descriptive statistics or an error message.
    """
    if df is None:
        return {"error": "No data is loaded."}

    if column not in df.columns:
        return {"error": f"'{column}' is not a column. Options: {df.columns.tolist()}"}

    return df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation and p-value between two columns.

    Args:
    col1: Name of the first numeric column.
    col2: Name of the second numeric column.

    Returns:
    A dictionary containing both column names, Pearson correlation,
    and p-value, or an error message.
    """
    if df is None:
        return {"error": "No data is loaded."}

    if col1 not in df.columns or col2 not in df.columns:
        return {"error": "One or both columns are not in the dataset."}

    try:
        pearson_r, p_value = pearsonr(df[col1], df[col2])
    except (TypeError, ValueError):
        return {"error": "Both columns must contain valid numeric data."}

    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(pearson_r), 4),
        "p_value": round(float(p_value), 4),
    }


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a column for a specific year.

    Args:
        column: The column used to rank countries.
        year: The year to filter the dataset by.
        n: The number of top countries to return. Defaults to 5.

    Returns:
        A dictionary containing the year, column, and top N countries,
        or an error message if the input is invalid.
    """
    if df is None:
        return {"error": "No data is loaded."}

    if column not in df.columns:
        return {"error": f"'{column}' is not a column. Options: {df.columns.tolist()}"}

    if n <= 0:
        return {"error": "n must be greater than 0."}

    year_df = df[df["year"] == year]

    if year_df.empty:
        return {"error": f"No data found for the year {year}."}

    try:
        top_n = year_df.nlargest(n, column)[["country", column]].to_dict(orient="records")
    except (TypeError, ValueError):
        return {"error": f"'{column}' must contain numeric values."}

    return {
        "year": year,
        "column": column,
        "top_n_countries": top_n
    }


# Task 2: Build the Agent
model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.

The full pandas DataFrame is provided as the variable df.
Use df directly for custom analysis and plotting.
The load_happiness_data tool returns only a dictionary with dataset metadata, not the DataFrame.
Never create placeholder, sample, or invented data. Always use the actual dataset.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)


# Task 3: Run Guided Queries
queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
]


if __name__ == "__main__":
    # Load the dataset before passing df to the agent
    load_happiness_data()

    # Make sure the output directory exists
    Path("outputs").mkdir(exist_ok=True)

    # --- Task 3: Five guided queries ---
    for i, query in enumerate(queries, start=1):
        print(f"\n--- Guided Query {i}: {query} ---")
        response = agent.run(
            query,
            reset=False,
            additional_args={"df": df}
        )
        print(response)

    # Verify that Query 5 created the plot
    plot_path = Path("outputs/happiness_by_region.png")
    print("\nPlot exists:", plot_path.exists())
    print("Plot size:", plot_path.stat().st_size if plot_path.exists() else "File not found")

    # --- Task 4: Your own questions ---
    my_query_1 = "What were the five happiest countries in 2024?"
    response_1 = agent.run(
        my_query_1,
        reset=False,
        additional_args={"df": df}
    )
    print("\n--- My Query 1 ---")
    print(response_1)
    # Comment: This query triggered tool use. The agent called the get_top_n_countries tool to retrieve the five happiest countries in 2024. It did not need to write custom pandas code.

    my_query_2 = (
        "Which region had the largest improvement in average happiness "
        "score from 2015 to 2024? Calculate the average score for each "
        "region in both years and compare the changes."
    )
    response_2 = agent.run(
        my_query_2,
        reset=False,
        additional_args={"df": df}
    )
    print("\n--- My Query 2 ---")
    print(response_2)
    # Comment: This query triggered code generation. The agent wrote custom pandas code to calculate average happiness scores by region for 2015 and 2024, compare the results, and identify the region with the largest improvement.


# --- Reflection ---
#
# 1. The agent reported a positive correlation (r = 0.6313) between GDP per capita and happiness score. It described the relationship as moderately strong and statistically significant. The p-value was very close to zero, so the result is unlikely to be due to chance.

# 2. I was surprised that the CodeAgent could write its own pandas code to compare happiness scores across regions and identify which region improved the most. It could answer a question that was not covered by the existing tools.

# 3. An additional tool that could be useful is one that compares a country's happiness score across different years. It would help answer questions about how a specific country's happiness has changed over time.



