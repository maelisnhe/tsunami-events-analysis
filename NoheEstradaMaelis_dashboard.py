from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


PROJECT_TITLE = "Tsunami Events Analysis Dashboard"
PROJECT_SUBTITLE = "Analyse temporelle, spatiale et physique des événements de tsunamis"
DATASET_NAME = "tsunami_dataset.csv"
DATASET_SOURCE = "Provided project dataset"
TEAM_MEMBERS = "Maëlis Nohe Estrada, Raoul Dragus, Reda Ait Taleb"
GITHUB_LINK = "https://github.com/maelisnhe/tsunami-events-analysis"
HTML_EXPORT_NAME = "NoheEstradaMaelis_dashboard.html"

BG_COLOR = "#07111F"
CARD_COLOR = "#0F1B2E"
CARD_ALT = "#13233A"
TEXT_COLOR = "#F4F7FB"
MUTED_TEXT = "#BCD0E5"
BORDER_COLOR = "#223A5E"
ACCENT_PRIMARY = "#2D6CDF"
ACCENT_SECONDARY = "#4CC9F0"
ACCENT_DANGER = "#F77F00"


def find_csv_files(folder_path: Path) -> List[Path]:
    """
    Input:
        folder_path: path to the project folder.
    Output:
        list of CSV files found in the folder.
    """
    return sorted(folder_path.glob("*.csv"))


def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Input:
        csv_path: path to the tsunami CSV file.
    Output:
        pandas DataFrame containing the raw dataset.
    """
    return pd.read_csv(csv_path, low_memory=False)


def select_tsunami_csv(csv_files: List[Path]) -> Path:
    """
    Input:
        csv_files: list of CSV files.
    Output:
        selected path corresponding to tsunami_dataset.csv.
    """
    for csv_path in csv_files:
        if csv_path.name.lower() == "tsunami_dataset.csv":
            return csv_path

    return csv_files[0]


def preprocess_tsunami_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        df: raw tsunami DataFrame.
    Output:
        cleaned DataFrame with numeric columns converted and useful date fields prepared.
    """
    cleaned_df = df.copy()

    numeric_columns = [
        "YEAR",
        "MONTH",
        "DAY",
        "HOUR",
        "MINUTE",
        "LATITUDE",
        "LONGITUDE",
        "EQ_MAGNITUDE",
        "EQ_DEPTH",
        "TS_INTENSITY",
        "DAMAGE",
        "HOUSES",
        "DEATHS",
    ]

    for column in numeric_columns:
        if column in cleaned_df.columns:
            cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors="coerce")

    if all(col in cleaned_df.columns for col in ["YEAR", "MONTH", "DAY"]):
        cleaned_df["EVENT_DATE"] = pd.to_datetime(
            {
                "year": cleaned_df["YEAR"],
                "month": cleaned_df["MONTH"],
                "day": cleaned_df["DAY"],
            },
            errors="coerce",
        )
    else:
        cleaned_df["EVENT_DATE"] = pd.NaT

    if "YEAR" in cleaned_df.columns:
        cleaned_df["DECADE"] = (cleaned_df["YEAR"] // 10) * 10
    else:
        cleaned_df["DECADE"] = np.nan

    return cleaned_df


def compute_overview_kpis(df: pd.DataFrame) -> Dict[str, object]:
    """
    Input:
        df: cleaned tsunami DataFrame.
    Output:
        dictionary containing overview KPI values.
    """
    return {
        "total_events": int(len(df)),
        "total_countries": int(df["COUNTRY"].nunique()) if "COUNTRY" in df.columns else np.nan,
        "year_min": float(df["YEAR"].min()) if "YEAR" in df.columns else np.nan,
        "year_max": float(df["YEAR"].max()) if "YEAR" in df.columns else np.nan,
        "average_magnitude": float(df["EQ_MAGNITUDE"].mean()) if "EQ_MAGNITUDE" in df.columns else np.nan,
        "average_depth": float(df["EQ_DEPTH"].mean()) if "EQ_DEPTH" in df.columns else np.nan,
        "total_deaths": float(df["DEATHS"].sum(skipna=True)) if "DEATHS" in df.columns else np.nan,
    }


def compute_top_countries_by_events(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Input:
        df: cleaned tsunami DataFrame.
        top_n: number of countries to display.
    Output:
        DataFrame containing countries and number of tsunami events.
    """
    if "COUNTRY" not in df.columns:
        raise KeyError("COUNTRY column is required for this indicator.")

    data = df[["COUNTRY"]].dropna(subset=["COUNTRY"]).copy()

    result = (
        data.groupby("COUNTRY", as_index=False)
        .size()
        .rename(columns={"size": "event_count"})
        .sort_values("event_count", ascending=False)
        .head(top_n)
    )
    return result


def cluster_tsunami_events(df: pd.DataFrame, n_clusters: int = 4) -> Dict[str, object]:
    """
    Input:
        df: cleaned tsunami DataFrame.
        n_clusters: number of clusters.
    Output:
        dictionary containing clustered DataFrame, cluster summary and features used.
    """
    feature_columns = ["LATITUDE", "LONGITUDE", "EQ_MAGNITUDE", "EQ_DEPTH", "TS_INTENSITY"]
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    cluster_data = df[
        feature_columns + [c for c in ["COUNTRY", "REGION", "LOCATION_NAME"] if c in df.columns]
    ].copy()
    for column in feature_columns:
        cluster_data[column] = pd.to_numeric(cluster_data[column], errors="coerce")

    cluster_data = cluster_data.dropna(subset=feature_columns).copy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_data[feature_columns])

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_data["tsunami_cluster"] = model.fit_predict(scaled_features)

    cluster_summary = cluster_data.groupby("tsunami_cluster")[feature_columns].mean().round(3)

    return {
        "clustered_df": cluster_data,
        "cluster_summary": cluster_summary,
        "features_used": feature_columns,
    }


def compute_events_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        df: cleaned tsunami DataFrame.
    Output:
        DataFrame containing number of tsunami events by year and rolling average.
    """
    if "YEAR" not in df.columns:
        raise KeyError("YEAR column is required for this indicator.")

    temporal_data = df[["YEAR"]].dropna(subset=["YEAR"]).copy()

    result = (
        temporal_data.groupby("YEAR", as_index=False)
        .size()
        .rename(columns={"size": "event_count"})
        .sort_values("YEAR")
    )

    if len(result) >= 10:
        result["rolling_average_10y"] = result["event_count"].rolling(window=10, min_periods=3).mean()
    else:
        result["rolling_average_10y"] = np.nan

    return result


def compute_spatial_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        df: cleaned tsunami DataFrame.
    Output:
        DataFrame containing valid latitude, longitude and event information.
    """
    required_columns = ["LATITUDE", "LONGITUDE"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    keep_columns = [
        column
        for column in [
            "LATITUDE",
            "LONGITUDE",
            "COUNTRY",
            "REGION",
            "LOCATION_NAME",
            "YEAR",
            "TS_INTENSITY",
            "EQ_MAGNITUDE",
        ]
        if column in df.columns
    ]
    spatial_data = df[keep_columns].copy()

    for column in ["LATITUDE", "LONGITUDE", "TS_INTENSITY", "EQ_MAGNITUDE"]:
        if column in spatial_data.columns:
            spatial_data[column] = pd.to_numeric(spatial_data[column], errors="coerce")

    spatial_data = spatial_data.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()
    return spatial_data


def build_key_findings(
    top_countries_df: pd.DataFrame,
    cluster_results: Dict[str, object],
    events_over_time_df: pd.DataFrame,
    spatial_df: pd.DataFrame,
) -> List[str]:
    """
    Input:
        top_countries_df, cluster_results, events_over_time_df, spatial_df: computed indicator results.
    Output:
        list of concise key findings for the dashboard.
    """
    findings = []

    top_rows = top_countries_df.head(3)
    top_countries_text = ", ".join(
        [f"{row.COUNTRY} ({int(row.event_count)})" for row in top_rows.itertuples()]
    )
    findings.append(
        f"Dans ce dataset, les pays les plus représentés sont {top_countries_text}, ce qui suggère une concentration géographique des événements enregistrés."
    )

    cluster_sizes = (
        cluster_results["clustered_df"]["tsunami_cluster"].value_counts().sort_index()
    )
    smallest_cluster = int(cluster_sizes.idxmin())
    smallest_size = int(cluster_sizes.min())
    findings.append(
        f"Le clustering révèle plusieurs profils d'événements, avec un cluster {smallest_cluster} très peu représenté ({smallest_size} événements), ce qui suggère un groupe atypique plutôt qu'un profil général."
    )

    year_min = int(events_over_time_df["YEAR"].min())
    year_max = int(events_over_time_df["YEAR"].max())
    top_years = events_over_time_df.sort_values("event_count", ascending=False).head(3)
    top_years_text = ", ".join(
        [f"{int(row.YEAR)} ({int(row.event_count)})" for row in top_years.itertuples()]
    )
    findings.append(
        f"La série temporelle couvre la période {year_min} à {year_max}, et les années {top_years_text} apparaissent parmi les plus fréquentes dans le corpus observé."
    )

    top_regions = spatial_df.groupby("REGION").size().sort_values(ascending=False).head(3)
    top_regions_text = ", ".join([f"{idx} ({int(val)})" for idx, val in top_regions.items()])
    findings.append(
        f"La carte mondiale suggère une concentration des événements dans plusieurs zones récurrentes, notamment {top_regions_text}."
    )

    return findings


def format_number(value: float, decimals: int = 0) -> str:
    """
    Input:
        value: numeric value.
        decimals: number of decimals for regular formatting.
    Output:
        formatted string for dashboard display.
    """
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"
    if decimals > 0:
        return f"{value:.{decimals}f}"
    return f"{value:,.0f}"


def kpi_card(title: str, value: str, accent_color: str) -> html.Div:
    """
    Input:
        title: KPI title.
        value: formatted KPI value.
        accent_color: card accent color.
    Output:
        Dash card component.
    """
    return html.Div(
        [
            html.Div(title, className="kpi-title"),
            html.Div(value, className="kpi-value", style={"color": accent_color}),
        ],
        className="kpi-card",
    )


def build_figures(
    top_countries_df: pd.DataFrame,
    cluster_results: Dict[str, object],
    events_over_time_df: pd.DataFrame,
    spatial_df: pd.DataFrame,
    top_n: int = 15,
    cluster_filter: Optional[int] = None,
) -> Dict[str, go.Figure]:
    """
    Input:
        top_countries_df, cluster_results, events_over_time_df, spatial_df: computed results.
        top_n: number of countries displayed.
        cluster_filter: optional cluster selection.
    Output:
        dictionary of Plotly figures used in the dashboard.
    """
    countries_for_chart = top_countries_df.head(top_n).sort_values("event_count", ascending=True)

    fig_top_countries = px.bar(
        countries_for_chart,
        x="event_count",
        y="COUNTRY",
        orientation="h",
        title="Indicator 1 — Countries with the highest number of tsunami events",
        labels={"event_count": "Nombre d'événements", "COUNTRY": "Pays"},
        color_discrete_sequence=[ACCENT_PRIMARY],
    )
    fig_top_countries.update_layout(
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        font_color=TEXT_COLOR,
        margin=dict(l=20, r=20, t=60, b=20),
        height=600,
    )

    clustered_df = cluster_results["clustered_df"]
    if cluster_filter is None:
        scatter_df = clustered_df.copy()
    else:
        scatter_df = clustered_df[clustered_df["tsunami_cluster"] == cluster_filter].copy()

    fig_cluster_scatter = px.scatter(
        scatter_df,
        x="LONGITUDE",
        y="LATITUDE",
        color=scatter_df["tsunami_cluster"].astype(str),
        hover_data={
            "COUNTRY": True if "COUNTRY" in scatter_df.columns else False,
            "REGION": True if "REGION" in scatter_df.columns else False,
            "LOCATION_NAME": True if "LOCATION_NAME" in scatter_df.columns else False,
            "EQ_MAGNITUDE": True,
            "EQ_DEPTH": True,
            "TS_INTENSITY": True,
        },
        title="Indicator 2 — Tsunami event clustering",
        labels={"LONGITUDE": "Longitude", "LATITUDE": "Latitude", "color": "Cluster"},
        render_mode="svg",
        color_discrete_sequence=[ACCENT_PRIMARY, ACCENT_SECONDARY, ACCENT_DANGER, "#90E0EF"],
    )
    fig_cluster_scatter.update_layout(
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        font_color=TEXT_COLOR,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    cluster_heatmap_df = cluster_results["cluster_summary"]
    fig_cluster_heatmap = px.imshow(
        cluster_heatmap_df,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale="Blues",
        title="Moyennes des variables par cluster",
    )
    fig_cluster_heatmap.update_layout(
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        font_color=TEXT_COLOR,
        margin=dict(l=20, r=20, t=70, b=20),
        title_font_size=22,
        height=520,
        coloraxis_colorbar=dict(title="Valeur"),
    )
    fig_cluster_heatmap.update_xaxes(title_font_size=15, tickfont=dict(size=12))
    fig_cluster_heatmap.update_yaxes(title_font_size=15, tickfont=dict(size=12))
    fig_cluster_heatmap.update_traces(textfont={"size": 12})

    fig_time = go.Figure()
    fig_time.add_trace(
        go.Scatter(
            x=events_over_time_df["YEAR"],
            y=events_over_time_df["event_count"],
            mode="lines",
            name="Nombre annuel d'événements",
            line=dict(color=ACCENT_PRIMARY, width=2),
        )
    )
    if events_over_time_df["rolling_average_10y"].notna().sum() > 0:
        fig_time.add_trace(
            go.Scatter(
                x=events_over_time_df["YEAR"],
                y=events_over_time_df["rolling_average_10y"],
                mode="lines",
                name="Moyenne glissante sur 10 ans",
                line=dict(color=ACCENT_DANGER, width=3),
            )
        )
    fig_time.update_layout(
        title="Indicator 3 — Evolution of tsunami events over time",
        xaxis_title="Année",
        yaxis_title="Nombre d'événements",
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        font_color=TEXT_COLOR,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    color_argument = "TS_INTENSITY" if "TS_INTENSITY" in spatial_df.columns else "REGION"
    hover_data = {
        "COUNTRY": True if "COUNTRY" in spatial_df.columns else False,
        "REGION": True if "REGION" in spatial_df.columns else False,
        "YEAR": True if "YEAR" in spatial_df.columns else False,
        "EQ_MAGNITUDE": True if "EQ_MAGNITUDE" in spatial_df.columns else False,
        "TS_INTENSITY": True if "TS_INTENSITY" in spatial_df.columns else False,
    }

    fig_spatial = px.scatter_geo(
        spatial_df,
        lat="LATITUDE",
        lon="LONGITUDE",
        color=color_argument,
        hover_name="LOCATION_NAME" if "LOCATION_NAME" in spatial_df.columns else None,
        hover_data=hover_data,
        title="Indicator 4 — Geographic distribution of tsunami events",
        projection="natural earth",
        color_continuous_scale="Blues",
    )
    fig_spatial.update_layout(
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        font_color=TEXT_COLOR,
        margin=dict(l=20, r=20, t=60, b=20),
        geo=dict(
            bgcolor=CARD_COLOR,
            showland=True,
            landcolor="#15263F",
            showocean=True,
            oceancolor="#09172A",
            showcountries=True,
            countrycolor="#29456C",
        ),
        height=650,
    )

    return {
        "top_countries": fig_top_countries,
        "cluster_scatter": fig_cluster_scatter,
        "cluster_heatmap": fig_cluster_heatmap,
        "time_series": fig_time,
        "spatial_map": fig_spatial,
    }


def build_dash_app(df: pd.DataFrame) -> Dash:
    """
    Input:
        df: raw tsunami DataFrame.
    Output:
        fully configured Dash app.
    """
    app = Dash(__name__)
    app.title = PROJECT_TITLE
    app.index_string = f"""
    <!DOCTYPE html>
    <html>
        <head>
            {{%metas%}}
            <title>{{%title%}}</title>
            {{%favicon%}}
            {{%css%}}
            <style>
                body {{ margin: 0; font-family: 'Segoe UI', Arial, sans-serif; background: {BG_COLOR}; color: {TEXT_COLOR}; }}
                .page {{ min-height: 100vh; padding: 24px; background: radial-gradient(circle at top, #10243f 0%, {BG_COLOR} 55%); }}
                .header {{ background: linear-gradient(135deg, {CARD_COLOR} 0%, {CARD_ALT} 100%); border: 1px solid {BORDER_COLOR}; border-radius: 20px; padding: 28px; box-shadow: 0 12px 28px rgba(0,0,0,0.24); margin-bottom: 24px; }}
                .header h1 {{ margin: 0 0 10px 0; font-size: 34px; color: {ACCENT_SECONDARY}; }}
                .header p {{ margin: 6px 0; color: {MUTED_TEXT}; line-height: 1.6; }}
                .header a {{ color: {ACCENT_SECONDARY}; text-decoration: none; }}
                .header a:hover {{ text-decoration: underline; }}
                .section-card {{ background: {CARD_COLOR}; border: 1px solid {BORDER_COLOR}; border-radius: 18px; padding: 22px; margin-bottom: 24px; box-shadow: 0 10px 24px rgba(0,0,0,0.18); }}
                .section-card h2 {{ margin-top: 0; color: {TEXT_COLOR}; }}
                .section-card p {{ color: {MUTED_TEXT}; line-height: 1.6; }}
                .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-top: 18px; }}
                .kpi-card {{ background: {CARD_ALT}; border: 1px solid {BORDER_COLOR}; border-radius: 16px; padding: 18px; }}
                .kpi-title {{ color: {MUTED_TEXT}; font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px; }}
                .kpi-value {{ font-size: 30px; font-weight: 700; }}
                .controls-row {{ display: flex; gap: 18px; flex-wrap: wrap; margin-bottom: 18px; }}
                .control-block {{ min-width: 220px; }}
                .control-block label {{ display: block; margin-bottom: 8px; color: {MUTED_TEXT}; }}
                .graph-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; }}
                .full-width {{ grid-column: 1 / -1; }}
                .bullet-list {{ color: {MUTED_TEXT}; line-height: 1.8; padding-left: 18px; }}
            </style>
        </head>
        <body>
            {{%app_entry%}}
            <footer>
                {{%config%}}
                {{%scripts%}}
                {{%renderer%}}
            </footer>
        </body>
    </html>
    """

    project_dir = Path(__file__).resolve().parent
    cleaned_df = preprocess_tsunami_data(df)
    kpis = compute_overview_kpis(cleaned_df)
    default_top_countries = compute_top_countries_by_events(cleaned_df, top_n=15)
    cluster_results = cluster_tsunami_events(cleaned_df, n_clusters=4)
    events_over_time_df = compute_events_over_time(cleaned_df)
    spatial_df = compute_spatial_distribution(cleaned_df)
    key_findings = build_key_findings(
        default_top_countries, cluster_results, events_over_time_df, spatial_df
    )
    figures = build_figures(
        default_top_countries,
        cluster_results,
        events_over_time_df,
        spatial_df,
        top_n=15,
        cluster_filter=None,
    )

    cluster_options = [{"label": "Tous les clusters", "value": "all"}] + [
        {"label": f"Cluster {cluster_id}", "value": int(cluster_id)}
        for cluster_id in sorted(cluster_results["clustered_df"]["tsunami_cluster"].unique())
    ]

    app.layout = html.Div(
        className="page",
        children=[
            html.Div(
                className="header",
                children=[
                    html.H1(PROJECT_TITLE),
                    html.P(PROJECT_SUBTITLE),
                    html.P(f"Dataset : {DATASET_NAME}"),
                    html.P(f"Source : {DATASET_SOURCE}"),
                    html.P(f"Équipe : {TEAM_MEMBERS}"),
                    html.P(["GitHub : ", html.A(GITHUB_LINK, href=GITHUB_LINK, target="_blank")]),
                ],
            ),
            html.Div(
                className="section-card",
                children=[
                    html.H2("Overview"),
                    html.Div(
                        className="kpi-grid",
                        children=[
                            kpi_card("Nombre total d'événements", format_number(kpis["total_events"]), ACCENT_SECONDARY),
                            kpi_card("Nombre de pays", format_number(kpis["total_countries"]), ACCENT_PRIMARY),
                            kpi_card("Année min", format_number(kpis["year_min"]), ACCENT_PRIMARY),
                            kpi_card("Année max", format_number(kpis["year_max"]), ACCENT_PRIMARY),
                            kpi_card("Magnitude moyenne", format_number(kpis["average_magnitude"], decimals=2), ACCENT_DANGER),
                            kpi_card("Profondeur moyenne", format_number(kpis["average_depth"], decimals=2), ACCENT_DANGER),
                            kpi_card("Nombre total de décès", format_number(kpis["total_deaths"]), ACCENT_DANGER),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="section-card",
                children=[
                    html.H2("Indicator 1 — Countries with the highest number of tsunami events"),
                    html.P("Cet indicateur présente les pays les plus représentés dans le dataset."),
                    html.Div(
                        className="controls-row",
                        children=[
                            html.Div(
                                className="control-block",
                                children=[
                                    html.Label("Nombre de pays affichés"),
                                    dcc.Dropdown(
                                        id="top-n-dropdown",
                                        options=[
                                            {"label": "Top 10", "value": 10},
                                            {"label": "Top 15", "value": 15},
                                            {"label": "Top 20", "value": 20},
                                        ],
                                        value=15,
                                        clearable=False,
                                    ),
                                ],
                            )
                        ],
                    ),
                    dcc.Graph(
                        id="top-countries-graph",
                        figure=figures["top_countries"],
                        config={"displayModeBar": False},
                    ),
                ],
            ),
            html.Div(
                className="section-card",
                children=[
                    html.H2("Indicator 2 — Tsunami event clustering"),
                    html.P("Le clustering identifie des profils similaires d'événements."),
                    html.Div(
                        className="controls-row",
                        children=[
                            html.Div(
                                className="control-block",
                                children=[
                                    html.Label("Filtre de cluster"),
                                    dcc.Dropdown(
                                        id="cluster-filter-dropdown",
                                        options=cluster_options,
                                        value="all",
                                        clearable=False,
                                    ),
                                ],
                            )
                        ],
                    ),
                    html.Div(
                        className="graph-grid",
                        children=[
                            dcc.Graph(
                                id="cluster-scatter-graph",
                                figure=figures["cluster_scatter"],
                                config={"displayModeBar": False},
                            ),
                            dcc.Graph(
                                id="cluster-heatmap-graph",
                                figure=figures["cluster_heatmap"],
                                config={"displayModeBar": False},
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="section-card",
                children=[
                    html.H2("Indicator 3 — Evolution of tsunami events over time"),
                    html.P("Cet indicateur étudie la distribution temporelle des événements."),
                    dcc.Graph(
                        id="time-series-graph",
                        figure=figures["time_series"],
                        config={"displayModeBar": False},
                    ),
                ],
            ),
            html.Div(
                className="section-card",
                children=[
                    html.H2("Indicator 4 — Geographic distribution of tsunami events"),
                    html.P("Cet indicateur présente la répartition géographique des événements."),
                    dcc.Graph(
                        id="spatial-map-graph",
                        figure=figures["spatial_map"],
                        config={"displayModeBar": False},
                    ),
                ],
            ),
            html.Div(
                className="section-card",
                children=[
                    html.H2("Methodology"),
                    html.Ul(
                        className="bullet-list",
                        children=[
                            html.Li("KDD process pour structurer l'analyse globale du projet."),
                            html.Li("Chargement du CSV tsunami_dataset.csv depuis le dossier du projet."),
                            html.Li("Exploration descriptive du dataset et identification des variables clés."),
                            html.Li("Prétraitement local des valeurs manquantes selon les besoins de chaque indicateur."),
                            html.Li("Construction de quatre indicateurs couvrant les dimensions territoriale, physique, temporelle et spatiale."),
                            html.Li("Visualisation synthétique des résultats dans un dashboard interactif."),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="section-card",
                children=[
                    html.H2("Limitations"),
                    html.Ul(
                        className="bullet-list",
                        children=[
                            html.Li("Certaines données historiques sont incomplètes."),
                            html.Li("Certaines coordonnées sont manquantes."),
                            html.Li("Les impacts déclarés dépendent des sources historiques."),
                            html.Li("Les analyses restent descriptives."),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="section-card",
                children=[
                    html.H2("Key findings"),
                    html.Ul(
                        className="bullet-list",
                        children=[html.Li(text) for text in key_findings],
                    ),
                ],
            ),
        ],
    )

    @app.callback(Output("top-countries-graph", "figure"), Input("top-n-dropdown", "value"))
    def update_top_countries(top_n: int) -> go.Figure:
        updated_top_countries = compute_top_countries_by_events(cleaned_df, top_n=top_n)
        return build_figures(
            updated_top_countries,
            cluster_results,
            events_over_time_df,
            spatial_df,
            top_n=top_n,
            cluster_filter=None,
        )["top_countries"]

    @app.callback(Output("cluster-scatter-graph", "figure"), Input("cluster-filter-dropdown", "value"))
    def update_cluster_scatter(cluster_value: str) -> go.Figure:
        selected_cluster = None if cluster_value == "all" else int(cluster_value)
        return build_figures(
            default_top_countries,
            cluster_results,
            events_over_time_df,
            spatial_df,
            top_n=15,
            cluster_filter=selected_cluster,
        )["cluster_scatter"]

    return app


def export_dashboard_html(
    kpis: Dict[str, object],
    figures: Dict[str, go.Figure],
    key_findings: List[str],
    html_output_path: Path,
) -> None:
    """
    Input:
        KPI values, generated figures, key findings and output path.
    Output:
        static HTML export written to disk.
    """
    kpi_cards_html = f"""
    <div class="kpi-grid">
      <div class="kpi-card"><div class="kpi-title">Nombre total d'événements</div><div class="kpi-value" style="color:{ACCENT_SECONDARY};">{format_number(kpis['total_events'])}</div></div>
      <div class="kpi-card"><div class="kpi-title">Nombre de pays</div><div class="kpi-value" style="color:{ACCENT_PRIMARY};">{format_number(kpis['total_countries'])}</div></div>
      <div class="kpi-card"><div class="kpi-title">Année min</div><div class="kpi-value" style="color:{ACCENT_PRIMARY};">{format_number(kpis['year_min'])}</div></div>
      <div class="kpi-card"><div class="kpi-title">Année max</div><div class="kpi-value" style="color:{ACCENT_PRIMARY};">{format_number(kpis['year_max'])}</div></div>
      <div class="kpi-card"><div class="kpi-title">Magnitude moyenne</div><div class="kpi-value" style="color:{ACCENT_DANGER};">{format_number(kpis['average_magnitude'], decimals=2)}</div></div>
      <div class="kpi-card"><div class="kpi-title">Profondeur moyenne</div><div class="kpi-value" style="color:{ACCENT_DANGER};">{format_number(kpis['average_depth'], decimals=2)}</div></div>
      <div class="kpi-card"><div class="kpi-title">Nombre total de décès</div><div class="kpi-value" style="color:{ACCENT_DANGER};">{format_number(kpis['total_deaths'])}</div></div>
    </div>
    """

    key_findings_html = "".join([f"<li>{finding}</li>" for finding in key_findings])

    html_content = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>{PROJECT_TITLE}</title>
      <style>
        body {{ margin: 0; font-family: 'Segoe UI', Arial, sans-serif; background: {BG_COLOR}; color: {TEXT_COLOR}; }}
        .page {{ min-height: 100vh; padding: 24px; background: radial-gradient(circle at top, #10243f 0%, {BG_COLOR} 55%); }}
        .header {{ background: linear-gradient(135deg, {CARD_COLOR} 0%, {CARD_ALT} 100%); border: 1px solid {BORDER_COLOR}; border-radius: 20px; padding: 28px; box-shadow: 0 12px 28px rgba(0,0,0,0.24); margin-bottom: 24px; }}
        .header h1 {{ margin: 0 0 10px 0; color: {ACCENT_SECONDARY}; font-size: 34px; }}
        .header p, .section-card p, .section-card li {{ color: {MUTED_TEXT}; line-height: 1.6; }}
        .header a {{ color: {ACCENT_SECONDARY}; text-decoration: none; }}
        .section-card {{ background: {CARD_COLOR}; border: 1px solid {BORDER_COLOR}; border-radius: 18px; padding: 22px; margin-bottom: 24px; box-shadow: 0 10px 24px rgba(0,0,0,0.18); }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-top: 18px; }}
        .kpi-card {{ background: {CARD_ALT}; border: 1px solid {BORDER_COLOR}; border-radius: 16px; padding: 18px; }}
        .kpi-title {{ color: {MUTED_TEXT}; font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px; }}
        .kpi-value {{ font-size: 30px; font-weight: 700; }}
        .grid-two {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; }}
        h2 {{ margin-top: 0; }}
      </style>
    </head>
    <body>
      <div class="page">
        <div class="header">
          <h1>{PROJECT_TITLE}</h1>
          <p>{PROJECT_SUBTITLE}</p>
          <p>Dataset : {DATASET_NAME}</p>
          <p>Source : {DATASET_SOURCE}</p>
          <p>Équipe : {TEAM_MEMBERS}</p>
          <p>GitHub : <a href="{GITHUB_LINK}" target="_blank">{GITHUB_LINK}</a></p>
        </div>

        <div class="section-card">
          <h2>Overview</h2>
          {kpi_cards_html}
        </div>

        <div class="section-card">
          <h2>Indicator 1 — Countries with the highest number of tsunami events</h2>
          <p>Cet indicateur présente les pays les plus représentés dans le dataset.</p>
          {figures['top_countries'].to_html(full_html=False, include_plotlyjs='cdn')}
        </div>

        <div class="section-card">
          <h2>Indicator 2 — Tsunami event clustering</h2>
          <p>Le clustering identifie des profils similaires d'événements.</p>
          <div class="grid-two">
            <div>{figures['cluster_scatter'].to_html(full_html=False, include_plotlyjs=False)}</div>
            <div>{figures['cluster_heatmap'].to_html(full_html=False, include_plotlyjs=False)}</div>
          </div>
        </div>

        <div class="section-card">
          <h2>Indicator 3 — Evolution of tsunami events over time</h2>
          <p>Cet indicateur étudie la distribution temporelle des événements.</p>
          {figures['time_series'].to_html(full_html=False, include_plotlyjs=False)}
        </div>

        <div class="section-card">
          <h2>Indicator 4 — Geographic distribution of tsunami events</h2>
          <p>Cet indicateur présente la répartition géographique des événements.</p>
          {figures['spatial_map'].to_html(full_html=False, include_plotlyjs=False)}
        </div>

        <div class="section-card">
          <h2>Methodology</h2>
          <ul>
            <li>KDD process</li>
            <li>Chargement CSV</li>
            <li>Exploration</li>
            <li>Prétraitement local</li>
            <li>Construction des indicateurs</li>
            <li>Visualisation</li>
          </ul>
        </div>

        <div class="section-card">
          <h2>Limitations</h2>
          <ul>
            <li>Certaines données historiques sont incomplètes.</li>
            <li>Certaines coordonnées sont manquantes.</li>
            <li>Les impacts déclarés dépendent des sources historiques.</li>
            <li>Les analyses restent descriptives.</li>
          </ul>
        </div>

        <div class="section-card">
          <h2>Key findings</h2>
          <ul>
            {key_findings_html}
          </ul>
        </div>
      </div>
    </body>
    </html>
    """

    html_output_path.write_text(html_content, encoding="utf-8")


def main() -> None:
    """
    Input:
        None.
    Output:
        launches the dashboard and writes the HTML export.
    """
    project_dir = Path(__file__).resolve().parent
    csv_files = find_csv_files(project_dir)
    if not csv_files:
        raise FileNotFoundError("Aucun fichier CSV n'a été trouvé dans le dossier du projet.")

    selected_csv_path = select_tsunami_csv(csv_files)
    raw_df = load_data(selected_csv_path)
    cleaned_df = preprocess_tsunami_data(raw_df)

    kpis = compute_overview_kpis(cleaned_df)
    top_countries_df = compute_top_countries_by_events(cleaned_df, top_n=15)
    cluster_results = cluster_tsunami_events(cleaned_df, n_clusters=4)
    events_over_time_df = compute_events_over_time(cleaned_df)
    spatial_df = compute_spatial_distribution(cleaned_df)
    key_findings = build_key_findings(
        top_countries_df, cluster_results, events_over_time_df, spatial_df
    )

    figures = build_figures(
        top_countries_df,
        cluster_results,
        events_over_time_df,
        spatial_df,
        top_n=15,
        cluster_filter=None,
    )

    export_dashboard_html(kpis, figures, key_findings, project_dir / HTML_EXPORT_NAME)
    app = build_dash_app(raw_df)
    app.run(debug=False)


if __name__ == "__main__":
    main()
