"""Geographic visualizations for tariff data."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from typing import Dict, List, Any, Optional, Union, Tuple
import io
import base64

from ..utils import logger, config


class GeoVisualizer:
    """
    Geographic visualizations for tariff data.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        country_coords_file: Optional[str] = None,
    ):
        """
        Initialize the geographic visualizer.

        Args:
            data: DataFrame or dictionary with tariff data
            country_coords_file: Path to CSV file with country coordinates (optional)
        """
        if isinstance(data, pd.DataFrame):
            self.df = data
        elif isinstance(data, dict) and "events" in data:
            # Convert events to DataFrame
            from ..preprocessing.processor import TariffProcessor

            processor = TariffProcessor()
            self.df = processor.to_dataframe(data)
        else:
            raise ValueError(
                "Data must be a DataFrame or a dictionary with 'events' key"
            )

        # Load country coordinates
        if country_coords_file and os.path.exists(country_coords_file):
            self.country_coords = self._load_country_coords(country_coords_file)
        else:
            # Try to find default file
            default_file = os.path.join(
                "data", "reference", "country_codes_iso_3166_1_alpha_2_code.csv"
            )
            if os.path.exists(default_file):
                self.country_coords = self._load_country_coords(default_file)
            else:
                logger.warning(
                    "Country coordinates file not found. Map visualizations will be limited."
                )
                self.country_coords = {}

        # Set style
        self._set_plot_style()

    def _set_plot_style(self) -> None:
        """Set the default plot style for all visualizations."""
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "white"

    def _load_country_coords(self, file_path: str) -> Dict[str, Dict[str, float]]:
        """
        Load country coordinates from CSV file.

        Args:
            file_path: Path to CSV file with country coordinates

        Returns:
            Dictionary mapping country codes to coordinates
        """
        try:
            country_df = pd.read_csv(file_path)

            # Check for required columns
            required_cols = [
                "Alpha-2 code",
                "Latitude (average)",
                "Longitude (average)",
            ]
            if not all(col in country_df.columns for col in required_cols):
                logger.warning(f"Missing required columns in {file_path}")
                return {}

            # Create mapping
            coords = {}
            for _, row in country_df.iterrows():
                try:
                    code = row["Alpha-2 code"].strip()
                    lat = float(row["Latitude (average)"])
                    lon = float(row["Longitude (average)"])
                    coords[code] = {"lat": lat, "lon": lon}
                except (ValueError, TypeError) as e:
                    logger.debug(
                        f"Error parsing coordinates for {row['Alpha-2 code']}: {e}"
                    )

            logger.info(f"Loaded coordinates for {len(coords)} countries")
            return coords

        except Exception as e:
            logger.error(f"Error loading country coordinates: {e}")
            return {}

    def plot_tariff_world_map(self, figsize: Tuple[int, int] = (15, 10)) -> str:
        """
        Plot world map with countries colored by tariff count.

        Args:
            figsize: Figure size (width, height) in inches

        Returns:
            Base64 encoded PNG image
        """
        if not self.country_coords or "imposing_country_code" not in self.df.columns:
            return ""

        # Count tariffs by imposing country
        country_counts = self.df["imposing_country_code"].value_counts().to_dict()

        # Prepare data for plotting
        countries = []
        lats = []
        lons = []
        counts = []

        for code, count in country_counts.items():
            if code in self.country_coords:
                countries.append(code)
                lats.append(self.country_coords[code]["lat"])
                lons.append(self.country_coords[code]["lon"])
                counts.append(count)

        if not countries:
            return ""

        # Create the figure
        plt.figure(figsize=figsize)

        # Set the map projection
        projection = config.get("visualization.map_projection", "mollweide")
        ax = plt.axes(projection=projection)

        # Add coastlines and country borders
        ax.coastlines()

        # Create color map
        norm = Normalize(vmin=min(counts), vmax=max(counts))
        cmap = plt.cm.get_cmap("viridis")

        # Plot points with size and color based on count
        sc = ax.scatter(
            lons,
            lats,
            transform=plt.ccrs.PlateCarree(),
            s=[max(20, c * 5) for c in counts],  # Size based on count
            c=counts,  # Color based on count
            cmap=cmap,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )

        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label("Number of Tariff Measures")

        # Add title
        plt.title("Countries Imposing Tariffs", fontsize=15)

        # Convert plot to base64 string
        img = self._fig_to_base64(plt.gcf())
        plt.close()

        return img

    def plot_tariff_flow_map(
        self, figsize: Tuple[int, int] = (15, 10), min_count: int = 2
    ) -> str:
        """
        Plot world map with arrows showing tariff flows between countries.

        Args:
            figsize: Figure size (width, height) in inches
            min_count: Minimum count threshold for showing flows

        Returns:
            Base64 encoded PNG image
        """
        if not self.country_coords:
            return ""

        if (
            "imposing_country_code" not in self.df.columns
            or "targeted_country_codes" not in self.df.columns
        ):
            return ""

        # Count flows between countries
        flows = {}

        for _, row in self.df.iterrows():
            source = row.get("imposing_country_code")

            # Handle targets as a list or semicolon-separated string
            targets = row.get("targeted_country_codes", "")
            if isinstance(targets, str):
                targets = [t.strip() for t in targets.split(";") if t.strip()]

            for target in targets:
                if (
                    source
                    and target
                    and source in self.country_coords
                    and target in self.country_coords
                ):
                    flow_key = f"{source}_{target}"
                    flows[flow_key] = flows.get(flow_key, 0) + 1

        # Filter flows by minimum count
        filtered_flows = {k: v for k, v in flows.items() if v >= min_count}

        if not filtered_flows:
            return ""

        # Create the figure
        plt.figure(figsize=figsize)

        # Set the map projection
        projection = config.get("visualization.map_projection", "mollweide")
        ax = plt.axes(projection=projection)

        # Add coastlines and country borders
        ax.coastlines()

        # Create color map
        flow_counts = list(filtered_flows.values())
        norm = Normalize(vmin=min(flow_counts), vmax=max(flow_counts))
        cmap = plt.cm.get_cmap("plasma")

        # Plot background points for all countries
        all_countries = set(self.country_coords.keys())
        background_lats = [self.country_coords[c]["lat"] for c in all_countries]
        background_lons = [self.country_coords[c]["lon"] for c in all_countries]

        ax.scatter(
            background_lons,
            background_lats,
            transform=plt.ccrs.PlateCarree(),
            s=10,
            c="lightgray",
            alpha=0.3,
            edgecolors="none",
        )

        # Plot flows
        max_line_width = 5
        min_line_width = 1
        max_count = max(flow_counts)
        min_count = min(flow_counts)

        for flow_key, count in filtered_flows.items():
            source, target = flow_key.split("_")

            # Get coordinates
            start_lon = self.country_coords[source]["lon"]
            start_lat = self.country_coords[source]["lat"]
            end_lon = self.country_coords[target]["lon"]
            end_lat = self.country_coords[target]["lat"]

            # Calculate line width based on count
            if max_count > min_count:
                line_width = min_line_width + (
                    (count - min_count) / (max_count - min_count)
                ) * (max_line_width - min_line_width)
            else:
                line_width = min_line_width

            # Calculate intermediate point for curved line
            # Use a great circle path approximation
            intermediate_lon = (start_lon + end_lon) / 2
            intermediate_lat = (start_lat + end_lat) / 2

            # Add a slight curve by moving intermediate point
            displacement = 10  # Adjust for desired curve amount
            dx = end_lon - start_lon
            dy = end_lat - start_lat
            intermediate_lon += -dy / (displacement)
            intermediate_lat += dx / (displacement)

            # Create bezier curve using three points
            path_points = np.array(
                [
                    [start_lon, start_lat],
                    [intermediate_lon, intermediate_lat],
                    [end_lon, end_lat],
                ]
            )

            # Plot the path
            color = cmap(norm(count))
            ax.plot(
                path_points[:, 0],
                path_points[:, 1],
                transform=plt.ccrs.Geodetic(),
                color=color,
                linewidth=line_width,
                alpha=0.7,
                zorder=count,  # Higher counts appear on top
            )

            # Add arrowhead
            arrow_size = line_width * 2
            ax.scatter(
                end_lon,
                end_lat,
                transform=plt.ccrs.PlateCarree(),
                s=arrow_size * 10,
                color=color,
                marker=">",
                alpha=0.8,
                zorder=count + 100,  # Ensure arrows are on top
            )

        # Plot highlighted points for countries in flows
        flow_countries = set()
        for flow_key in filtered_flows:
            source, target = flow_key.split("_")
            flow_countries.add(source)
            flow_countries.add(target)

        flow_lats = [self.country_coords[c]["lat"] for c in flow_countries]
        flow_lons = [self.country_coords[c]["lon"] for c in flow_countries]

        ax.scatter(
            flow_lons,
            flow_lats,
            transform=plt.ccrs.PlateCarree(),
            s=30,
            c="white",
            alpha=1.0,
            edgecolors="black",
            linewidths=0.5,
            zorder=1000,  # Ensure these are on top
        )

        # Create a manually constructed legend
        from matplotlib.patches import Circle
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                color=cmap(norm(min_count)),
                linewidth=min_line_width,
                alpha=0.7,
                label=f"{min_count} events",
            ),
            Line2D(
                [0],
                [0],
                color=cmap(norm((max_count + min_count) // 2)),
                linewidth=(max_line_width + min_line_width) / 2,
                alpha=0.7,
                label=f"{(max_count + min_count) // 2} events",
            ),
            Line2D(
                [0],
                [0],
                color=cmap(norm(max_count)),
                linewidth=max_line_width,
                alpha=0.7,
                label=f"{max_count} events",
            ),
        ]

        ax.legend(
            handles=legend_elements, loc="lower right", title="Tariff Flow Intensity"
        )

        # Add title
        plt.title("Tariff Flows Between Countries", fontsize=15)

        # Convert plot to base64 string
        img = self._fig_to_base64(plt.gcf())
        plt.close()

        return img

    def plot_targeted_countries_map(self, figsize: Tuple[int, int] = (15, 10)) -> str:
        """
        Plot world map with countries colored by how often they are targeted.

        Args:
            figsize: Figure size (width, height) in inches

        Returns:
            Base64 encoded PNG image
        """
        if not self.country_coords or "targeted_country_codes" not in self.df.columns:
            return ""

        # Count tariffs targeting each country
        target_counts = {}

        for _, row in self.df.iterrows():
            # Handle targets as a list or semicolon-separated string
            targets = row.get("targeted_country_codes", "")
            if isinstance(targets, str):
                targets = [t.strip() for t in targets.split(";") if t.strip()]

            for target in targets:
                if target in self.country_coords:
                    target_counts[target] = target_counts.get(target, 0) + 1

        if not target_counts:
            return ""

        # Prepare data for plotting
        countries = []
        lats = []
        lons = []
        counts = []

        for code, count in target_counts.items():
            if code in self.country_coords:
                countries.append(code)
                lats.append(self.country_coords[code]["lat"])
                lons.append(self.country_coords[code]["lon"])
                counts.append(count)

        # Create the figure
        plt.figure(figsize=figsize)

        # Set the map projection
        projection = config.get("visualization.map_projection", "mollweide")
        ax = plt.axes(projection=projection)

        # Add coastlines and country borders
        ax.coastlines()

        # Create color map
        norm = Normalize(vmin=min(counts), vmax=max(counts))
        cmap = plt.cm.get_cmap("YlOrRd")

        # Plot points with size and color based on count
        sc = ax.scatter(
            lons,
            lats,
            transform=plt.ccrs.PlateCarree(),
            s=[max(20, c * 5) for c in counts],  # Size based on count
            c=counts,  # Color based on count
            cmap=cmap,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )

        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label("Number of Times Targeted")

        # Add title
        plt.title("Countries Targeted by Tariffs", fontsize=15)

        # Convert plot to base64 string
        img = self._fig_to_base64(plt.gcf())
        plt.close()

        return img

    def plot_regional_tariff_intensity(self, figsize: Tuple[int, int] = (12, 8)) -> str:
        """
        Plot a bar chart showing tariff intensity by region.

        Args:
            figsize: Figure size (width, height) in inches

        Returns:
            Base64 encoded PNG image
        """
        if "imposing_country_code" not in self.df.columns:
            return ""

        # Define regions
        regions = {
            "North America": ["US", "CA", "MX"],
            "Europe": [
                "AT",
                "BE",
                "BG",
                "HR",
                "CY",
                "CZ",
                "DK",
                "EE",
                "FI",
                "FR",
                "DE",
                "GR",
                "HU",
                "IE",
                "IT",
                "LV",
                "LT",
                "LU",
                "MT",
                "NL",
                "PL",
                "PT",
                "RO",
                "SK",
                "SI",
                "ES",
                "SE",
                "GB",
                "CH",
                "NO",
                "EU",
            ],
            "Asia": ["CN", "JP", "KR", "IN", "ID", "SG", "MY", "TH", "VN", "PH"],
            "Middle East": [
                "AE",
                "SA",
                "QA",
                "OM",
                "KW",
                "BH",
                "IR",
                "IQ",
                "IL",
                "JO",
                "SY",
                "LB",
            ],
            "Latin America": [
                "BR",
                "AR",
                "CO",
                "CL",
                "PE",
                "VE",
                "EC",
                "BO",
                "PY",
                "UY",
                "CR",
                "PA",
            ],
            "Africa": [
                "ZA",
                "NG",
                "EG",
                "MA",
                "DZ",
                "TN",
                "KE",
                "ET",
                "GH",
                "CI",
                "TZ",
            ],
            "Oceania": ["AU", "NZ", "PG", "FJ"],
        }

        # Count tariffs by region for imposing countries
        imposing_region_counts = {region: 0 for region in regions}

        for _, row in self.df.iterrows():
            imposer = row.get("imposing_country_code")
            if imposer:
                for region, countries in regions.items():
                    if imposer in countries:
                        imposing_region_counts[region] += 1
                        break

        # Count tariffs by region for targeted countries
        targeted_region_counts = {region: 0 for region in regions}

        for _, row in self.df.iterrows():
            targets = row.get("targeted_country_codes", "")
            if isinstance(targets, str):
                targets = [t.strip() for t in targets.split(";") if t.strip()]

            for target in targets:
                for region, countries in regions.items():
                    if target in countries:
                        targeted_region_counts[region] += 1
                        break

        # Prepare data for plotting
        region_names = list(regions.keys())
        imposing_counts = [imposing_region_counts[r] for r in region_names]
        targeted_counts = [targeted_region_counts[r] for r in region_names]

        # Create figure
        plt.figure(figsize=figsize)

        # Set bar width and positions
        width = 0.35
        x = np.arange(len(region_names))

        # Plot bars
        plt.bar(
            x - width / 2, imposing_counts, width, label="Imposing", color="steelblue"
        )
        plt.bar(
            x + width / 2, targeted_counts, width, label="Targeted", color="indianred"
        )

        # Add labels and title
        plt.xlabel("Region", fontsize=12)
        plt.ylabel("Number of Tariff Events", fontsize=12)
        plt.title("Tariff Intensity by Region", fontsize=15)
        plt.xticks(x, region_names, rotation=45, ha="right")
        plt.legend()

        # Add value labels on bars
        for i, count in enumerate(imposing_counts):
            if count > 0:
                plt.text(
                    i - width / 2, count + 0.1, str(count), ha="center", va="bottom"
                )

        for i, count in enumerate(targeted_counts):
            if count > 0:
                plt.text(
                    i + width / 2, count + 0.1, str(count), ha="center", va="bottom"
                )

        plt.tight_layout()

        # Convert plot to base64 string
        img = self._fig_to_base64(plt.gcf())
        plt.close()

        return img

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """
        Convert matplotlib figure to base64 encoded string.

        Args:
            fig: Matplotlib figure

        Returns:
            Base64 encoded PNG image
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        return img_str
