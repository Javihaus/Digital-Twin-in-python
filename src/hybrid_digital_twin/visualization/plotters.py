"""
Professional plotting utilities for battery data visualization.

This module provides high-quality plotting functions for visualizing
battery data, model predictions, and evaluation results.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.style as style
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns

from hybrid_digital_twin.utils.exceptions import VisualizationError

# Set professional styling
plt.style.use('default')
sns.set_palette("husl")


class BatteryPlotter:
    """
    Professional plotting utility for battery data visualization.

    This class provides methods for creating publication-quality plots
    for battery data analysis, model predictions, and evaluation results.
    """

    def __init__(self, style: str = "plotly_white", color_scheme: str = "default"):
        """
        Initialize the plotter with styling options.

        Args:
            style: Plotly template style
            color_scheme: Color scheme for plots
        """
        self.style = style
        self.color_scheme = color_scheme
        self.colors = self._get_color_palette(color_scheme)

    def plot_capacity_degradation(
        self,
        data: pd.DataFrame,
        title: str = "Battery Capacity Degradation",
        save_path: Optional[Path] = None,
        interactive: bool = True
    ) -> go.Figure:
        """
        Plot battery capacity degradation over cycles.

        Args:
            data: DataFrame with 'id_cycle' and 'Capacity' columns
            title: Plot title
            save_path: Optional path to save the plot
            interactive: Whether to create interactive plot

        Returns:
            Plotly figure object
        """
        try:
            if interactive:
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=data['id_cycle'],
                    y=data['Capacity'],
                    mode='lines+markers',
                    name='Capacity',
                    line=dict(color=self.colors['primary'], width=2),
                    marker=dict(size=4)
                ))

                fig.update_layout(
                    title=title,
                    xaxis_title="Cycle Number",
                    yaxis_title="Capacity (Ah)",
                    template=self.style,
                    hovermode='x unified'
                )

                if save_path:
                    fig.write_html(save_path.with_suffix('.html'))
                    fig.write_image(save_path.with_suffix('.png'), width=800, height=600)

                return fig
            else:
                plt.figure(figsize=(10, 6))
                plt.plot(data['id_cycle'], data['Capacity'], 'o-',
                        color=self.colors['primary'], linewidth=2, markersize=4)
                plt.title(title)
                plt.xlabel('Cycle Number')
                plt.ylabel('Capacity (Ah)')
                plt.grid(True, alpha=0.3)

                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close()

        except Exception as e:
            raise VisualizationError(f"Failed to plot capacity degradation: {str(e)}")

    def plot_prediction_comparison(
        self,
        actual: np.ndarray,
        physics_pred: np.ndarray,
        hybrid_pred: np.ndarray,
        cycles: Optional[np.ndarray] = None,
        title: str = "Model Prediction Comparison",
        save_path: Optional[Path] = None
    ) -> go.Figure:
        """
        Compare actual values with physics and hybrid predictions.

        Args:
            actual: Actual capacity values
            physics_pred: Physics model predictions
            hybrid_pred: Hybrid model predictions
            cycles: Optional cycle numbers
            title: Plot title
            save_path: Optional path to save the plot

        Returns:
            Plotly figure object
        """
        try:
            if cycles is None:
                cycles = np.arange(len(actual))

            fig = go.Figure()

            # Actual data
            fig.add_trace(go.Scatter(
                x=cycles,
                y=actual,
                mode='markers',
                name='Actual',
                marker=dict(
                    color=self.colors['actual'],
                    symbol='circle',
                    size=6
                )
            ))

            # Physics predictions
            fig.add_trace(go.Scatter(
                x=cycles,
                y=physics_pred,
                mode='lines',
                name='Physics Model',
                line=dict(
                    color=self.colors['physics'],
                    width=2,
                    dash='dash'
                )
            ))

            # Hybrid predictions
            fig.add_trace(go.Scatter(
                x=cycles,
                y=hybrid_pred,
                mode='lines',
                name='Hybrid Model',
                line=dict(
                    color=self.colors['hybrid'],
                    width=3
                )
            ))

            fig.update_layout(
                title=title,
                xaxis_title="Cycle Number",
                yaxis_title="Capacity (Ah)",
                template=self.style,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )

            if save_path:
                fig.write_html(save_path.with_suffix('.html'))
                fig.write_image(save_path.with_suffix('.png'), width=1000, height=600)

            return fig

        except Exception as e:
            raise VisualizationError(f"Failed to plot prediction comparison: {str(e)}")

    def plot_residuals(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        title: str = "Residual Analysis",
        save_path: Optional[Path] = None
    ) -> go.Figure:
        """
        Plot residuals for model diagnostics.

        Args:
            actual: Actual values
            predicted: Predicted values
            title: Plot title
            save_path: Optional path to save the plot

        Returns:
            Plotly figure object
        """
        try:
            residuals = actual - predicted

            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Residuals vs Predicted',
                    'Residuals Distribution',
                    'Q-Q Plot',
                    'Residuals vs Index'
                )
            )

            # Residuals vs Predicted
            fig.add_trace(
                go.Scatter(
                    x=predicted,
                    y=residuals,
                    mode='markers',
                    name='Residuals',
                    marker=dict(color=self.colors['primary'], size=4)
                ),
                row=1, col=1
            )

            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

            # Residuals histogram
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    nbinsx=30,
                    name='Distribution',
                    marker=dict(color=self.colors['secondary'])
                ),
                row=1, col=2
            )

            # Q-Q plot (simplified)
            from scipy import stats
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
            sample_quantiles = np.sort(residuals)

            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sample_quantiles,
                    mode='markers',
                    name='Q-Q',
                    marker=dict(color=self.colors['accent'], size=4)
                ),
                row=2, col=1
            )

            # Add perfect correlation line
            min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
            max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(dash='dash', color='red'),
                    showlegend=False
                ),
                row=2, col=1
            )

            # Residuals vs Index
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(residuals)),
                    y=residuals,
                    mode='markers',
                    name='Index',
                    marker=dict(color=self.colors['primary'], size=4)
                ),
                row=2, col=2
            )

            fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)

            fig.update_layout(
                title_text=title,
                template=self.style,
                showlegend=False,
                height=800
            )

            if save_path:
                fig.write_html(save_path.with_suffix('.html'))
                fig.write_image(save_path.with_suffix('.png'), width=1000, height=800)

            return fig

        except Exception as e:
            raise VisualizationError(f"Failed to plot residuals: {str(e)}")

    def plot_training_history(
        self,
        history: Dict,
        title: str = "Training History",
        save_path: Optional[Path] = None
    ) -> go.Figure:
        """
        Plot training history for ML model.

        Args:
            history: Training history dictionary
            title: Plot title
            save_path: Optional path to save the plot

        Returns:
            Plotly figure object
        """
        try:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Loss', 'Mean Absolute Error')
            )

            epochs = list(range(1, len(history['loss']) + 1))

            # Loss plot
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history['loss'],
                    mode='lines',
                    name='Training Loss',
                    line=dict(color=self.colors['primary'])
                ),
                row=1, col=1
            )

            if 'val_loss' in history and history['val_loss']:
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=history['val_loss'],
                        mode='lines',
                        name='Validation Loss',
                        line=dict(color=self.colors['secondary'])
                    ),
                    row=1, col=1
                )

            # MAE plot
            if 'mae' in history:
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=history['mae'],
                        mode='lines',
                        name='Training MAE',
                        line=dict(color=self.colors['primary'])
                    ),
                    row=1, col=2
                )

                if 'val_mae' in history and history['val_mae']:
                    fig.add_trace(
                        go.Scatter(
                            x=epochs,
                            y=history['val_mae'],
                            mode='lines',
                            name='Validation MAE',
                            line=dict(color=self.colors['secondary'])
                        ),
                        row=1, col=2
                    )

            fig.update_layout(
                title_text=title,
                template=self.style
            )

            fig.update_xaxes(title_text="Epoch")
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_yaxes(title_text="MAE", row=1, col=2)

            if save_path:
                fig.write_html(save_path.with_suffix('.html'))
                fig.write_image(save_path.with_suffix('.png'), width=1000, height=500)

            return fig

        except Exception as e:
            raise VisualizationError(f"Failed to plot training history: {str(e)}")

    def plot_prediction_scatter(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        title: str = "Predicted vs Actual",
        save_path: Optional[Path] = None
    ) -> go.Figure:
        """
        Create scatter plot of predicted vs actual values.

        Args:
            actual: Actual values
            predicted: Predicted values
            title: Plot title
            save_path: Optional path to save the plot

        Returns:
            Plotly figure object
        """
        try:
            # Calculate R² and RMSE for display
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))

            fig = go.Figure()

            # Scatter plot
            fig.add_trace(go.Scatter(
                x=actual,
                y=predicted,
                mode='markers',
                name=f'R² = {r2:.3f}<br>RMSE = {rmse:.4f}',
                marker=dict(
                    color=self.colors['primary'],
                    size=6,
                    opacity=0.7
                )
            ))

            # Perfect prediction line
            min_val = min(actual.min(), predicted.min())
            max_val = max(actual.max(), predicted.max())

            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red', width=2)
            ))

            fig.update_layout(
                title=title,
                xaxis_title="Actual Values",
                yaxis_title="Predicted Values",
                template=self.style
            )

            if save_path:
                fig.write_html(save_path.with_suffix('.html'))
                fig.write_image(save_path.with_suffix('.png'), width=800, height=600)

            return fig

        except Exception as e:
            raise VisualizationError(f"Failed to create scatter plot: {str(e)}")

    def _get_color_palette(self, scheme: str) -> Dict[str, str]:
        """Get color palette based on scheme."""
        palettes = {
            "default": {
                "primary": "#2E86AB",
                "secondary": "#A23B72",
                "accent": "#F18F01",
                "actual": "#525252",
                "physics": "#2E86AB",
                "hybrid": "#C73E1D"
            },
            "professional": {
                "primary": "#1f77b4",
                "secondary": "#ff7f0e",
                "accent": "#2ca02c",
                "actual": "#d62728",
                "physics": "#9467bd",
                "hybrid": "#8c564b"
            }
        }

        return palettes.get(scheme, palettes["default"])
