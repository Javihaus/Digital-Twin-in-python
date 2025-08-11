"""
Command Line Interface for the Hybrid Digital Twin Framework.

This module provides a professional CLI for training, predicting, and evaluating
hybrid digital twin models for battery capacity prediction.
"""

from typing import Optional, List
import sys
from pathlib import Path
import json
import yaml

import typer
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from loguru import logger

from hybrid_digital_twin import HybridDigitalTwin
from hybrid_digital_twin.data.data_loader import BatteryDataLoader
from hybrid_digital_twin.utils.exceptions import DigitalTwinError
from hybrid_digital_twin.visualization.plotters import BatteryPlotter

app = typer.Typer(
    name="hybrid-twin",
    help="Professional CLI for Hybrid Digital Twin battery modeling",
    add_completion=False,
    rich_markup_mode="rich"
)

console = Console()


@app.command()
def train(
    data: Path = typer.Argument(..., help="Path to battery data file"),
    output: Path = typer.Argument(..., help="Output path for trained model"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    target_column: str = typer.Option("Capacity", "--target", "-t", help="Target column name"),
    validation_split: float = typer.Option(0.2, "--val-split", help="Validation data fraction"),
    battery_id: Optional[str] = typer.Option(None, "--battery", "-b", help="Specific battery ID to train on"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    plot: bool = typer.Option(False, "--plot", help="Generate training plots"),
) -> None:
    """
    Train a hybrid digital twin model on battery data.
    
    This command trains both the physics-based and machine learning components
    of the hybrid digital twin and saves the trained model for future use.
    
    Example:
        hybrid-twin train data/discharge.csv models/battery_twin.pkl --config config.yaml
    """
    try:
        # Set up logging
        if verbose:
            logger.remove()
            logger.add(sys.stderr, level="DEBUG")
        
        console.print(Panel.fit(
            "[bold blue]Hybrid Digital Twin - Training[/bold blue]",
            border_style="blue"
        ))
        
        # Load configuration
        config_dict = {}
        if config and config.exists():
            with open(config) as f:
                if config.suffix.lower() in ['.yml', '.yaml']:
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)
            console.print(f"✅ Loaded configuration from {config}")
        
        # Load data
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading data...", total=None)
            
            loader = BatteryDataLoader()
            if battery_id:
                data_df = loader.load_nasa_dataset(battery_id, data)
            else:
                data_df = loader.load_csv(data)
            
            progress.update(task, completed=1, description="Data loaded successfully")
        
        console.print(f"📊 Loaded {len(data_df):,} samples with {len(data_df.columns)} features")
        
        # Display data info
        if target_column in data_df.columns:
            capacity_range = data_df[target_column].agg(['min', 'max'])
            console.print(f"🔋 Capacity range: {capacity_range['min']:.3f} - {capacity_range['max']:.3f} Ah")
        
        # Initialize and train model
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Training hybrid digital twin...", total=None)
            
            twin = HybridDigitalTwin(config=config_dict)
            metrics = twin.fit(
                data_df,
                target_column=target_column,
                validation_split=validation_split
            )
            
            progress.update(task, completed=1, description="Training completed")
        
        # Display training results
        console.print("\n[bold green]Training Results:[/bold green]")
        results_table = Table(show_header=True, header_style="bold magenta")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Train", justify="right")
        results_table.add_column("Validation", justify="right")
        
        key_metrics = ['rmse', 'mae', 'r2', 'mape']
        for metric in key_metrics:
            train_key = f"train_{metric}"
            val_key = f"val_{metric}"
            
            if train_key in metrics and val_key in metrics:
                train_val = f"{metrics[train_key]:.4f}"
                val_val = f"{metrics[val_key]:.4f}"
                if metric == 'r2':
                    train_val = f"{metrics[train_key]:.4f}"
                    val_val = f"{metrics[val_key]:.4f}"
                elif metric == 'mape':
                    train_val = f"{metrics[train_key]:.2f}%"
                    val_val = f"{metrics[val_key]:.2f}%"
                
                results_table.add_row(metric.upper(), train_val, val_val)
        
        console.print(results_table)
        
        # Save model
        output.parent.mkdir(parents=True, exist_ok=True)
        twin.save_model(output)
        console.print(f"💾 Model saved to {output}")
        
        # Generate plots if requested
        if plot:
            plot_dir = output.parent / "plots"
            plot_dir.mkdir(exist_ok=True)
            
            plotter = BatteryPlotter()
            
            # Training history plot
            if hasattr(twin.ml_model, 'training_history'):
                history_plot = plot_dir / "training_history.png"
                plotter.plot_training_history(twin.ml_model.training_history, save_path=history_plot)
                console.print(f"📈 Training history plot saved to {history_plot}")
            
            # Prediction comparison plot
            predictions = twin.predict(data_df, return_components=True)
            comparison_plot = plot_dir / "prediction_comparison.png"
            plotter.plot_prediction_comparison(
                actual=data_df[target_column].values,
                physics_pred=predictions.physics_prediction,
                hybrid_pred=predictions.hybrid_prediction,
                save_path=comparison_plot
            )
            console.print(f"📊 Prediction comparison plot saved to {comparison_plot}")
        
        console.print("\n[bold green]✅ Training completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Training failed: {str(e)}[/bold red]")
        raise typer.Exit(1)


@app.command()
def predict(
    model: Path = typer.Argument(..., help="Path to trained model file"),
    data: Path = typer.Argument(..., help="Path to data file for prediction"),
    output: Path = typer.Argument(..., help="Output path for predictions"),
    return_components: bool = typer.Option(False, "--components", help="Return individual model components"),
    return_uncertainty: bool = typer.Option(False, "--uncertainty", help="Return uncertainty estimates"),
    batch_size: int = typer.Option(1000, "--batch-size", help="Batch size for large datasets"),
    format: str = typer.Option("csv", "--format", help="Output format (csv, json, parquet)"),
) -> None:
    """
    Make predictions using a trained hybrid digital twin model.
    
    This command loads a trained model and generates predictions for new data.
    
    Example:
        hybrid-twin predict models/battery_twin.pkl data/new_data.csv predictions.csv
    """
    try:
        console.print(Panel.fit(
            "[bold blue]Hybrid Digital Twin - Prediction[/bold blue]",
            border_style="blue"
        ))
        
        # Load model
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading model...", total=None)
            twin = HybridDigitalTwin.load_model(model)
            progress.update(task, completed=1, description="Model loaded successfully")
        
        console.print(f"📥 Loaded model from {model}")
        
        # Load data
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading data...", total=None)
            
            loader = BatteryDataLoader()
            data_df = loader.load_csv(data, validate=False)  # Skip validation for prediction data
            
            progress.update(task, completed=1, description="Data loaded successfully")
        
        console.print(f"📊 Loaded {len(data_df):,} samples for prediction")
        
        # Make predictions
        all_predictions = []
        n_batches = (len(data_df) + batch_size - 1) // batch_size
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Making predictions...", total=n_batches)
            
            for i in range(0, len(data_df), batch_size):
                batch_data = data_df.iloc[i:i+batch_size]
                
                if return_components or return_uncertainty:
                    batch_pred = twin.predict(
                        batch_data,
                        return_components=True,
                        return_uncertainty=return_uncertainty
                    )
                else:
                    batch_pred = twin.predict(batch_data)
                
                all_predictions.append(batch_pred)
                progress.advance(task)
        
        # Combine results
        if return_components or return_uncertainty:
            # Combine PredictionResult objects
            combined_result = all_predictions[0]
            if len(all_predictions) > 1:
                combined_result.physics_prediction = np.concatenate([
                    pred.physics_prediction for pred in all_predictions
                ])
                combined_result.ml_correction = np.concatenate([
                    pred.ml_correction for pred in all_predictions
                ])
                combined_result.hybrid_prediction = np.concatenate([
                    pred.hybrid_prediction for pred in all_predictions
                ])
                if return_uncertainty and combined_result.uncertainty is not None:
                    combined_result.uncertainty = np.concatenate([
                        pred.uncertainty for pred in all_predictions
                    ])
            
            predictions = combined_result
        else:
            predictions = np.concatenate(all_predictions)
        
        # Prepare output data
        output_data = data_df.copy()
        
        if isinstance(predictions, np.ndarray):
            output_data['prediction'] = predictions
        else:
            output_data['physics_prediction'] = predictions.physics_prediction
            output_data['ml_correction'] = predictions.ml_correction
            output_data['hybrid_prediction'] = predictions.hybrid_prediction
            if return_uncertainty and predictions.uncertainty is not None:
                output_data['uncertainty'] = predictions.uncertainty
        
        # Save results
        output.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'csv':
            output_data.to_csv(output, index=False)
        elif format.lower() == 'json':
            output_data.to_json(output, orient='records', indent=2)
        elif format.lower() == 'parquet':
            output_data.to_parquet(output, index=False)
        else:
            raise ValueError(f"Unsupported output format: {format}")
        
        console.print(f"💾 Predictions saved to {output}")
        
        # Display summary statistics
        if isinstance(predictions, np.ndarray):
            pred_values = predictions
        else:
            pred_values = predictions.hybrid_prediction
        
        console.print(f"\n[bold green]Prediction Summary:[/bold green]")
        console.print(f"📊 Mean prediction: {pred_values.mean():.4f}")
        console.print(f"📊 Std prediction: {pred_values.std():.4f}")
        console.print(f"📊 Min prediction: {pred_values.min():.4f}")
        console.print(f"📊 Max prediction: {pred_values.max():.4f}")
        
        console.print("\n[bold green]✅ Predictions completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Prediction failed: {str(e)}[/bold red]")
        raise typer.Exit(1)


@app.command()
def evaluate(
    model: Path = typer.Argument(..., help="Path to trained model file"),
    test_data: Path = typer.Argument(..., help="Path to test data file"),
    target_column: str = typer.Option("Capacity", "--target", "-t", help="Target column name"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path for evaluation report"),
    plot: bool = typer.Option(False, "--plot", help="Generate evaluation plots"),
) -> None:
    """
    Evaluate a trained hybrid digital twin model on test data.
    
    This command loads a trained model and evaluates its performance on test data,
    providing comprehensive metrics and optional visualizations.
    
    Example:
        hybrid-twin evaluate models/battery_twin.pkl data/test_data.csv --output evaluation_report.json
    """
    try:
        console.print(Panel.fit(
            "[bold blue]Hybrid Digital Twin - Evaluation[/bold blue]",
            border_style="blue"
        ))
        
        # Load model and data
        twin = HybridDigitalTwin.load_model(model)
        console.print(f"📥 Loaded model from {model}")
        
        loader = BatteryDataLoader()
        test_df = loader.load_csv(test_data)
        console.print(f"📊 Loaded {len(test_df):,} test samples")
        
        # Evaluate model
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Evaluating model...", total=None)
            
            metrics = twin.evaluate(test_df, target_column=target_column)
            
            progress.update(task, completed=1, description="Evaluation completed")
        
        # Display results
        console.print("\n[bold green]Evaluation Results:[/bold green]")
        
        metrics_table = Table(show_header=True, header_style="bold magenta")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", justify="right")
        
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                if metric_name in ['mape']:
                    formatted_value = f"{value:.2f}%"
                elif metric_name in ['r2']:
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = f"{value:.4f}"
                
                metrics_table.add_row(metric_name.upper(), formatted_value)
        
        console.print(metrics_table)
        
        # Save evaluation report
        if output:
            report = {
                'model_path': str(model),
                'test_data_path': str(test_data),
                'target_column': target_column,
                'n_test_samples': len(test_df),
                'metrics': metrics,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, 'w') as f:
                json.dump(report, f, indent=2)
            
            console.print(f"📄 Evaluation report saved to {output}")
        
        # Generate plots if requested
        if plot:
            plot_dir = (output.parent if output else Path('.')) / "evaluation_plots"
            plot_dir.mkdir(exist_ok=True)
            
            plotter = BatteryPlotter()
            
            # Get predictions for plotting
            predictions = twin.predict(test_df, return_components=True)
            actual_values = test_df[target_column].values
            
            # Prediction vs actual plot
            scatter_plot = plot_dir / "prediction_vs_actual.png"
            plotter.plot_prediction_scatter(
                actual=actual_values,
                predicted=predictions.hybrid_prediction,
                save_path=scatter_plot
            )
            console.print(f"📈 Prediction scatter plot saved to {scatter_plot}")
            
            # Residuals plot
            residuals_plot = plot_dir / "residuals.png"
            plotter.plot_residuals(
                actual=actual_values,
                predicted=predictions.hybrid_prediction,
                save_path=residuals_plot
            )
            console.print(f"📊 Residuals plot saved to {residuals_plot}")
        
        console.print("\n[bold green]✅ Evaluation completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Evaluation failed: {str(e)}[/bold red]")
        raise typer.Exit(1)


@app.command()
def info(
    model: Path = typer.Argument(..., help="Path to trained model file"),
) -> None:
    """
    Display information about a trained hybrid digital twin model.
    
    This command loads a model and displays detailed information about its
    configuration, training history, and performance metrics.
    
    Example:
        hybrid-twin info models/battery_twin.pkl
    """
    try:
        console.print(Panel.fit(
            "[bold blue]Hybrid Digital Twin - Model Information[/bold blue]",
            border_style="blue"
        ))
        
        # Load model
        twin = HybridDigitalTwin.load_model(model)
        console.print(f"📥 Loaded model from {model}")
        
        # Display basic information
        console.print(f"\n[bold]Model Status:[/bold] {'✅ Trained' if twin.is_trained else '❌ Not trained'}")
        
        # Configuration
        if twin.config:
            console.print("\n[bold green]Configuration:[/bold green]")
            config_table = Table(show_header=True, header_style="bold magenta")
            config_table.add_column("Parameter", style="cyan")
            config_table.add_column("Value", justify="right")
            
            def add_config_rows(config_dict, prefix=""):
                for key, value in config_dict.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, dict):
                        add_config_rows(value, full_key)
                    else:
                        config_table.add_row(full_key, str(value))
            
            add_config_rows(twin.config)
            console.print(config_table)
        
        # Training history
        if twin.training_history:
            console.print("\n[bold green]Training History:[/bold green]")
            
            history_table = Table(show_header=True, header_style="bold magenta")
            history_table.add_column("Component", style="cyan")
            history_table.add_column("Metric", style="cyan")
            history_table.add_column("Value", justify="right")
            
            for component, metrics in twin.training_history.items():
                if isinstance(metrics, dict):
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            formatted_value = f"{metric_value:.4f}"
                            history_table.add_row(component, metric_name, formatted_value)
            
            console.print(history_table)
        
        # Model architecture info
        console.print("\n[bold green]Model Architecture:[/bold green]")
        arch_table = Table(show_header=True, header_style="bold magenta")
        arch_table.add_column("Component", style="cyan")
        arch_table.add_column("Type", style="cyan")
        arch_table.add_column("Status", justify="right")
        
        arch_table.add_row("Physics Model", "Degradation Model", "✅ Fitted" if twin.physics_model.is_fitted else "❌ Not fitted")
        arch_table.add_row("ML Model", "Neural Network", "✅ Fitted" if twin.ml_model.is_fitted else "❌ Not fitted")
        
        console.print(arch_table)
        
        console.print("\n[bold green]✅ Model information displayed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Failed to load model info: {str(e)}[/bold red]")
        raise typer.Exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()