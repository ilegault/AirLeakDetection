"""Report generation for comprehensive evaluation results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from jinja2 import Template


class ReportGenerator:
    """Generate evaluation reports in multiple formats."""

    def __init__(self, output_dir: str = "results/reports") -> None:
        """Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_markdown_report(
        self,
        metrics: Dict[str, Any],
        model_name: str = "Model",
        additional_info: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """Generate a Markdown report.

        Args:
            metrics: Evaluation metrics dictionary
            model_name: Name of the model
            additional_info: Additional information to include
            save_path: Path to save report

        Returns:
            Markdown report as string
        """
        report = f"# Evaluation Report: {model_name}\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Overall metrics
        report += "## Overall Metrics\n\n"
        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"

        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if isinstance(value, float):
                    report += f"| {key} | {value:.4f} |\n"
                else:
                    report += f"| {key} | {value} |\n"

        # Per-class metrics
        if "per_class_metrics" in metrics:
            report += "\n## Per-Class Metrics\n\n"
            report += "| Class | Precision | Recall | F1 |\n"
            report += "|-------|-----------|--------|----|\n"

            for class_id, class_metrics in metrics["per_class_metrics"].items():
                precision = class_metrics.get("precision", 0)
                recall = class_metrics.get("recall", 0)
                f1 = class_metrics.get("f1", 0)
                report += f"| {class_id} | {precision:.4f} | {recall:.4f} | {f1:.4f} |\n"

        # Additional information
        if additional_info:
            report += "\n## Additional Information\n\n"
            for key, value in additional_info.items():
                report += f"- **{key}:** {value}\n"

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                f.write(report)
        elif hasattr(self, "output_dir"):
            default_path = self.output_dir / f"report_markdown_{self.timestamp}.md"
            with open(default_path, "w") as f:
                f.write(report)

        return report

    def generate_html_report(
        self,
        metrics: Dict[str, Any],
        model_name: str = "Model",
        figure_paths: Optional[List[str]] = None,
        additional_info: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """Generate an HTML report with embedded figures.

        Args:
            metrics: Evaluation metrics dictionary
            model_name: Name of the model
            figure_paths: Paths to image files to embed
            additional_info: Additional information to include
            save_path: Path to save report

        Returns:
            HTML report as string
        """
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evaluation Report: {{ model_name }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 1000px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; }
                h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
                h2 { color: #555; margin-top: 30px; }
                table { border-collapse: collapse; width: 100%; margin: 15px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #007bff; color: white; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .info { background-color: #e7f3ff; padding: 10px; border-left: 4px solid #007bff; margin: 15px 0; }
                .figure { text-align: center; margin: 20px 0; }
                .figure img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
                .metric-value { font-weight: bold; color: #007bff; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Evaluation Report: {{ model_name }}</h1>
                <div class="info">
                    <strong>Generated:</strong> {{ timestamp }}<br>
                    <strong>Total Samples:</strong> {{ total_samples }}
                </div>

                <h2>Overall Metrics</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for key, value in overall_metrics %}
                        <tr>
                            <td>{{ key }}</td>
                            <td><span class="metric-value">{{ "%.4f"|format(value) if value is number else value }}</span></td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>

                {% if per_class_metrics %}
                <h2>Per-Class Metrics</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Class</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for class_id, metrics in per_class_metrics %}
                        <tr>
                            <td>{{ class_id }}</td>
                            <td>{{ "%.4f"|format(metrics['precision']) }}</td>
                            <td>{{ "%.4f"|format(metrics['recall']) }}</td>
                            <td>{{ "%.4f"|format(metrics['f1']) }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                {% endif %}

                {% if additional_info %}
                <h2>Additional Information</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Key</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for key, value in additional_info.items() %}
                        <tr>
                            <td>{{ key }}</td>
                            <td>{{ value }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                {% endif %}

                {% if figures %}
                <h2>Visualizations</h2>
                {% for figure_path in figures %}
                <div class="figure">
                    <img src="data:image/png;base64,{{ figure_path }}" alt="Figure">
                </div>
                {% endfor %}
                {% endif %}
            </div>
        </body>
        </html>
        """

        # Extract overall metrics
        overall_metrics = [
            (k, v) for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "per_class_metrics"
        ]

        # Extract per-class metrics
        per_class_metrics = []
        if "per_class_metrics" in metrics:
            per_class_metrics = list(metrics["per_class_metrics"].items())

        # Render template
        template = Template(html_template)
        html_report = template.render(
            model_name=model_name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_samples=metrics.get("total_samples", "N/A"),
            overall_metrics=overall_metrics,
            per_class_metrics=per_class_metrics if per_class_metrics else None,
            additional_info=additional_info or {},
            figures=figure_paths or [],
        )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                f.write(html_report)
        elif hasattr(self, "output_dir"):
            default_path = self.output_dir / f"report_html_{self.timestamp}.html"
            with open(default_path, "w") as f:
                f.write(html_report)

        return html_report

    def generate_latex_tables(
        self,
        metrics: Dict[str, Any],
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> str:
        """Generate LaTeX table for metrics.

        Args:
            metrics: Evaluation metrics dictionary
            model_name: Name of the model
            save_path: Path to save LaTeX file

        Returns:
            LaTeX code as string
        """
        latex = "\\documentclass{article}\n"
        latex += "\\usepackage{booktabs}\n"
        latex += "\\begin{document}\n\n"

        latex += f"\\section{{{model_name} Evaluation Report}}\n\n"

        # Overall metrics table
        latex += "\\subsection{Overall Metrics}\n"
        latex += "\\begin{table}[h]\n"
        latex += "\\centering\n"
        latex += "\\begin{tabular}{lr}\n"
        latex += "\\toprule\n"
        latex += "\\textbf{Metric} & \\textbf{Value} \\\\\n"
        latex += "\\midrule\n"

        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if isinstance(value, float):
                    latex += f"{key} & {value:.4f} \\\\\n"
                else:
                    latex += f"{key} & {value} \\\\\n"

        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n\n"

        # Per-class metrics table
        if "per_class_metrics" in metrics:
            latex += "\\subsection{Per-Class Metrics}\n"
            latex += "\\begin{table}[h]\n"
            latex += "\\centering\n"
            latex += "\\begin{tabular}{lrrr}\n"
            latex += "\\toprule\n"
            latex += "\\textbf{Class} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} \\\\\n"
            latex += "\\midrule\n"

            for class_id, class_metrics in metrics["per_class_metrics"].items():
                precision = class_metrics.get("precision", 0)
                recall = class_metrics.get("recall", 0)
                f1 = class_metrics.get("f1", 0)
                latex += f"{class_id} & {precision:.4f} & {recall:.4f} & {f1:.4f} \\\\\n"

            latex += "\\bottomrule\n"
            latex += "\\end{tabular}\n"
            latex += "\\end{table}\n\n"

        latex += "\\end{document}\n"

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                f.write(latex)
        elif hasattr(self, "output_dir"):
            default_path = self.output_dir / f"report_latex_{self.timestamp}.tex"
            with open(default_path, "w") as f:
                f.write(latex)

        return latex

    def generate_json_report(
        self,
        metrics: Dict[str, Any],
        model_name: str = "Model",
        additional_info: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """Generate a JSON report.

        Args:
            metrics: Evaluation metrics dictionary
            model_name: Name of the model
            additional_info: Additional information to include
            save_path: Path to save JSON report

        Returns:
            JSON report as string
        """
        report_dict = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": self._convert_to_json_serializable(metrics),
            "additional_info": additional_info or {},
        }

        json_str = json.dumps(report_dict, indent=2)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                f.write(json_str)
        elif hasattr(self, "output_dir"):
            default_path = self.output_dir / f"report_json_{self.timestamp}.json"
            with open(default_path, "w") as f:
                f.write(json_str)

        return json_str

    @staticmethod
    def _convert_to_json_serializable(obj: Any) -> Any:
        """Convert numpy arrays and other types to JSON serializable format.

        Args:
            obj: Object to convert

        Returns:
            JSON serializable object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: ReportGenerator._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ReportGenerator._convert_to_json_serializable(item) for item in obj]
        else:
            return obj