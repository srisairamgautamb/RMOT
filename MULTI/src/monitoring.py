"""
Research Monitoring and Alerting System

Lightweight monitoring for research experiments:
- Metrics logging (file-based, no external services required)
- Optional Prometheus export
- Optional Slack webhook alerts
- Console dashboards

For research use: no Docker/Kubernetes required!
"""

import time
import json
import os
import csv
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict, field
import numpy as np

# Optional imports for extended functionality
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class PipelineMetrics:
    """Metrics from a single pipeline run."""
    timestamp: str
    iteration: int
    n_assets: int
    n_paths: int
    
    # Timing
    total_time_sec: float
    calibration_time_sec: float = 0.0
    correlation_time_sec: float = 0.0
    simulation_time_sec: float = 0.0
    pricing_time_sec: float = 0.0
    
    # Quality
    max_calibration_error: float = 0.0
    max_correlation_error: float = 0.0
    avg_bound_width: float = 0.0
    
    # Parameters
    H_values: List[float] = field(default_factory=list)
    rho_matrix: List[List[float]] = field(default_factory=list)
    
    # Status
    success: bool = True
    error_message: str = ""


class ResearchMonitor:
    """
    Comprehensive monitoring for research experiments.
    
    Features:
    - File-based metrics logging (CSV, JSON)
    - Console dashboard
    - Optional Prometheus metrics export
    - Optional Slack alerts
    """
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "/tmp/rmot_experiments",
        slack_webhook: Optional[str] = None,
        alert_thresholds: Dict = None
    ):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.slack_webhook = slack_webhook
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'max_time_sec': 60.0,
            'max_calibration_error': 0.20,
            'max_correlation_error': 0.10,
            'max_bound_width': 1.0
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize logs
        self.metrics_history: List[PipelineMetrics] = []
        self.alerts_sent: List[Dict] = []
        
        # File paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(output_dir, f"{experiment_name}_{timestamp}.csv")
        self.json_path = os.path.join(output_dir, f"{experiment_name}_{timestamp}.json")
        self.prometheus_path = os.path.join(output_dir, f"{experiment_name}_metrics.prom")
        
        print(f"📊 Monitor initialized: {experiment_name}")
        print(f"   Output: {output_dir}")
    
    def log_metrics(self, metrics: PipelineMetrics):
        """Log metrics and check for alerts."""
        self.metrics_history.append(metrics)
        
        # Write to CSV
        self._write_csv(metrics)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        # Print console dashboard
        self._print_dashboard(metrics)
    
    def _write_csv(self, metrics: PipelineMetrics):
        """Append metrics to CSV file."""
        file_exists = os.path.exists(self.csv_path)
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if new file
            if not file_exists:
                headers = [
                    'timestamp', 'iteration', 'n_assets', 'n_paths',
                    'total_time_sec', 'calibration_time_sec', 'simulation_time_sec',
                    'max_calibration_error', 'max_correlation_error', 'avg_bound_width',
                    'success', 'error_message'
                ]
                writer.writerow(headers)
            
            # Write data
            row = [
                metrics.timestamp,
                metrics.iteration,
                metrics.n_assets,
                metrics.n_paths,
                f"{metrics.total_time_sec:.3f}",
                f"{metrics.calibration_time_sec:.3f}",
                f"{metrics.simulation_time_sec:.3f}",
                f"{metrics.max_calibration_error:.4f}",
                f"{metrics.max_correlation_error:.4f}",
                f"{metrics.avg_bound_width:.4f}",
                metrics.success,
                metrics.error_message
            ]
            writer.writerow(row)
    
    def _check_alerts(self, metrics: PipelineMetrics):
        """Check for threshold violations and send alerts."""
        alerts = []
        
        # Check time
        if metrics.total_time_sec > self.alert_thresholds['max_time_sec']:
            alerts.append({
                'severity': 'WARNING',
                'metric': 'total_time_sec',
                'value': metrics.total_time_sec,
                'threshold': self.alert_thresholds['max_time_sec'],
                'message': f"Slow pipeline: {metrics.total_time_sec:.1f}s > {self.alert_thresholds['max_time_sec']}s"
            })
        
        # Check calibration error
        if metrics.max_calibration_error > self.alert_thresholds['max_calibration_error']:
            alerts.append({
                'severity': 'WARNING',
                'metric': 'max_calibration_error',
                'value': metrics.max_calibration_error,
                'threshold': self.alert_thresholds['max_calibration_error'],
                'message': f"High calibration error: {metrics.max_calibration_error:.3f}"
            })
        
        # Check correlation error
        if metrics.max_correlation_error > self.alert_thresholds['max_correlation_error']:
            alerts.append({
                'severity': 'CRITICAL',
                'metric': 'max_correlation_error',
                'value': metrics.max_correlation_error,
                'threshold': self.alert_thresholds['max_correlation_error'],
                'message': f"High correlation error: {metrics.max_correlation_error:.3f}"
            })
        
        # Check failure
        if not metrics.success:
            alerts.append({
                'severity': 'CRITICAL',
                'metric': 'success',
                'value': False,
                'threshold': True,
                'message': f"Pipeline failed: {metrics.error_message}"
            })
        
        # Send alerts
        for alert in alerts:
            self._send_alert(alert, metrics.iteration)
    
    def _send_alert(self, alert: Dict, iteration: int):
        """Send alert via console and optionally Slack."""
        alert['timestamp'] = datetime.now().isoformat()
        alert['iteration'] = iteration
        self.alerts_sent.append(alert)
        
        # Console alert
        severity_icon = "🔴" if alert['severity'] == 'CRITICAL' else "🟡"
        print(f"\n{severity_icon} ALERT [{alert['severity']}]: {alert['message']}")
        
        # Slack alert (if configured)
        if self.slack_webhook and HAS_REQUESTS:
            self._send_slack_alert(alert)
    
    def _send_slack_alert(self, alert: Dict):
        """Send alert to Slack webhook."""
        color = "#ff0000" if alert['severity'] == 'CRITICAL' else "#ffcc00"
        
        payload = {
            'attachments': [{
                'color': color,
                'title': f"Multi-Asset RMOT Alert: {alert['severity']}",
                'text': alert['message'],
                'fields': [
                    {'title': 'Metric', 'value': alert['metric'], 'short': True},
                    {'title': 'Value', 'value': str(alert['value']), 'short': True},
                    {'title': 'Threshold', 'value': str(alert['threshold']), 'short': True},
                    {'title': 'Iteration', 'value': str(alert['iteration']), 'short': True}
                ],
                'ts': int(time.time())
            }]
        }
        
        try:
            response = requests.post(self.slack_webhook, json=payload, timeout=5)
            if response.status_code == 200:
                print("   → Slack alert sent")
            else:
                print(f"   → Slack failed: {response.status_code}")
        except Exception as e:
            print(f"   → Slack error: {e}")
    
    def _print_dashboard(self, metrics: PipelineMetrics):
        """Print console dashboard."""
        print(f"\n{'─'*60}")
        print(f"📊 Iteration {metrics.iteration} | {metrics.timestamp}")
        print(f"{'─'*60}")
        print(f"  Time: {metrics.total_time_sec:.2f}s | Assets: {metrics.n_assets} | Paths: {metrics.n_paths:,}")
        print(f"  Calibration Error: {metrics.max_calibration_error:.4f}")
        print(f"  Correlation Error: {metrics.max_correlation_error:.4f}")
        print(f"  Avg Bound Width: ${metrics.avg_bound_width:.4f}")
        print(f"  Status: {'✅ Success' if metrics.success else '❌ Failed'}")
    
    def export_prometheus(self):
        """Export metrics in Prometheus format (for optional use)."""
        lines = [
            "# HELP rmot_pipeline_time_seconds Pipeline execution time",
            "# TYPE rmot_pipeline_time_seconds gauge",
        ]
        
        if self.metrics_history:
            latest = self.metrics_history[-1]
            lines.append(f'rmot_pipeline_time_seconds {latest.total_time_sec}')
            lines.append(f'rmot_calibration_error {latest.max_calibration_error}')
            lines.append(f'rmot_correlation_error {latest.max_correlation_error}')
            lines.append(f'rmot_bound_width {latest.avg_bound_width}')
            lines.append(f'rmot_iterations_total {len(self.metrics_history)}')
            lines.append(f'rmot_alerts_total {len(self.alerts_sent)}')
        
        with open(self.prometheus_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"📈 Prometheus metrics exported: {self.prometheus_path}")
    
    def print_summary(self):
        """Print experiment summary."""
        if not self.metrics_history:
            print("No metrics recorded")
            return
        
        times = [m.total_time_sec for m in self.metrics_history]
        cal_errors = [m.max_calibration_error for m in self.metrics_history]
        corr_errors = [m.max_correlation_error for m in self.metrics_history]
        widths = [m.avg_bound_width for m in self.metrics_history]
        successes = sum(1 for m in self.metrics_history if m.success)
        
        print(f"\n{'='*70}")
        print(f"📊 EXPERIMENT SUMMARY: {self.experiment_name}")
        print(f"{'='*70}")
        print(f"  Iterations: {len(self.metrics_history)}")
        print(f"  Success rate: {successes}/{len(self.metrics_history)} ({100*successes/len(self.metrics_history):.0f}%)")
        print(f"  Total alerts: {len(self.alerts_sent)}")
        print(f"\n  Timing:")
        print(f"    Mean: {np.mean(times):.2f}s")
        print(f"    Std:  {np.std(times):.2f}s")
        print(f"    Min:  {np.min(times):.2f}s")
        print(f"    Max:  {np.max(times):.2f}s")
        print(f"\n  Calibration Error:")
        print(f"    Mean: {np.mean(cal_errors):.4f}")
        print(f"    Max:  {np.max(cal_errors):.4f}")
        print(f"\n  Correlation Error:")
        print(f"    Mean: {np.mean(corr_errors):.4f}")
        print(f"    Max:  {np.max(corr_errors):.4f}")
        print(f"\n  Bound Width:")
        print(f"    Mean: ${np.mean(widths):.4f}")
        print(f"\n  Output files:")
        print(f"    CSV:  {self.csv_path}")
        print(f"    JSON: {self.json_path}")
    
    def save_full_report(self):
        """Save complete experiment report as JSON."""
        report = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'n_iterations': len(self.metrics_history),
            'n_alerts': len(self.alerts_sent),
            'metrics': [asdict(m) for m in self.metrics_history],
            'alerts': self.alerts_sent,
            'thresholds': self.alert_thresholds
        }
        
        with open(self.json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Full report saved: {self.json_path}")


def create_metrics_from_result(
    result: Dict,
    config,
    iteration: int,
    elapsed: float
) -> PipelineMetrics:
    """Create PipelineMetrics from pipeline result."""
    
    # Extract metrics
    marginal = result['marginal_calibration']
    correlation = result['correlation_estimation']
    bounds = result['frtb_bounds']
    
    # Calibration error
    max_cal_error = float(np.max(marginal.calibration_errors))
    
    # Correlation error (vs initial guess)
    rho_target = config.correlation_guess
    rho_est = correlation.rho
    max_corr_error = float(np.max(np.abs(rho_target - rho_est)))
    
    # Bound width
    avg_width = float(np.mean([b.width for b in bounds]))
    
    return PipelineMetrics(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        iteration=iteration,
        n_assets=config.n_assets,
        n_paths=result.get('n_paths', 0),
        total_time_sec=elapsed,
        max_calibration_error=max_cal_error,
        max_correlation_error=max_corr_error,
        avg_bound_width=avg_width,
        H_values=[p.H for p in marginal.params],
        rho_matrix=rho_est.tolist(),
        success=True
    )


def run_monitored_experiment(
    tickers: List[str],
    n_iterations: int = 5,
    update_interval: int = 60,
    slack_webhook: Optional[str] = None
):
    """
    Run monitored experiment with full alerting.
    
    This is the recommended way to run research experiments.
    """
    from .real_time_data import RealTimeDataStream
    from .pipeline import multi_asset_rmot_pipeline
    
    print("\n" + "=" * 70)
    print("🔬 MONITORED RESEARCH EXPERIMENT")
    print("=" * 70)
    
    # Initialize
    stream = RealTimeDataStream()
    monitor = ResearchMonitor(
        experiment_name=f"rmot_{'_'.join(tickers)}",
        slack_webhook=slack_webhook
    )
    
    for iteration in range(n_iterations):
        try:
            # Fetch data
            config = stream.fetch_live_data(tickers)
            
            # Run pipeline
            start = time.time()
            result = multi_asset_rmot_pipeline(config, n_paths=30000, verbose=False)
            elapsed = time.time() - start
            
            # Create metrics
            metrics = create_metrics_from_result(result, config, iteration, elapsed)
            
            # Log
            monitor.log_metrics(metrics)
            
        except Exception as e:
            # Log failure
            metrics = PipelineMetrics(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                iteration=iteration,
                n_assets=0,
                n_paths=0,
                total_time_sec=0.0,
                success=False,
                error_message=str(e)
            )
            monitor.log_metrics(metrics)
        
        if iteration < n_iterations - 1:
            time.sleep(update_interval)
    
    # Summary
    monitor.print_summary()
    monitor.save_full_report()
    monitor.export_prometheus()
    
    return monitor


if __name__ == "__main__":
    # Demo with synthetic metrics
    monitor = ResearchMonitor("demo_experiment")
    
    for i in range(3):
        metrics = PipelineMetrics(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            iteration=i,
            n_assets=2,
            n_paths=30000,
            total_time_sec=1.5 + 0.5 * i,
            max_calibration_error=0.05 + 0.02 * i,
            max_correlation_error=0.02 + 0.01 * i,
            avg_bound_width=0.5 + 0.1 * i,
            H_values=[0.08, 0.12],
            success=True
        )
        monitor.log_metrics(metrics)
        time.sleep(1)
    
    monitor.print_summary()
    monitor.save_full_report()
