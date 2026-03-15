"""
Entry point for the ML Usage Pattern Detection pipeline.

Usage:
  python main.py                          # incremental run, S3 Parquet source
  python main.py --athena                 # use Athena as data source
  python main.py --full-run               # ignore incremental state, reprocess all
  python main.py --config path/to/cfg.yaml
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(config: dict):
    level = getattr(logging, config["logging"]["level"], logging.INFO)
    log_file = config["logging"].get("file", "logs/pipeline.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)-40s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )


def print_cost_summary(report: dict):
    cost = report.get("cost_optimization", {})
    if not cost:
        return

    print("\n" + "=" * 68)
    print("  COST OPTIMIZATION OPPORTUNITIES")
    print("=" * 68)

    for category, items in cost.items():
        if not items:
            continue
        print(f"\n  [{category.upper().replace('_', ' ')}]")
        for item in items:
            print(f"\n    • {item['category']}")
            print(f"      Finding : {item['finding']}")
            print(f"      Action  : {item['action']}")
            print(f"      Saving  : {item['estimated_saving']}")
            if "implementation" in item:
                print(f"      Example : {item['implementation']}")

    print("\n" + "=" * 68)


def print_anomaly_summary(report: dict):
    alerts = report.get("anomaly_alerts", {}).get("service_alerts", [])
    if not alerts:
        print("\n  No anomaly alerts above threshold.")
        return
    print(f"\n  ANOMALY ALERTS ({len(alerts)} services)\n  " + "-" * 40)
    for a in alerts:
        print(
            f"  • {a['service']}  "
            f"score={a['anomaly_score']:.3f}  "
            f"err={a['error_rate']*100:.1f}%  "
            f"p95={a['latency_p95_ms']:.0f}ms"
        )
        print(f"    → {a['recommendation']}")


def print_capacity_summary(report: dict):
    recs = report.get("capacity_plan", {}).get("scale_recommendations", [])
    if not recs:
        return
    print(f"\n  CAPACITY RECOMMENDATIONS ({len(recs)} services)\n  " + "-" * 40)
    for r in recs[:10]:
        print(
            f"  • [{r['urgency'].upper()}] {r['service']}  "
            f"scale {r['recommended_scale_factor']:.1f}×"
        )
        print(f"    → {r['action']}")


def main():
    parser = argparse.ArgumentParser(description="ML Usage Pattern Detection Pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument(
        "--athena", action="store_true", help="Use Athena as data source (default: S3 Parquet)"
    )
    parser.add_argument(
        "--full-run", action="store_true", help="Reprocess all data (ignore incremental state)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    logger = logging.getLogger(__name__)
    logger.info(f"Config: {args.config}")

    from src.pipeline.orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator(config)
    results = orchestrator.run(use_athena=args.athena, force_full_run=args.full_run)

    if results.get("status") in ("no_new_data", "empty_after_parsing"):
        logger.info(results["status"])
        sys.exit(0)

    print_anomaly_summary(results)
    print_capacity_summary(results)
    print_cost_summary(results)

    meta = results.get("metadata", {})
    print(
        f"\n  Services: {meta.get('total_services_analysed', '?')}  |  "
        f"Requests: {meta.get('total_requests', 0):,}  |  "
        f"Range: {meta.get('date_range', {})}\n"
    )


if __name__ == "__main__":
    main()
