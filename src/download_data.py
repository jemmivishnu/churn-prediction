"""Downloads the public IBM Telco Customer Churn dataset into data/."""

from pathlib import Path

import requests

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Telco-Customer-Churn.csv"


def main():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        print(f"Dataset already exists at {DATA_PATH}")
        return
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    DATA_PATH.write_bytes(response.content)
    print(f"Downloaded dataset to {DATA_PATH} ({len(response.content) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
