from pathlib import Path

import modal

app = modal.App("dash-app-test")

dashboard_local_path = Path(__file__).parent.parent / "polygon/stock_dashboard.py"
dashboard_remote_path = "/root/stock_dashboard.py"

munge_local_path = Path(__file__).parent.parent / "polygon/munge.py"
munge_remote_path = "/root/munge.py"

fetch_local_path = Path(__file__).parent.parent / "polygon/fetch.py"
fetch_remote_path = "/root/fetch.py"

web_app_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "boto3",
        "botocore",
        "dash",
        "dash-bootstrap-components",
        "numpy",
        "pandas",
        "pyarrow",
        "polars",
        "plotly",
        "plotly[express]",
        "structlog",
    )
    .add_local_file(
        dashboard_local_path,
        dashboard_remote_path,
    )
    .add_local_file(
        munge_local_path,
        munge_remote_path,
    )
    .add_local_file(
        fetch_local_path,
        fetch_remote_path,
    )
)


@app.function(
    image=web_app_image,
    max_containers=1,
    secrets=[modal.Secret.from_name("polygon"), modal.Secret.from_name("dirs")],
    volumes={
        "/root/polygon/options_csvs": modal.Volume.from_name("options"),
        "/root/polygon/stocks_csvs": modal.Volume.from_name("stocks"),
    },
    startup_timeout=20,
)
@modal.concurrent(max_inputs=2)
@modal.web_server(8050)
def ui():
    import subprocess

    subprocess.Popen(
        ["python", dashboard_remote_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
