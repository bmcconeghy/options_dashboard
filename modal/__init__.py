from pathlib import Path

import modal

app = modal.App("options")

munge_local_path = Path(__file__).parent.parent / "polygon/munge.py"
munge_remote_path = "/root/munge.py"

fetch_local_path = Path(__file__).parent.parent / "polygon/fetch.py"
fetch_remote_path = "/root/fetch.py"

dashboard_local_path = Path(__file__).parent.parent / "polygon/app.py"
dashboard_remote_path = "/root/app.py"

panel_web_app_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "boto3",
        "botocore",
        "dash",
        "dash-bootstrap-components",
        "numpy",
        "pandas",
        "panel",
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
    image=panel_web_app_image,
    max_containers=1,
    secrets=[modal.Secret.from_name("polygon"), modal.Secret.from_name("dirs")],
    volumes={
        "/root/polygon/options_csvs": modal.Volume.from_name("options"),
        "/root/polygon/stocks_csvs": modal.Volume.from_name("stocks"),
    },
    startup_timeout=20,
)
@modal.concurrent(max_inputs=2)
@modal.web_server(8000)
def dashboard():  # Naming it this way so the URL is cleaner
    import subprocess

    subprocess.Popen(
        [
            "panel",
            "serve",
            "--port",
            "8000",
            "--address",
            "0.0.0.0",
            "--allow-websocket-origin",
            "*",
            dashboard_remote_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
