import enum
import os
from pathlib import Path

import boto3
from botocore.config import Config
from dateutil.parser import parse as dateutil_parser

BUCKET_NAME = "flatfiles"
ROOT_DIR = Path(os.environ.get("ROOT_DIR"))


def get_s3_client():
    """Create an S3 client for Polygon's S3-compatible storage."""
    session = boto3.Session(
        aws_access_key_id=os.environ.get("POLYGON_API_KEY_ID"),
        aws_secret_access_key=os.environ.get("POLYGON_API_SECRET_ACCESS_KEY"),
    )
    s3_client = session.client(
        "s3",
        endpoint_url="https://files.polygon.io",
        config=Config(signature_version="s3v4"),
    )
    return s3_client


class PolygonDataSourceBasePrefix(enum.StrEnum):
    """Possible S3 prefixes depending on the data you need."""

    global_crypto = enum.auto()
    global_forex = enum.auto()
    us_indices = enum.auto()
    us_options_opra = enum.auto()
    us_stocks_sip = enum.auto()


class PolygonDataSourceLevel(enum.StrEnum):
    """Possible S3 data levels depending on the data you need."""

    day_aggs_v1 = enum.auto()
    minute_aggs_v1 = enum.auto()
    quotes_v1 = enum.auto()
    trades_v1 = enum.auto()


def get_newest_flat_files_for_prefix(
    base_prefix: PolygonDataSourceBasePrefix = PolygonDataSourceBasePrefix.us_stocks_sip,
    level: PolygonDataSourceLevel = PolygonDataSourceLevel.minute_aggs_v1,
    bucket_name: str = BUCKET_NAME,
    num_files: int = 1,
    s3_client=None,
) -> list[str]:
    """Get the newest file for a given combination of base prefix and level."""
    if s3_client is None:
        s3_client = get_s3_client()
    paginator = s3_client.get_paginator("list_objects_v2")

    all_objects = [
        obj["Key"]
        for page in paginator.paginate(
            Bucket=bucket_name, Prefix=base_prefix + "/" + level
        )
        for obj in page.get("Contents", [])
    ]
    all_objects.sort(
        key=lambda filename: dateutil_parser(filename.split("/")[-1].split(".")[0]),
        reverse=True,
    )
    return all_objects[:num_files]


def download_file_from_s3(
    object_key: str,
    bucket_name: str = BUCKET_NAME,
    root_dir: str = ROOT_DIR,
    s3_client=None,
) -> Path:
    """Download a file from S3 to a local directory."""
    if s3_client is None:
        s3_client = get_s3_client()
    local_file_name = object_key.split("/")[-1]
    local_file_path = root_dir / local_file_name
    s3_client.download_file(bucket_name, object_key, local_file_path)
    return local_file_path
