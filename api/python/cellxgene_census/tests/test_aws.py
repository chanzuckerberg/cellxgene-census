import pytest
import cellxgene_census
import os

def test_can_open_with_anonymous_access():
    os.environ["AWS_ACCESS_KEY_ID"] = "fake_id"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "fake_key"
    cellxgene_census.open_soma()