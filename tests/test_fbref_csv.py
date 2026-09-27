from pathlib import Path
import pandas as pd


def test_fbref_snapshot_files_are_readable():
    files = list((Path(__file__).parents[1] / "data" / "fbref_snapshots").glob("*.csv"))
    assert files
    frame = pd.read_csv(files[0])
    assert not frame.empty
