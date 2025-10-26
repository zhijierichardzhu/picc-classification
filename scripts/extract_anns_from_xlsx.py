import os
import pandas as pd
from pathlib import Path
import numpy as np
from dateutil.parser import parse


def extract_phase_boundries_from_xlsx(xlsx_path: str | Path):
    def fillna_and_convert(series: pd.Series):
        return series.astype(str).replace(r'^\s*$', np.nan, regex=True).str.strip().replace(r'；', ':', regex=True)

    xlsx_df = pd.read_excel(xlsx_path, engine="calamine")
    
    start_time_series = fillna_and_convert(xlsx_df["起"]).iloc[:32]
    end_time_series = fillna_and_convert(xlsx_df["终"]).iloc[:32]

    return (
        [parse(time).time() if time not in [np.nan, "", " ", "nan"] else None for time in start_time_series.to_list()],
        [parse(time).time() if time not in [np.nan, "", " ", "nan"] else None for time in end_time_series.to_list()]
    )


if __name__ == "__main__":
    xlsx_dir = Path("dataset") / "excels"
    output_dir = Path("dataset") / "annotations"

    for xlsx_fn in os.listdir(xlsx_dir):
        # Skip state file of opened xlsx
        if xlsx_fn.startswith("~"):
            continue

        start_times, end_times = extract_phase_boundries_from_xlsx(xlsx_dir / xlsx_fn)

        processed_list = []

        for start, end in zip(start_times, end_times):
            if start is None or end is None:
                processed_list.append("\n")
                continue
            processed_list.append(f"{start.strftime('%H:%M:%S.%f')} {end.strftime('%H:%M:%S.%f')}\n")
        
        xlsx_stem = Path(xlsx_fn).stem.replace(" ", "_")
        with open(output_dir / f"{xlsx_stem}.txt", "w") as fp:
            fp.writelines(processed_list)
