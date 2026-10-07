# data/

Git ignores all files in this folder except this README. Do not commit datasets.

## 1. Synthetic data (default, no download)

`crash-whatif generate --out data/crashes.csv` writes a synthetic file. The tests and
`crash-whatif demo` make the same table in memory, so they need no file.

| Mode | Label | Use |
|---|---|---|
| `signal` (default) | Log-odds model with documented effects (`LOGIT_EFFECTS` in `synthetic.py`) | Check that the explanations find the true causes |
| `make_rule` | `severe` for every make except Maruti Suzuki, 2 % flips | Show the defect of the public file with `crash-whatif audit` |

The file uses the column names and the messy values of the public file: flags as `'1.0'`,
`'True'`, `'0.0'`, `'False'` or blank, and the text `'nan'` in some category columns.

## 2. Public synthetic file (optional)

| Item | Value |
|---|---|
| Source | "Synthetic Indian Automobile Crash Data", published on Kaggle |
| URL | https://www.kaggle.com/datasets/swish9/synthetic-indian-automobile-crash-data |
| Terms | Read the licence on the Kaggle page before you use or share the data |
| File | `Crash_Data.csv` (10,000 rows, 25 columns) |
| Target | `Crash_severity`: `severe`, `moderate`, `minor` |
| Personal data | None. All rows are synthetic |

```bash
crash-whatif audit --csv data/Crash_Data.csv     # shows that make alone decides the label
crash-whatif report --csv data/Crash_Data.csv
```

This file is synthetic, and its label is almost a rule on `Vehicle_make`. A result on it is
not a finding about real crashes.

## 3. Real crash data (planned)

Real crash records, for example NHTSA CRSS or FARS (USA) or STATS19 (Great Britain), have
different columns. Map them to the project columns in `schema.py` (`SOURCE_COLUMNS`) before use.

## 4. Columns

| Project column | Source column | Type |
|---|---|---|
| `make`, `vehicle_type`, `engine_type`, `transmission` | `Vehicle_make`, `Vehicle_type`, `Engine_type`, `Transmission_type` | category |
| `vehicle_year`, `engine_cc`, `cylinders` | `Vehicle_year`, `Engine_displacement`, `Number_of_cylinders` | number |
| `weight_kg`, `length_mm`, `width_mm`, `height_mm` | `Vehicle_weight`, `Vehicle_length`, `Vehicle_width`, `Vehicle_height` | number |
| `safety_rating`, `airbags` | `Safety_rating`, `Number_of_airbags` | number |
| `abs`, `esc`, `tcs`, `tpms` | `ABS_presence`, `ESC_presence`, `TCS_presence`, `TPMS_presence` | flag 0/1 |
| `location`, `weather`, `road_surface`, `time_of_day`, `day_of_week` | `Crash_location`, `Weather_conditions`, `Road_surface_conditions`, `Time_of_day`, `Day_of_week` | category |
| `driver_age`, `driver_gender` | `Driver_age`, `Driver_gender` | number, category |
| `severity` | `Crash_severity` | label |
