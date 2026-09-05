from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_EMPLOYEES = 1000

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "employee_attrition_dataset_v2.csv"
)


# ============================================================
# RANDOM GENERATOR
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clipped_normal(
    mean: float,
    std: float,
    minimum: float,
    maximum: float,
    size: int = N_EMPLOYEES,
) -> np.ndarray:
    values = rng.normal(
        mean,
        std,
        size,
    )

    return np.clip(
        values,
        minimum,
        maximum,
    )


# ============================================================
# EMPLOYEE BASIC INFORMATION
# ============================================================

employee_id = np.arange(
    1,
    N_EMPLOYEES + 1,
)

age = np.rint(
    clipped_normal(
        38,
        9,
        22,
        60,
    )
).astype(int)

gender = rng.choice(
    ["Male", "Female"],
    size=N_EMPLOYEES,
    p=[0.55, 0.45],
)

marital_status = rng.choice(
    ["Single", "Married", "Divorced"],
    size=N_EMPLOYEES,
    p=[0.34, 0.52, 0.14],
)

department = rng.choice(
    [
        "IT",
        "Sales",
        "Marketing",
        "Finance",
        "HR",
    ],
    size=N_EMPLOYEES,
    p=[
        0.28,
        0.24,
        0.18,
        0.18,
        0.12,
    ],
)

job_role = rng.choice(
    [
        "Analyst",
        "Assistant",
        "Executive",
        "Manager",
    ],
    size=N_EMPLOYEES,
    p=[
        0.35,
        0.25,
        0.25,
        0.15,
    ],
)

job_level = rng.choice(
    [1, 2, 3, 4, 5],
    size=N_EMPLOYEES,
    p=[
        0.25,
        0.30,
        0.25,
        0.13,
        0.07,
    ],
)


# ============================================================
# TENURE — GENERATED CONSISTENTLY
# ============================================================

# Maximum possible company tenure based on age.
max_company_years = np.maximum(
    age - 21,
    1,
)

years_at_company = np.rint(
    clipped_normal(
        8,
        6,
        0,
        25,
    )
).astype(int)

years_at_company = np.minimum(
    years_at_company,
    max_company_years,
)

# Current-role tenure can NEVER exceed company tenure.
years_in_current_role = np.zeros(
    N_EMPLOYEES,
    dtype=int,
)

for i in range(N_EMPLOYEES):

    company_years = years_at_company[i]

    if company_years == 0:
        years_in_current_role[i] = 0
    else:
        years_in_current_role[i] = rng.integers(
            0,
            company_years + 1,
        )


# Promotion interval can NEVER exceed company tenure.
years_since_last_promotion = np.zeros(
    N_EMPLOYEES,
    dtype=int,
)

for i in range(N_EMPLOYEES):

    company_years = years_at_company[i]

    if company_years == 0:
        years_since_last_promotion[i] = 0
    else:
        years_since_last_promotion[i] = rng.integers(
            0,
            company_years + 1,
        )


# ============================================================
# COMPENSATION
# ============================================================

monthly_income = (
    4500
    + job_level * 2200
    + years_at_company * 180
    + rng.normal(
        0,
        1800,
        N_EMPLOYEES,
    )
)

monthly_income = np.clip(
    monthly_income,
    3000,
    30000,
).round().astype(int)

hourly_rate = np.rint(
    clipped_normal(
        32,
        10,
        15,
        60,
    )
).astype(int)


# ============================================================
# JOB EXPERIENCE / PERFORMANCE
# ============================================================

work_life_balance = rng.choice(
    [1, 2, 3, 4],
    size=N_EMPLOYEES,
    p=[
        0.12,
        0.28,
        0.42,
        0.18,
    ],
)

job_satisfaction = rng.choice(
    [1, 2, 3, 4],
    size=N_EMPLOYEES,
    p=[
        0.12,
        0.27,
        0.42,
        0.19,
    ],
)

performance_rating = rng.choice(
    [1, 2, 3, 4],
    size=N_EMPLOYEES,
    p=[
        0.08,
        0.35,
        0.45,
        0.12,
    ],
)

training_hours = np.rint(
    clipped_normal(
        45,
        18,
        10,
        100,
    )
).astype(int)

project_count = np.rint(
    clipped_normal(
        5,
        2,
        1,
        10,
    )
).astype(int)


# ============================================================
# WORKING CONDITIONS
# ============================================================

overtime = rng.choice(
    ["Yes", "No"],
    size=N_EMPLOYEES,
    p=[
        0.28,
        0.72,
    ],
)

average_hours = np.rint(
    clipped_normal(
        43,
        6,
        30,
        60,
    )
).astype(int)

# Overtime employees work somewhat more hours.
average_hours = np.where(
    overtime == "Yes",
    np.minimum(
        average_hours + rng.integers(
            2,
            8,
            N_EMPLOYEES,
        ),
        60,
    ),
    average_hours,
)

absenteeism = np.rint(
    clipped_normal(
        8,
        5,
        0,
        25,
    )
).astype(int)

work_environment_satisfaction = rng.choice(
    [1, 2, 3, 4],
    size=N_EMPLOYEES,
    p=[
        0.12,
        0.28,
        0.42,
        0.18,
    ],
)

relationship_with_manager = rng.choice(
    [1, 2, 3, 4],
    size=N_EMPLOYEES,
    p=[
        0.10,
        0.25,
        0.45,
        0.20,
    ],
)

job_involvement = rng.choice(
    [1, 2, 3, 4],
    size=N_EMPLOYEES,
    p=[
        0.08,
        0.25,
        0.48,
        0.19,
    ],
)

distance_from_home = np.rint(
    clipped_normal(
        20,
        12,
        1,
        60,
    )
).astype(int)

number_of_companies = np.rint(
    clipped_normal(
        3,
        2,
        1,
        10,
    )
).astype(int)


# ============================================================
# ATTRITION RISK GENERATION
# ============================================================

# We deliberately create a probabilistic relationship
# between employee characteristics and attrition.
#
# This is NOT a deterministic rule.
# Noise remains so the resulting ML problem is realistic.

risk = np.full(
    N_EMPLOYEES,
    -2.75,
    dtype=float,
)

# Overtime
risk += np.where(
    overtime == "Yes",
    0.65,
    0.0,
)

# Job satisfaction
risk += (
    3 - job_satisfaction
) * 0.30

# Work-life balance
risk += (
    3 - work_life_balance
) * 0.25

# Manager relationship
risk += (
    3 - relationship_with_manager
) * 0.20

# Job involvement
risk += (
    3 - job_involvement
) * 0.20

# Long working hours
risk += np.maximum(
    average_hours - 42,
    0,
) * 0.045

# Absenteeism
risk += absenteeism * 0.025

# Distance
risk += distance_from_home * 0.008

# Years since promotion
risk += years_since_last_promotion * 0.035

# Long stagnation in current role
risk += years_in_current_role * 0.015

# Lower training
risk += np.maximum(
    45 - training_hours,
    0,
) * 0.012

# Lower income
risk += np.maximum(
    12000 - monthly_income,
    0,
) / 12000 * 0.20

# Small department/role effects
risk += np.where(
    department == "Sales",
    0.10,
    0.0,
)

risk += np.where(
    job_role == "Assistant",
    0.08,
    0.0,
)

# Random individual-level variation.
risk += rng.normal(
    0,
    0.45,
    N_EMPLOYEES,
)


# Logistic probability
probability = 1 / (
    1 + np.exp(-risk)
)

attrition = np.where(
    rng.random(N_EMPLOYEES)
    < probability,
    "Yes",
    "No",
)


# ============================================================
# BUILD DATAFRAME
# ============================================================

df = pd.DataFrame(
    {
        "Employee_ID": employee_id,
        "Age": age,
        "Gender": gender,
        "Marital_Status": marital_status,
        "Department": department,
        "Job_Role": job_role,
        "Job_Level": job_level,
        "Monthly_Income": monthly_income,
        "Hourly_Rate": hourly_rate,
        "Years_at_Company": years_at_company,
        "Years_in_Current_Role": years_in_current_role,
        "Years_Since_Last_Promotion": years_since_last_promotion,
        "Work_Life_Balance": work_life_balance,
        "Job_Satisfaction": job_satisfaction,
        "Performance_Rating": performance_rating,
        "Training_Hours_Last_Year": training_hours,
        "Overtime": overtime,
        "Project_Count": project_count,
        "Average_Hours_Worked_Per_Week": average_hours,
        "Absenteeism": absenteeism,
        "Work_Environment_Satisfaction": work_environment_satisfaction,
        "Relationship_with_Manager": relationship_with_manager,
        "Job_Involvement": job_involvement,
        "Distance_From_Home": distance_from_home,
        "Number_of_Companies_Worked": number_of_companies,
        "Attrition": attrition,
    }
)


# ============================================================
# FINAL VALIDATION
# ============================================================

assert len(df) == N_EMPLOYEES

assert df["Employee_ID"].is_unique

assert df["Years_in_Current_Role"].le(
    df["Years_at_Company"]
).all()

assert df["Years_Since_Last_Promotion"].le(
    df["Years_at_Company"]
).all()

assert df["Years_at_Company"].le(
    df["Age"] - 21
).all()

assert df.isna().sum().sum() == 0


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print(
    "EMPLOYEE ATTRITION — DATASET V2 GENERATED"
)
print("=" * 60)

print(
    f"Rows:                 {len(df)}"
)

print(
    f"Columns:              {len(df.columns)}"
)

print(
    f"Attrition Yes:        {(df['Attrition'] == 'Yes').sum()}"
)

print(
    f"Attrition No:         {(df['Attrition'] == 'No').sum()}"
)

print(
    f"Attrition rate:       "
    f"{(df['Attrition'] == 'Yes').mean() * 100:.2f}%"
)

print(
    f"Output:               {OUTPUT_PATH}"
)

print(
    "\nTenure consistency checks: PASSED"
)

print(
    "Missing-value check:        PASSED"
)

print(
    "Duplicate-ID check:         PASSED"
)

print("\n" + "=" * 60)