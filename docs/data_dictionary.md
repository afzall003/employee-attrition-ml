# Employee Attrition Prediction — Data Dictionary

## 1. Dataset Overview

This dataset is intended for predictive modeling and analysis of
employee attrition.

The supplied dataset contains:

- 1,000 employee records in the provided CSV file
- 26 columns
- 25 employee attributes/identifier fields
- 1 target variable: `Attrition`

> Note: The accompanying dataset description states that the dataset
> contains 10,000 employees. The supplied CSV currently contains
> 1,000 records. This discrepancy is documented and has not been
> silently corrected.

---

## 2. Target Variable

| Field | Definition | Type | ML Role |
|---|---|---|---|
| `Attrition` | Indicates whether the employee left the company (`Yes`/`No`). | Categorical | Target |

Target distribution in the supplied CSV:

| Class | Records | Percentage |
|---|---:|---:|
| `No` | 811 | 81.1% |
| `Yes` | 189 | 18.9% |

The target is therefore imbalanced, with `No` representing the
majority class.

---

## 3. Employee Identifier

| Field | Definition | Type | ML Role |
|---|---|---|---|
| `Employee_ID` | Unique identifier for each employee. | Integer | Identifier; excluded from model features |

`Employee_ID` will be retained for employee-level identification
and prediction output but will not be used as a predictive feature.

---

## 4. Demographic Features

| Field | Definition | Type | ML Role |
|---|---|---|---|
| `Age` | Age of the employee. | Integer | Feature |
| `Gender` | Gender of the employee (`Female`/`Male`). | Categorical | Feature |
| `Marital_Status` | Marital status (`Single`, `Married`, `Divorced`). | Categorical | Feature |

---

## 5. Organizational and Job Features

| Field | Definition | Type | ML Role |
|---|---|---|---|
| `Department` | Department in which the employee works. | Categorical | Feature |
| `Job_Role` | Specific role within the department. | Categorical | Feature |
| `Job_Level` | Level in the organizational hierarchy. | Integer / Ordinal | Feature |
| `Years_at_Company` | Number of years the employee has been with the company. | Integer | Feature |
| `Years_in_Current_Role` | Number of years the employee has been in their current role. | Integer | Feature |
| `Years_Since_Last_Promotion` | Time since the employee's last promotion. | Integer | Feature |
| `Number_of_Companies_Worked` | Total number of companies the employee has worked for. | Integer | Feature |

---

## 6. Compensation Features

| Field | Definition | Type | ML Role |
|---|---|---|---|
| `Monthly_Income` | Monthly salary/income of the employee. | Integer | Feature |
| `Hourly_Rate` | Hourly rate for hourly employees. | Integer | Feature |

---

## 7. Satisfaction and Performance Features

| Field | Definition | Type | ML Role |
|---|---|---|---|
| `Work_Life_Balance` | Rating of the employee's work-life balance. | Integer / Ordinal | Feature |
| `Job_Satisfaction` | Job satisfaction rating on a 1–5 scale. | Integer / Ordinal | Feature |
| `Performance_Rating` | Performance rating on a 1–5 scale. | Integer / Ordinal | Feature |
| `Work_Environment_Satisfaction` | Rating of satisfaction with the work environment. | Integer / Ordinal | Feature |
| `Relationship_with_Manager` | Rating of the employee's relationship with their manager. | Integer / Ordinal | Feature |
| `Job_Involvement` | Rating of employee job involvement. | Integer / Ordinal | Feature |

---

## 8. Workload and Employee Activity Features

| Field | Definition | Type | ML Role |
|---|---|---|---|
| `Training_Hours_Last_Year` | Number of training hours completed during the previous year. | Integer | Feature |
| `Overtime` | Indicates whether the employee works overtime (`Yes`/`No`). | Categorical | Feature |
| `Project_Count` | Number of projects managed by the employee. | Integer | Feature |
| `Average_Hours_Worked_Per_Week` | Average number of hours worked per week. | Integer | Feature |
| `Absenteeism` | Number of days the employee was absent during the previous year. | Integer | Feature |
| `Distance_From_Home` | Distance from the employee's home to the workplace, in kilometers. | Integer | Feature |

---

## 9. Categorical Domains

The currently observed categorical values in the supplied dataset are:

### Gender

- `Female`
- `Male`

### Marital_Status

- `Single`
- `Married`
- `Divorced`

### Department

- `Finance`
- `HR`
- `IT`
- `Marketing`
- `Sales`

### Job_Role

- `Analyst`
- `Assistant`
- `Executive`
- `Manager`

### Overtime

- `No`
- `Yes`

### Attrition

- `No`
- `Yes`

These observed values will be validated during the data-quality
stage.

---

## 10. Preliminary Data-Quality Findings

The initial audit of the supplied CSV found:

- 0 missing values
- 0 duplicate rows
- 0 duplicate employee IDs
- 0 missing employee IDs
- No negative values in the currently checked numerical fields

The audit also identified records requiring investigation:

- `Years_in_Current_Role` greater than `Years_at_Company`
- `Years_Since_Last_Promotion` greater than `Years_at_Company`

These are currently treated as **investigation flags**, not automatic
data errors. They will not be removed or modified without a documented
business justification.

---

## 11. Modeling Principles

The following principles will be followed during model development:

1. `Employee_ID` will not be used as a predictive feature.
2. The target variable `Attrition` will be separated before model
   training.
3. Preprocessing will be fitted only on the training data.
4. Test data will remain isolated until final model evaluation.
5. Categorical variables will be encoded through a reproducible
   preprocessing pipeline.
6. Model performance will not be judged using accuracy alone because
   the target classes are imbalanced.
7. Data-quality findings will be documented rather than silently
   altering the raw dataset.
8. Any feature found to create target leakage will be excluded or
   appropriately handled.