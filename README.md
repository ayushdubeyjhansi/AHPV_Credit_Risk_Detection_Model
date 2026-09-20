# AHPV_Credit_Risk_Detection_Model
Using **ANN** & **Fuzzy logic** to build and train a credit risk detection model help in approval of any kind of loan.

**Dataset**: [BANKING LOAN APPROVAL DATASET](https://www.kaggle.com/datasets/parthangare/banking-loan-approval-dataset) (kaggle).

**Team :** NAME NOT FOUND
- Ayush Dubey (24SCSE1410094) 
- Himanshu . (24SCSE1410339)
- Pradhuman Dixit (24SCSE1410272)
- Vinit Sharama (24SCSE1410006)
### Dataset Overview

| Column | Type | Description | Example Values |
| :--- | :--- | :--- | :--- |
| `age` | Integer | Age of the customer | `35`, `54` |
| `job` | Categorical | Type of employment | `management`, `technician`, `blue-collar` |
| `marital` | Categorical | Marital status | `single`, `married`, `divorced` |
| `education` | Categorical | Highest education level reached | `primary`, `secondary`, `tertiary`, `unknown` |
| `default` | Binary / Boolean | Credit default history | `yes`, `no` |
| `balance` | Numeric | Average yearly balance in account | `1500`, `-200`, `8500` |
| `housing_loan` | Binary / Boolean | Has an active housing loan | `yes`, `no` |
| `personal_loan` | Binary / Boolean | Has an active personal loan | `yes`, `no` |
| `contact` | Categorical | Communication channel used | `cellular`, `telephone`, `unknown` |
| `day` | Integer | Last contact day of the month | `1` to `31` |
| `month` | Categorical | Last contact month of the year | `jan`, `may`, `nov` |
| `duration` | Integer | Last contact duration (seconds) | `120`, `340` |
| `campaign` | Integer | Number of contacts during this campaign | `1`, `3`, `6` |
| `pdays` | Integer | Days passed after previous campaign contact (`-1` = not contacted) | `-1`, `99`, `180` |
| `previous` | Integer | Number of contacts performed before this campaign | `0`, `2`, `5` |
| `poutcome` | Categorical | Outcome of previous marketing campaign | `success`, `failure`, `other`, `unknown` |
| `LOAN_APPROVAL` | Binary  | Target label indicating final loan approval status | `1`, `0` |
