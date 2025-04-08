import pandas as pd

# Load the data
df = pd.read_csv('survey_cleaned.csv')

# Function to clean 'Gender' entries
def clean_gender(gender):
    gender = str(gender).strip().lower()
    if "male" in gender or gender in ["m", "man", "cis male"]:
        return "Male"
    elif "female" in gender or gender in ["f", "woman", "cis female"]:
        return "Female"
    else:
        return None

# Function to clean 'no_employees' entries
def clean_no_employees(value):
    value = str(value).strip()
    # Check for valid ranges or keywords like 'More than'
    if "-" in value or "More than" in value:
        return value
    # Remove anything else (e.g., dates like 'Jun-25', '01-May')
    return None

# Apply cleaning functions
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].apply(clean_gender)
    df = df.dropna(subset=['Gender'])  # Remove rows with invalid genders

if 'no_employees' in df.columns:
    df['no_employees'] = df['no_employees'].apply(clean_no_employees)
    df = df.dropna(subset=['no_employees'])  # Remove rows with invalid employee ranges

# Save the cleaned dataset
df.to_csv('cleaned_survey_data.csv', index=False)

# Print a preview of the cleaned data
print(df.head())
