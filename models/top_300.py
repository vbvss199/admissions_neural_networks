import pandas as pd

# Load the admitted_application_data_sorted.xlsx file
file_path = '/Users/bhaskaravanacharla/Downloads/Documents/Machine_learning/admissionnn/models/admitted_application_data_sorted.xlsx'
data = pd.read_excel(file_path)

# Filter the rows where 'admitted' is 0 (not admitted)
not_admitted_applications = data[data['admitted'] == 0]

# Retrieve the top 300 rows (since the data is already sorted)
top_300_not_admitted = not_admitted_applications.head(300)

# Extract the Application Reference IDs
application_ids = top_300_not_admitted['Application Reference ID']

# Print the Application IDs
print(application_ids)

# Optionally, save the Application IDs to a new Excel file
application_ids.to_excel('top_300_not_admitted_application_ids.xlsx', index=False)
