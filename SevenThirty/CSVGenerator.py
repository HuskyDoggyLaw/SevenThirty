import csv
import random

# Function to generate random customer data
def generate_random_customer():
    names = ['Alice', 'Bob', 'Charlie', 'David', 'Emma', 'Frank', 'Grace', 'Hannah', 'Ivy', 'Jack']
    ages = list(range(18, 65))
    emails = ['example1@example.com', 'example2@example.com', 'example3@example.com']

    name = random.choice(names)
    age = random.choice(ages)
    email = random.choice(emails)

    return [name, age, email]

# File path to save the CSV
file_path = 'customer_data.csv'

# Number of customers to generate
num_customers = 100

# CSV header
header = ['Name', 'Age', 'Email']

# Generating customer data and writing to the CSV file
with open(file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(header)

    for _ in range(num_customers):
        customer_data = generate_random_customer()
        csv_writer.writerow(customer_data)

print(f"{num_customers} random customer records have been written to {file_path}.")

