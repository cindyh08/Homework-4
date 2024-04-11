import sqlite3


conn = sqlite3.connect('your_database_name.db')
cursor = conn.cursor()

#This is the total amount of orders that took place on March 18,2023
cursor.execute("SELECT COUNT(Order_id) AS Total_Orders FROM SALES WHERE Date = '2023-03-18';")
total_orders_1 = cursor.fetchone()[0]

# People that have the  first name ‘John’ and last name Doe’
cursor.execute("SELECT COUNT(s.Order_id) AS Total_Orders FROM SALES s JOIN CUSTOMERS c ON s.Customer_id = c.customer_id WHERE s.Date = '2023-03-18' AND c.first_name = 'John' AND c.last_name = 'Doe';")
total_orders_2 = cursor.fetchone()[0]

#Average amount spent per person
cursor.execute("SELECT COUNT(DISTINCT s.Customer_id) AS Total_Customers, AVG(s.Revenue) AS Average_Spend_Per_Customer FROM SALES s WHERE s.Date >= '2023-01-01' AND s.Date < '2023-02-01';")
result_3 = cursor.fetchone()
total_customers_3 = result_3[0]
average_spend_per_customer_3 = result_3[1]

# For the department that generated less than 600 dollars
cursor.execute("SELECT department FROM ITEMS GROUP BY department HAVING SUM(price) < 600;")
departments_4 = cursor.fetchall()

# This is for the highest and lowest revenue generated by an order
cursor.execute("SELECT MAX(Revenue) AS Max_Revenue, MIN(Revenue) AS Min_Revenue FROM SALES;")
result_5 = cursor.fetchone()
max_revenue_5 = result_5[0]
min_revenue_5 = result_5[1]

# This is for the purchaes in the most lucrative order
cursor.execute("SELECT * FROM SALES WHERE Revenue = (SELECT MAX(Revenue) FROM SALES);")
orders_most_lucrative = cursor.fetchall()


print(" Here are the total Orders on March 18th, 2023:", total_orders_1)
print(" Total Orders by John Doe:", total_orders_2)
print(" Total Customers in January 2023:", total_customers_3)
print(" Average spending cost per person in January 2023:", average_spend_per_customer_3)
print(" This is for departments that have  less than $600 revenue in 2022:", departments_4)
print(" Highest Revenue:", max_revenue_5)
print(" Lowest Revenue:", min_revenue_5)
print(" Orders that are from the most lucrative order:", orders_most_lucrative)

cursor.close()
conn.close()