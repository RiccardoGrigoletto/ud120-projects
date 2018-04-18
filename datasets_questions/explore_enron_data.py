#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
import sys
sys.path.append("../final_project/")
import pickle
from poi_email_addresses import poiEmails 

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

poi = []
email_addresses_counter = 0
total_payments_counter = 0
total_poi_payments_counter = 0
max_total_payments = {"person":"","total_payments":0}
max_exercised_stock_options = {"person":"","exercised_stock_options":0}
min_exercised_stock_options = {"person":"","exercised_stock_options":-1}
max_salary = {"person":"","salary":0}
min_salary = {"person":"","salary":-1}
total_payments = []
salaries = []
for person_name in enron_data:
    if (person_name != "TOTAL"):
        if (enron_data[person_name]["poi"]==1): poi.append(enron_data[person_name])
        if (isinstance(enron_data[person_name]["total_payments"],int)):
            if (max_total_payments["total_payments"] < enron_data[person_name]["total_payments"]): 
                max_total_payments["person"] = person_name
                max_total_payments["total_payments"] = enron_data[person_name]["total_payments"]
        if (isinstance(enron_data[person_name]["exercised_stock_options"],int)):
            if (max_exercised_stock_options["exercised_stock_options"] < enron_data[person_name]["exercised_stock_options"]): 
                max_exercised_stock_options["person"] = person_name
                max_exercised_stock_options["exercised_stock_options"] = enron_data[person_name]["exercised_stock_options"]
            if (min_exercised_stock_options["exercised_stock_options"] == -1 
            or min_exercised_stock_options["exercised_stock_options"] > enron_data[person_name]["exercised_stock_options"]):
                min_exercised_stock_options["person"] = person_name
                min_exercised_stock_options["exercised_stock_options"] = enron_data[person_name]["exercised_stock_options"]
        if (isinstance(enron_data[person_name]["salary"],int)):
            if (max_salary["salary"] < enron_data[person_name]["salary"]): 
                max_salary["person"] = person_name
                max_salary["salary"] = enron_data[person_name]["salary"]
            if (min_salary["salary"] == -1 
            or min_salary["salary"] > enron_data[person_name]["salary"]):
                min_salary["person"] = person_name
                min_salary["salary"] = enron_data[person_name]["salary"]
        if isinstance(enron_data[person_name]["salary"],int):
            salaries.append(
                {"person":person_name,
                "salary":enron_data[person_name]["salary"],
                "email":enron_data[person_name]["email_address"]})
        if isinstance(enron_data[person_name]["total_payments"],int):
            total_payments.append(
                {"person":person_name,
                "total_payments":enron_data[person_name]["total_payments"],
                "email":enron_data[person_name]["email_address"]}) 
        if (enron_data[person_name]["email_address"]!="NaN"): 
            email_addresses_counter+=1

print("people number: %d"%len(enron_data))
print("POIs number: %d"%len(poi))

print("salaries: %d"%len(salaries))

print("email_addresses_counter: %d"%email_addresses_counter)

print("total_payments_counter: %f (percent)"%(len(total_payments)*100/len(enron_data)))
print("total_payments_counter: %f "%(len(total_payments)))

for tp in poi:
    if tp["total_payments"]: total_poi_payments_counter+=1
print("total_poi_payments_counter: %f (percent)"%(total_poi_payments_counter*100/len(poi)))

print("max_total_payments :")
print(max_total_payments)

print("max_exercised_stock_options: %.4f , min_exercised_stock_options: %.4f"%(max_exercised_stock_options["exercised_stock_options"],min_exercised_stock_options["exercised_stock_options"]))
print("max_salary: %.4f , min_salary: %.4f"%(max_salary["salary"],min_salary["salary"]))
