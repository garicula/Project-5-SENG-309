# Garrick Morley
# SENG 309 Project A4
# This program trains the data model

import joblib

from tkinter import *
#imports tkinter uses explicitly

# Load our trained model
model = joblib.load('insurance_charges.pkl')

ws = tkinter.Tk()
ws.title("House Model")
ws.geometry('500x150')

def sq_feet_value():
    sq_feet = value1.get()

value1 = Entry(ws, fg = "blue")
value1.pack(pady=30)

Button(
    ws,
    text="Enter a number", 
    padx=15, 
    pady=10,
    command=sq_feet_value
    ).pack()


# Define the charge in the same order of the training set
charge_1 = [
    19,        # Age
    female,    # Sex
    27.9,      # BMI
    0,         # Children
    yes,       # Smoker
    southwest, # Region
]

# Put the one charge we're estimating into the array
charges = [
    charge_1
]

# Make a prediction for the insurance charge in the array
insurance_charge = model.predict(charges)

# Get the first prediction returned
predicted_charge = insurance_charge[0]

# Print the results
print("Insurance Charge Details:")
print(f"- {charge_1[0]} Age")
print(f"- {charge_1[1]} Sex")
print(f"- {charge_1[2]} BMI")
print(f"- {charge_1[3]} Children")
print(f"- {charge_1[4]} Smoker")
print(f"- {charge_1[5]} Region")
print(f"Estimated charge: ${predicted_charge:,.2f}")

