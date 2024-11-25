import tkinter as tk
from tkinter import messagebox
import pickle

# Load the trained model and vectorizer
model_path = "spam_classifier.pkl"
vectorizer_path = "count_vectorizer.pkl"

with open(model_path, "rb") as model_file, open(vectorizer_path, "rb") as vectorizer_file:
    model = pickle.load(model_file)
    vectorizer = pickle.load(vectorizer_file)

# Function to classify email
def classify_email():
    email_text = email_input.get("1.0", tk.END).strip()  # Get the text input
    if not email_text:
        messagebox.showwarning("Input Error", "Please enter email text!")
        return

    # Transform the input text using the vectorizer
    email_features = vectorizer.transform([email_text])

    # Predict using the model
    prediction = model.predict(email_features)
    result = "Spam" if prediction[0] == 1 else "Ham"

    # Display the result
    result_label.config(text=f"Result: {result}", fg="green" if result == "Ham" else "red")

# Tkinter GUI setup
root = tk.Tk()
root.title("Email Spam Classifier")
root.geometry("500x400")

title_label = tk.Label(root, text="Email Spam Classifier", fg="Blue" ,font=("Helvetica", 16))
title_label.pack(pady=10)


email_label = tk.Label(root, text="Enter Email Text:", font=("DejaVu Sans Mono", 12))
email_label.pack(pady=5)

email_input = tk.Text(root, height=10, width=50, font=("Helvetica", 12))
email_input.pack(pady=10)


classify_button = tk.Button(root, text="Classify", font=("Helvetica", 12), fg="Blue", command=classify_email)
classify_button.pack(pady=10)


result_label = tk.Label(root, text="Result: ", font=("Helvetica", 14))
result_label.pack(pady=20)


root.mainloop()
