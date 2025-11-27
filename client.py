import requests
import json
import sys

# --- Configuration ---
# This must match the URL prefix in 'routes.py' and the Gunicorn host/port
API_URL = "http://localhost:5000/vinicius_rubens/api/predict"

def get_user_input() -> tuple[str, str] | tuple[None, None]:
    """
    Prompts the user for weight and size, with validation.
    Returns (None, None) if the user wants to exit.
    """
    
    # --- 1. Get Weight (with validation loop) ---
    while True:
        weight_str = input("\nEnter package weight in grams (e.g., 300.0) or 'sair' to exit: ")
        
        if weight_str.lower() in ['sair', 'exit']:
            return None, None # Exit signal
        
        try:
            # Check if it's a positive number.
            weight_val = float(weight_str)
            if weight_val <= 0:
                print("  [ERROR] Weight must be a positive number. Please try again.")
            else:
                break # Valid number, exit loop
        except ValueError:
            print("  [ERROR] Invalid input. Please enter a number (e.g., 300.0).")

    # --- 2. Get Size ---
    size_str = input("Enter package size ('Small Package' or 'Large Package') or 'sair' to exit: ")
    if size_str.lower() in ['sair', 'exit']:
        return None, None # Exit signal
    
    # Return strings, as Pydantic will handle the conversion
    return weight_str, size_str

def call_api(weight: str, size: str):
    """
    Builds the payload, calls the API, and prints the response.
    """
    
    payload = {
        "package_weight_gr": weight,
        "package_size": size
    }
    
    print(f"\nSending request to {API_URL}...")
    
    try:
        # Send the POST request with a 10-second timeout
        response = requests.post(API_URL, json=payload, timeout=10)
        
        # This will automatically raise an error for 4xx/5xx responses
        response.raise_for_status() 

        # --- Success ---
        print("\n--- ✅ API Success Response ---")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        
    except requests.exceptions.ConnectionError:
        print("\n--- ❌ API Connection Error ---")
        print(f"Error: Could not connect to the API. Is Gunicorn running at {API_URL}?")
    
    except requests.exceptions.Timeout:
        print("\n--- ❌ API Timeout Error ---")
        print("Error: The request timed out.")
        
    except requests.exceptions.HTTPError as e:
        # This handles 4xx (Bad Request, Validation) and 5xx (Server Error)
        print(f"\n--- ❌ API Error (HTTP {e.response.status_code}) ---")
        try:
            # Try to print the JSON error from our API
            print(json.dumps(e.response.json(), indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            # If the error is not JSON (e.g., a proxy error)
            print(e.response.text)
            
    except Exception as e:
        print(f"\n--- ❌ An Unexpected Error Occurred ---")
        print(f"Error: {e}")

def main():
    """
    Main application loop.
    """
    print("--- 🚀 API Prediction Client ---")
    print(f"Connecting to: {API_URL}")
    print("Type 'sair' or 'exit' at any prompt to quit.")
    
    while True:
        weight, size = get_user_input()
        
        if weight is None: # Exit signal
            break
            
        call_api(weight, size)
        print("-" * 40) # Separator

    print("\nExiting client. Goodbye!")

if __name__ == "__main__":
    main()