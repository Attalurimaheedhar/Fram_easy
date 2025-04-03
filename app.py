from flask import Flask, request, render_template, url_for # Added url_for import just in case, though used in template
import pickle
import numpy as np
import os
import mysql.connector
from mysql.connector import Error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'crop_recommendation_model.pkl')
app = Flask(__name__)

# --- Database Configuration ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Mahee2006.',
    'database': 'ImageDatabase'
}

def load_model():
    """Loads the trained ML model."""
    try:
        with open(MODEL_PATH, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

def get_db_connection():
    """Establishes a connection to the MySQL database."""
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        print(f"Error connecting to MySQL Database: {e}")
    return connection

def fetch_images_from_db():
    """Fetches RECENT image paths and descriptions from the database (for gallery view)."""
    # This function remains for the potential gallery on the home page
    images = []
    connection = get_db_connection()
    if connection is None:
        return images

    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)
        # Fetching latest images for potential use on home page if needed
        query = "SELECT image_path, description FROM images ORDER BY id DESC LIMIT 10"
        cursor.execute(query)
        images = cursor.fetchall()
        print(f"Fetched {len(images)} recent images from DB.")
    except Error as e:
        print(f"Error fetching recent images from database: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()
    return images

# --- NEW FUNCTION ---
def fetch_image_for_crop(crop_name):
    """Fetches image data for a specific crop from the database."""
    image_data = None # Return None if not found
    connection = get_db_connection()
    if connection is None:
        print("DB connection failed, cannot fetch image for crop.")
        return image_data

    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)

        # --- ADJUST SQL QUERY AS NEEDED ---
        # OPTION 1 (Better): If you have a 'crop_name' column
        # query = "SELECT image_path, description FROM images WHERE crop_name = %s LIMIT 1"
        # cursor.execute(query, (crop_name,))

        # OPTION 2 (Workaround): Search within the description field
        # Using LIKE with wildcards to find descriptions containing the crop name
        query = "SELECT image_path, description FROM images WHERE description LIKE %s LIMIT 1"
        search_term = f"%{crop_name}%" # e.g., %Rice%
        cursor.execute(query, (search_term,))

        image_data = cursor.fetchone() # Fetch only one matching row
        if image_data:
            print(f"Fetched image for crop '{crop_name}': {image_data['image_path']}")
        else:
            print(f"No image found for crop '{crop_name}' using search term '{search_term}'")

    except Error as e:
        print(f"Error fetching image for crop '{crop_name}': {e}")
    except Exception as e_gen:
        print(f"An unexpected error occurred fetching image for crop '{crop_name}': {e_gen}")
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()
            # print("MySQL connection is closed")
    return image_data # Returns a dictionary {image_path: ..., description: ...} or None


@app.route('/')
def home():
    # The home page might show recent images or nothing image-related initially
    # Let's fetch recent images for potential display if no prediction has been made
    recent_images = fetch_images_from_db()
    # Render the template initially without prediction or specific image data
    return render_template('recomend.html', images=recent_images) # Pass 'images' for potential gallery display

@app.route('/predict', methods=['POST'])
def predict():
    prediction_text = ""
    predicted_crop_name = None
    relevant_image_data = None # Initialize variable for the specific image

    if model is None:
         return render_template('recomend.html',
                                prediction_text="Error: Model could not be loaded.",
                                images=fetch_images_from_db()) # Show gallery on error maybe? Or pass None

    try:
        # Retrieve form inputs
        nitrogen = float(request.form.get('nitrogen', 0))
        phosphorus = float(request.form.get('phosphorus', 0))
        potassium = float(request.form.get('potassium', 0))
        temperature = float(request.form.get('temperature', 0))
        humidity = float(request.form.get('humidity', 0))
        ph = float(request.form.get('ph', 0))
        rainfall = float(request.form.get('rainfall', 0))

        # Basic validation (can be more robust)
        if not (0 <= nitrogen <= 200 and 0 <= phosphorus <= 200 and
                0 <= potassium <= 200 and 0 <= temperature <= 100 and
                0 <= humidity <= 100 and 0 <= ph <= 14 and
                0 <= rainfall <= 4000):
             prediction_text="Invalid input values. Please check ranges."
             # Fetch recent images to potentially display gallery even with input error
             recent_images = fetch_images_from_db()
             return render_template('recomend.html',
                                    prediction_text=prediction_text,
                                    images=recent_images) # Pass recent images

        else:
            # Prepare data and predict
            input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
            prediction = model.predict(input_data)
            predicted_crop_name = prediction[0] # Get the predicted crop name string
            prediction_text = f'Recommended Crop: {predicted_crop_name}'

            # --- FETCH RELEVANT IMAGE ---
            relevant_image_data = fetch_image_for_crop(predicted_crop_name)
            # relevant_image_data will be a dict {image_path: ..., description: ...} or None

    except ValueError:
        prediction_text = "Error: Please enter valid numeric values for all fields."
    except Exception as e:
        print(f"Prediction Error: {e}")
        prediction_text = f"An error occurred during prediction. Please try again."

    # Render the template with prediction result AND the specific image data (or None)
    # We no longer need to pass the full 'images' list here unless needed for another purpose
    return render_template('recomend.html',
                           prediction_text=prediction_text,
                           predicted_crop_name=predicted_crop_name, # Pass the name too
                           predicted_image_data=relevant_image_data) # Pass the single image dict (or None)


if __name__ == '__main__':
    static_folder = os.path.join(BASE_DIR, 'static')
    uploads_folder = os.path.join(static_folder, 'uploads') # Define uploads path
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)
    if not os.path.exists(uploads_folder): # Ensure static/uploads exists
        os.makedirs(uploads_folder)
    app.run(debug=True)