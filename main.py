import cv2
import torch
import time
from collections import defaultdict
import tkinter as tk
from tkinter import Label, Frame, Text
from PIL import Image, ImageTk
import qrcode
import sqlite3
import mediapipe as mp

# Load YOLOv5 model from a specific checkpoint file
model_path = r'E:/opencv/yolov5/runs/runs/train/exp10/weights/best.pt'
repo_path = r'E:/opencv/yolov5'
model = torch.hub.load(repo_path, 'custom', path=model_path, source='local', force_reload=True)
model.eval()

# Initialize MediaPipe for hand gesture recognition
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Dictionary of object prices
prices = {
    'LU Oreo Biscuit 19gm': 5,
    'Bisconni Chocolate Chip Cookies 46.8gm': 50,
    'Coca Cola Can 250ml': 15,
    'Colgate Maximum Cavity Protection 75gm': 30,
    'Bisconni Chocolate Chip Cookies 46.8gm':50,
    'Fanta 500ml':30,
    'Fresher Guava Nectar 500ml':50,
    'Fruita Vitals Red Grapes 200ml':50,
    'Islamabad Tea 238gm':50,
    'Kolson Slanty Jalapeno 18gm':50,
    'Kurkure Chutney Chaska 62gm':50,
    'LU Candi Biscuit 60gm':20,
    'Lays Masala 34gm':20,
    'Lifebuoy Total Protect Soap 96gm':30,
    'Lipton Yellow Label Tea 95gm':50,
    'Meezan Ultra Rich Tea 190gm':20,
    'Peek Freans Sooper Biscuit 13.2gm':30,
    'Safeguard Bar Soap Pure White 175gm':10,
    'Sunsilk Shampoo Soft - Smooth 160ml':200,
    'Super Crisp BBQ 30gm':30,
    
    # Add more object-price pairs as needed
}

# Dictionary of recommendations
recommendations = {
    'LU Oreo Biscuit 19gm': ['Colgate Maximum Cavity Protection 75gm', 'Coca Cola Can 250ml'],
    'Bisconni Chocolate Chip Cookies 46.8gm': ['Coca Cola Can 250ml'],
    'Coca Cola Can 250ml': ['LU Oreo Biscuit 19gm', 'Bisconni Chocolate Chip Cookies 46.8gm'],
    'Colgate Maximum Cavity Protection 75gm': ['LU Oreo Biscuit 19gm']
}

# Dictionary to track detection times and permanent detected objects
detection_times = defaultdict(lambda: defaultdict(float))
permanent_detected_objects = defaultdict(lambda: {'price': 0, 'count': 0})
total_price = 0
object_display_threshold = 3  # Seconds
object_removal_threshold = 5  # Seconds after which an object starts being removed
object_removal_interval = 2 

# Initialize the SQLite3 database
def init_db():
    conn = sqlite3.connect('billing.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS billing (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id INTEGER,
                    item TEXT,
                    count INTEGER,
                    price REAL,
                    total_price REAL
                )''')
    conn.commit()
    conn.close()

# Function to insert billing details into the database
def insert_billing_data(customer_id, items, total_price):
    conn = sqlite3.connect('billing.db')
    c = conn.cursor()
    for item, info in items.items():
        c.execute('''INSERT INTO billing (customer_id, item, count, price, total_price) VALUES (?, ?, ?, ?, ?)''',
                  (customer_id, item, info['count'], info['price'], total_price))
    conn.commit()
    conn.close()

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Initialize the database
        init_db()
        
        # Set the window to full screen
        self.window.attributes('-fullscreen', True)
        self.window.bind("<Escape>", self.exit_fullscreen)

        # Set up webcam capture
        self.cap = cv2.VideoCapture(0)  # 0 for the default webcam, change to another number if you have multiple webcams
        
        # Check if the webcam is opened successfully
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            exit()
        
        self.main_frame = Frame(window, bg='#E1BEE7')
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=(50, 50))  # Added top and bottom padding
        
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        # Camera Column
        self.camera_column = Frame(self.main_frame, bg='#F3EAC0', bd=2, relief="solid")
        self.camera_column.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
        
        self.camera_label = Label(self.camera_column, text="Camera Feed", font=("Helvetica", 20, "bold"), bg='#F3EAC0', fg='#333333')
        self.camera_label.pack(fill="x", pady=10)
        
        self.camera_frame = Frame(self.camera_column, bg='#F3EAC0')
        self.camera_frame.pack(fill="both", expand=True)

        self.label = Label(self.camera_frame, bg='#F3EAC0')
        self.label.pack(fill="both", expand=True)

         # Billing Column
        self.billing_column = Frame(self.main_frame, bg='#B2DFDB', bd=2, relief="solid")
        self.billing_column.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')

        self.billing_label = Label(self.billing_column, text="Billing Information", font=("Helvetica", 20, "bold"), bg='#B2DFDB', fg='#333333')
        self.billing_label.pack(fill="x", pady=10)

        self.total_price_label = Label(self.billing_column, text=f'Total Price: Rs.{total_price}', font=("Helvetica", 14), bg='#B2DFDB', fg='#333333')
        self.total_price_label.pack(pady=(10, 20))

        self.detected_prices_text = Text(self.billing_column, wrap="word", font=("Helvetica", 16), bg='#ecf0f1', fg='#2c3e50', height=10, width=40)  # Changed font size to 16 and set fixed height and width
        self.detected_prices_text.pack(anchor='w', padx=10, pady=10, fill="both", expand=True)

        # Recommendation Column
        self.recommendation_column = Frame(self.main_frame, bg='#C0C0F2', bd=2, relief="solid")
        self.recommendation_column.grid(row=0, column=2, padx=5, pady=5, sticky='nsew')

        self.recommendation_label = Label(self.recommendation_column, text="Recommendations", font=("Helvetica", 20, "bold"), bg='#C0C0F2', fg='#333333')
        self.recommendation_label.pack(fill="x", pady=10)

        self.recommendation_text = Text(self.recommendation_column, wrap="word", font=("Helvetica", 16), bg='#ecf0f1', fg='#2c3e50', height=10, width=40)  # Changed font size to 16 and set fixed height and width
        self.recommendation_text.pack(expand=True, padx=10, pady=10, fill="both")


        self.qr_label = Label(self.billing_column, bg='#1abc9c')
        self.qr_label.pack(side='bottom', pady=10)

        # Instructions Label
        instructions = (
            "Instructions:\n"
            "1. Hold an object for 3 seconds to add it to the billing list.\n"
            "2. Hold an object for 5 seconds to remove it from the billing list.\n"
            "3. Show a thumbs-up gesture for 3 seconds to save the billing data to the database."
        )
        self.instructions_label = Label(
            self.main_frame,
            text=instructions,
            font=("Helvetica", 14, "bold"),  # Changed font size to 14 and set bold
            bg='#E1BEE7',
            fg='#333333',
            justify='center',
            wraplength=800,
            bd=2,
            relief="solid"
        )
        self.instructions_label.grid(row=1, column=0, columnspan=3, padx=5, pady=20, sticky='ew')
        self.thumbs_up_detected_time = None
        self.data_saved = False
        self.customer_id = None
        # Add notification label
        self.notification_label = Label(
            self.main_frame,
            text="",
            font=("Helvetica", 14, "bold"),
            bg='#E1BEE7',
            fg='#333333'
        )
        self.notification_label.grid(row=2, column=0, columnspan=3, padx=5, pady=20, sticky='ew')

        self.thumbs_up_detected_time = None
        self.data_saved = False
        self.customer_id = None

        self.update()

        
        # Adjust the weight of the columns to reduce their size
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(2, weight=1)

        self.window.mainloop()
    
    def exit_fullscreen(self, event=None):
        self.window.attributes("-fullscreen", False)

    def detect_thumbs_up(self, frame):
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Check if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get coordinates of thumb tip and thumb IP
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
                thumb_tip_coords = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
                thumb_ip_coords = (int(thumb_ip.x * frame.shape[1]), int(thumb_ip.y * frame.shape[0]))

                # Get coordinates of index finger MCP and middle finger MCP
                index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                index_finger_mcp_coords = (int(index_finger_mcp.x * frame.shape[1]), int(index_finger_mcp.y * frame.shape[0]))
                middle_finger_mcp_coords = (int(middle_finger_mcp.x * frame.shape[1]), int(middle_finger_mcp.y * frame.shape[0]))

                # Check if thumb tip is above thumb IP and to the left or right of index and middle finger MCPs
                if (thumb_tip_coords[1] < thumb_ip_coords[1] and
                    (thumb_tip_coords[0] > index_finger_mcp_coords[0] or thumb_tip_coords[0] < middle_finger_mcp_coords[0])):
                    return True
        return False


    def update(self):
        global total_price

        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame.")
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame)

        current_time = time.time()
        current_objects = []

        frame_height, frame_width, _ = frame.shape
        central_region_x_min = frame_width * 0.4
        central_region_x_max = frame_width * 0.6
        central_region_y_min = frame_height * 0.4
        central_region_y_max = frame_height * 0.6

        for *xyxy, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]
            price = prices.get(label, 0)

            bbox_center_x = (xyxy[0] + xyxy[2]) / 2
            bbox_center_y = (xyxy[1] + xyxy[3]) / 2

            if (central_region_x_min <= bbox_center_x <= central_region_x_max and
                central_region_y_min <= bbox_center_y <= central_region_y_max):

                if 'first_seen' not in detection_times[label]:
                    detection_times[label]['first_seen'] = current_time
                detection_times[label]['last_seen'] = current_time

                time_diff = current_time - detection_times[label]['first_seen']
                if 3 <= time_diff < object_removal_threshold:
                    if not detection_times[label].get('counted', False):
                        if label not in permanent_detected_objects:
                            permanent_detected_objects[label] = {'price': price, 'count': 1, 'last_removed': 0}
                        else:
                            permanent_detected_objects[label]['count'] += 1
                        total_price += price
                        detection_times[label]['counted'] = True
                elif time_diff >= object_removal_threshold:
                    if detection_times[label].get('counted', False):
                        if label in permanent_detected_objects:
                            # Handle removal with additional interval
                            last_removed_time = permanent_detected_objects[label]['last_removed']
                            if current_time - last_removed_time >= object_removal_interval:
                                count = permanent_detected_objects[label]['count']
                                if count > 1:
                                    permanent_detected_objects[label]['count'] -= 1
                                    total_price -= price
                                else:
                                    total_price -= price
                                    del permanent_detected_objects[label]
                                permanent_detected_objects[label]['last_removed'] = current_time

                text = f'{label} {conf:.2f} Price: Rs.{price}'
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, text, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                current_objects.append(label)

        items_to_remove = [label for label, info in permanent_detected_objects.items() if info['count'] == 0]
        for label in items_to_remove:
            del permanent_detected_objects[label]

        self.total_price_label.config(text=f'Total Price: Rs.{total_price}')
        self.detected_prices_text.delete('1.0', tk.END)

        all_detected_prices = []
        for label, info in permanent_detected_objects.items():
            count = info['count']
            price = info['price']
            formatted_price = f'{label} x{count} Price: Rs.{price * count}'
            all_detected_prices.append(formatted_price)

        self.detected_prices_text.insert(tk.END, '\n'.join(all_detected_prices))

        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(f'Total Price: Rs.{total_price}')
        qr.make(fit=True)
        img_qr = qr.make_image(fill='black', back_color='white')
        img_qr = img_qr.resize((150, 150), Image.LANCZOS)
        imgtk_qr = ImageTk.PhotoImage(image=img_qr)
        self.qr_label.imgtk = imgtk_qr
        self.qr_label.config(image=imgtk_qr)

        self.recommendation_text.delete('1.0', tk.END)
        if current_objects:
            first_detected_object = current_objects[0]
            related_products = recommendations.get(first_detected_object, [])
            self.recommendation_text.insert(tk.END, '\n'.join(related_products))

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)

        if self.detect_thumbs_up(frame):
            if self.thumbs_up_detected_time is None:
                self.thumbs_up_detected_time = current_time
            elif current_time - self.thumbs_up_detected_time >= 3:
                if self.customer_id is None:
                    self.customer_id = int(time.time())
                insert_billing_data(self.customer_id, permanent_detected_objects, total_price)
                print(f"Billing information saved to database for customer {self.customer_id}.")
                self.notification_label.config(text="Billing Stored in Database",fg='green')
                self.window.after(3000, self.clear_notification)  # Clear the notification after 3 seconds
                self.reset_billing_information()
        else:
            self.thumbs_up_detected_time = None

        outdated_labels = [label for label in detection_times.keys() if current_time - detection_times[label]['last_seen'] > object_display_threshold]
        for label in outdated_labels:
            del detection_times[label]

        self.window.after(10, self.update)

    def clear_notification(self):
        self.notification_label.config(text="")



    def reset_billing_information(self):
        global total_price
        total_price = 0
        permanent_detected_objects.clear()
        self.detected_prices_text.delete('1.0', tk.END)
        self.total_price_label.config(text=f'Total Price: Rs.{total_price}')
        self.recommendation_text.delete('1.0', tk.END)
        self.qr_label.config(image='')
        self.customer_id = None  # Reset customer ID



# Create a window and pass it to the Application object
App(tk.Tk(), "YOLOv5 Object Detection and Billing System")
