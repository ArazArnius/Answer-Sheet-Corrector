import numpy as np
import cv2
import os
import pandas as pd
import time

start = time.time()

class ClassScore:
    students = {}
    
    def __init__(self, name, answer_key, answer_sheet):
        self.name = name
        self.answer_key = answer_key # the key of the exam (correct answers)
        self.answer_sheet = cv2.imread(answer_sheet, 0) # the answers given by the students
        self.student_answer, self.representation = self.process_answer_sheet()
        self.answer_sheet_address = answer_sheet
        
    @classmethod
    def create_students(cls, files_list, answer_key, num_students=23):
        for n in range(num_students):
            student_name = "student_" + str(n+1)
            student = ClassScore(name=student_name, answer_key=answer_key, answer_sheet=files_list[n])
            cls.students[student_name] = student
            
    def score(self):
        return "{:.2f}".format(np.sum(self.student_answer == self.answer_key) / len(self.answer_key) * 100)
    
    def save_status(self):
        status = np.where(self.student_answer == self.answer_key, "True", "False")
        status = np.where(self.student_answer == 0, "-  ", status)
        return status
    
    def process_answer_sheet(self):
        gray_answer_sheet = cv2.resize(self.answer_sheet, (1634, 2388))
        im = cv2.cvtColor(gray_answer_sheet, cv2.COLOR_GRAY2BGR)
        image = np.copy(im)

        _, binary_image = cv2.threshold(gray_answer_sheet, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((2, 2), np.uint8)
        
        # Applying an opening morphology on the image will remove so many unwanted contours
        opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Sort contours cuz we most likely won't need the very big or very small contours
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Create a copy of the original image to draw the contours
        result_image = np.copy(image)
        
        # Initialize storage for the answers of the student
        answer = np.zeros(165, dtype=int)
        
        # Answer_indexes[i]; i represents i-th group of 50 questions ((i)*(50)+(question number)) 
        # And each of the elements in it is range of choices
        answer_indexes = np.array([[129,269], [387, 527], [644, 784], [900, 1040]])
        
        # This is the first pixel we are going to use to calculate the question number
        question_indexes = 635

        # In this program we will filter each contour based on area, points_count, their position, and whether it is mostly filled or not to get our desired data from the contours
        # Iterate through the contours but ignore the very big and very small ones (the numbers came from testing)
        for contour in sorted_contours[50:500]: # For this particular set of pictures [106:363] would be enough
            # Ignore contours with area less than 15*10 (small) or greater than 30*24 (big)
            area = cv2.contourArea(contour)
            if area < (22 * 10) or area > (30 * 24):
                continue

            # Find how much of the contour is filled with white (black in original photos) values :{
            # Get the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)
            center = [y + h // 2, x + w // 2] # We will use this to check where the contour is based
            rect_area = w * h
            white_pixel_count = cv2.countNonZero(binary_image[y:y+h, x:x+w])
            # Calculate the percentage of white pixels inside the bounding rectangle
            white_pixel_percentage = (white_pixel_count / rect_area) * 100
            # }
            
            points_count = len(contour)
            # Filter contours again based on the points count of each contour and also the position of the contour
            if (4 < points_count < 60) and (607 < y < 2181 and 63 < x < 1066):
                if white_pixel_percentage > 50: # Only go on if the contour is filled
                    # This will iterate through a list of 50 members filled with ranges of the rows of 1-50 questions ([start, end])
                    # We know for every 10 questions the image has another block so we add (32) for every 10 questions we process and each question has a range of 28 pixels
                    for q, q_n_index in enumerate([int(question_indexes + i // 10 * 32 + i * 28),int(question_indexes + i // 10 * 32 + (i+1) * 28)] for i in range(50)):
                        if q_n_index[0] < center[0] < q_n_index[1]: # Check row of the contour to find the question number
                            question_num = q + 1
                            
                            # Our answer index consists of 4 members representing a group of 50 questions we have to find out which group we are processing
                            for i, q_index  in enumerate(answer_indexes):
                                q_start, q_end = q_index[0], q_index[1]
                                
                                # Check column of the contour and correct the question number based on that
                                if q_start < center[1] < q_end:
                                    question_num += i * 50
                                    
                                    # Now we need to have indexes for each of these answer indexes
                                    # If we make a new index of each of these, showing start and end point of each choice (1,2,3,4),
                                    # We would be able to assign the filled choice of the student to its corresponding question number
                                    
                                    # explanation: [start,end], start point of choice 1 is q_ind + (i*(thresh)) => i is 0 for choice 1 and thresh is ((q_end - q_start)/4) 
                                    for j, a_index in enumerate([int(q_start + i * (q_end - q_start) / 4), int(q_start + (i + 1) * (q_end - q_start) / 4)] for i in range(4)):
                                        if a_index[0] < center[1] < a_index[1]:
                                            if answer[question_num - 1] == 0: # Check if it has not been checked before
                                                answer[question_num - 1] = j + 1
                                                break # To stop extra computation
                                            else:
                                                answer[question_num - 1] = 6 # Meaning wrong answer
                                                break
                                    
                            # Write the data of the contour on the image
                            if answer[question_num - 1] == self.answer_key[question_num - 1]:
                                cv2.drawContours(result_image, [contour], -1, (67, 255, 59), 2)
                            else:
                                cv2.drawContours(result_image, [contour], -1, (8, 8, 255), 2)
                                cv2.putText(result_image, f"  Correct:{self.answer_key[question_num-1]}", (center[::-1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                            
        # If you want to have the representation of all students, turn this on
        cv2.imwrite(f"result_{self.name}.png", result_image)
        print(f"Done ({self.name})")
        return answer, result_image
    
    @staticmethod
    def save_all_status():
        students = ClassScore.students.values()

        # Create the first row (header) 
        header = ['Name\Question Number'] + [str(i) for i in range(1, 166)]
        # This will give the name of the answer sheet using the address saved for each object
        answer_sheet_name = [student.answer_sheet_address.split('/')[-1].split('.')[0] for student in students]

        # Create the data
        data = []
        for i, student in enumerate(students):
            row = [answer_sheet_name[i]] + list(student.save_status())
            data.append(row)

        # Write the DataFrame to a CSV file
        df = pd.DataFrame(data, columns=header)
        filename = "Statuses.csv"
        df.to_csv(filename, index=False)
        print(f"\nData saved to {filename}.")

    @staticmethod
    def save_all():
        students = ClassScore.students.values()

        # Create the first row (header)
        header = ["Name", "Score"]
        # This will give the name of the answer sheet using the address saved for each object
        answer_sheet_name = [student.answer_sheet_address.split('/')[-1].split('.')[0] for student in students]

        # Create the data
        data = []
        for i, student in enumerate(students):
            row = [answer_sheet_name[i], student.score()]
            data.append(row)

        # Write the DataFrame to a CSV file
        df = pd.DataFrame(data, columns=header)
        filename = "Scores.csv"
        df.to_csv(filename, index=False)
        print(f"\nData saved to {filename}.")
        
    def __repr__(self):
        # Create the data
        data = {
            "Student": self.student_answer,
            "Key": self.answer_key,
            "Status": self.save_status()
        }

        # Print out the DataFrame
        df = pd.DataFrame(data, index=range(1, 166))
        pd.set_option("display.max_rows", 165)
        print("\n*", df)

        cv2.imwrite(f"result_{self.name}.png", self.representation)

        # If you want to have the csv file of the student, turn this on
        df.to_csv(f"{self.name}.csv", index_label="Q num")

        return f"\n*Representation of {self.name} printed out."

def process_answer_key(answer_key_org):
    answer_key = np.copy(answer_key_org)[435:,:] # [435:,:] is to remove the 2 unwanted contours on the top of the picture
    # Create a mask of all pixels with BGR color [14, 219, 123]
    mask = cv2.inRange(answer_key, (14, 219, 123), (14, 219, 123))
    # Find contours in the mask image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    # Fix the key.png file into a processable image (make indexes for each questions)(rb stands for reverse binarized)
    gray_ans = cv2.cvtColor(answer_key, cv2.COLOR_BGR2GRAY)
    _, rb_key = cv2.threshold(gray_ans[:, 50:112], 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((10,20))
    rb_key = cv2.dilate(rb_key, kernel, iterations=1)
    rb_key = cv2.cvtColor(rb_key, cv2.COLOR_GRAY2BGR) # To be able to paste it on the answer_key
    answer_key[:, 50:112] = rb_key
    
    question_col_index = 80 # The column in which all the question numbers are placed (our index)
    question_index = [] # Start and end row for each question
    key = np.zeros(165, dtype=int)
    
    # If you want to see what we process, turn this on
    # cv2.imwrite("KEY.png", answer_key)

    # Iterate through the rows of the specified column
    start_row = None
    end_row = None
    # Iterate through all the rows in column 80 to find the questions since questions are now distinct from one another
    for row in range(answer_key.shape[0]):
        temp = answer_key[row, question_col_index] 

        # The start and end point of our indexes are approximately equal to the answer's rows
        # Look for the start of a question (white value)
        if np.array_equal(temp, [255, 255, 255]) and start_row is None:
            start_row = row

        # Look for the end of a question (black value)
        elif np.array_equal(temp, [0, 0, 0]) and start_row is not None and end_row is None:
            end_row = row - 1
            question_index.append([start_row,end_row])

            # Set the variables for the next question
            start_row = None
            end_row = None
            
    # Now process Contours to create the key for the exam      
    for contour in contours:
        # Find the center of each contour
        moments = cv2.moments(contour)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        

        # Figure the question number from the question_index
        for i, (start_row, end_row) in enumerate(question_index):
            if start_row <= cy <= end_row:
                question_number = i # question number (in original photo) - 1
                break
                
        # Figure which choice is the correct answer based on the contours's center's columns
        # (The numbers are assigned manually)
        if 290 <= cx <= 340:
            key[question_number] = 1
        elif 705 <= cx <= 755:
            key[question_number] = 2
        elif 1116 <= cx <= 1167:
            key[question_number] = 3
        elif 1526 <= cx <= 1577:
            key[question_number] = 4

    return key

# Main():
files_list = ["./ResponseLetter/" + member for member in os.listdir("./ResponseLetter/")]

# Process the answer key
answer_key = cv2.imread(files_list[-1]) # In our case the last file in "./ResponseLetter/" directory named key.png
key = process_answer_key(answer_key)

print("\n***\n")
# Create a class for each student and save it in the ClassScore.students dictionary
ClassScore.create_students(files_list=files_list, answer_key=key, num_students=23)

ClassScore.save_all_status()
ClassScore.save_all()

# Get runtime
print(f"\n*Run time: {time.time() - start}s*")

while True:
    name = input("\nWhich student's status do you wish to check? (e.g. student_1 or 'stop')\n").lower()
    
    if name != "stop":
        if name in ClassScore.students:
            print(ClassScore.students[name])
        else:
            print("\nInvalid student name. Please try again.")
    else:
        print("\nThank you. See you later!\n\n***")
        break
