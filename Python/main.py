from fnmatch import translate
import re
import cv2 as cv
import numpy as np
import time
import chess
import com
import os

# Constants
LIVE = False
NEW_CALIBRATION_DATA = False
BIGNUMBER = 100
C_INCR = 1
C_DECR = 10
IMG_H = 480
IMG_W = 640
SQUARE_SIZE = 105  # Size of a chess square in pixels
MOVEMENT_THRESHOLD = 50
check = False
check_2 = False

# Parameters for the movement threshold
thresh_slider_max = 200
thresh_slider = 50
movement_threshold = thresh_slider
top_locations = []
count_threshold = 5

# List to store chess piece positions and corner points
piece_list = []
move_count = 0
location_count = {}
new_corner=[]


class Position:
    def __init__(self, num, letter):
        self.num = num
        self.letter = letter


class Piece:
    def __init__(self, nr, pos):
        self.nr = nr
        self.pos = pos


def on_trackbar(val):
    global movement_threshold
    movement_threshold = val


def crop_chessboard(frame, grid, offset=5):

    # Extract points for the four corners
    top_left = grid[0]                # Presumed top-left corner (0,0)
    top_right = grid[8]               # Presumed top-right corner (0,8)
    bottom_left = grid[72]            # Presumed bottom-left corner (8,0)
    bottom_right = grid[80]           # Presumed bottom-right corner (8,8)

    # Check if the corners are swapped
    if bottom_right[0] < top_left[0]:
        # Swap the corners if needed
        top_left, bottom_right = bottom_right, top_left
        top_right, bottom_left = bottom_left, top_right

    # Calculate the cropping coordinates with offsets
    x_start = int(min(top_left[0], bottom_left[0])) - offset
    x_end = int(max(top_right[0], bottom_right[0])) + offset
    y_start = int(min(top_left[1], top_right[1])) - offset
    y_end = int(max(bottom_left[1], bottom_right[1])) + offset

    # Crop the frame based on calculated coordinates
    if y_start < y_end and x_start < x_end:
        cropped_frame = frame[y_start:y_end, x_start:x_end]
    else:
        cropped_frame = frame[y_end-10:y_start+15, x_start+4:x_end-2]

    return cropped_frame, x_start, y_start  # Return cropped frame and offsets


def find_all_chessboard_corners():
    repeat = True
    while repeat == True:
        cap = cv.VideoCapture(0)
        ret, frame = cap.read()

        frame_resized = cv.resize(frame, (640, 480))
        key = cv.waitKey(10)
        if key == 27:  # 'Esc' to exit
            repeat = False
        while True:
            cv.imshow("xx", frame_resized)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            found, corners = cv.findChessboardCorners(gray, (7, 7))  # Adjust grid size if needed
            # if found:
            #     cv.drawChessboardCorners(frame, (7, 7), corners, found)
            #     
            #     cap.release()
            #     cv.destroyAllWindows()
            #     return corners.tolist()
            #     cv.imshow("xx", frame_resized)
            # else:
            #     break


def write_grid_to_file(grid, filename="grid.txt"):

    with open(filename, "w") as file:
        for i in range(9):
            # Write each row of the grid
            row = grid[i * 9 : (i + 1) * 9]
            formatted_row = " ".join(f"({x:.1f}, {y:.1f})" for x, y in row)
            file.write(formatted_row + "\n")
    print(f"Grid written to {filename}")


def read_grid_from_file(filename="grid.txt"):

    grid = []
    with open(filename, "r") as file:
        for line_number, line in enumerate(file, start=1):
            # Use regex to find all (x, y) pairs in the line
            matches = re.findall(r'\(([^,]+), ([^,]+)\)', line)
            row = []
            for match in matches:
                try:
                    x, y = match
                    row.append((float(x), float(y)))
                except ValueError as e:
                    print(f"Error parsing point '{match}' on line {line_number}: {e}")
            grid.extend(row)

    # Ensure the grid has 81 points for a 9x9 structure
    if len(grid) != 81:
        print(f"Error: The grid does not contain 81 points, found {len(grid)} instead.")
    else:
        print("Grid successfully read with 81 points.")

    return grid


def detect_movement(fgmask):
    contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.putText(fgmask, str(move_count), (20, 100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    bound_rects = [cv.boundingRect(contour) for contour in contours if cv.contourArea(contour) > movement_threshold]
    bound_rects = sorted(bound_rects, key=lambda rect: rect[2] * rect[3], reverse=True)
    return bound_rects[:2]  # Return only the two largest

def draw_points(grid_points, img):
    # for i, pt in enumerate(pointlist):
    #     x, y = pt[i]  # Extract the first (and only) row
    #     cv.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)  # Circle in green
    #     # cv.putText(img, str(i), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for (x, y) in grid_points:
        cv.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)  # Draw a green circle at each point


def init_piece_list():
    # Add white pawns

    piece_list.append(Piece("PAWN_W", Position(1+1, 'a')))
    piece_list.append(Piece("PAWN_W", Position(1+1, 'b')))
    piece_list.append(Piece("PAWN_W", Position(1+1, 'c')))
    piece_list.append(Piece("PAWN_W", Position(1+1, 'd')))
    piece_list.append(Piece("PAWN_W", Position(1+1, 'e')))
    piece_list.append(Piece("PAWN_W", Position(1+1, 'f')))
    piece_list.append(Piece("PAWN_W", Position(1+1, 'g')))
    piece_list.append(Piece("PAWN_W", Position(1+1, 'h')))

    # Add black pawns
    piece_list.append(Piece("PAWN_B", Position(6+1, 'a')))
    piece_list.append(Piece("PAWN_B", Position(6+1, 'b')))
    piece_list.append(Piece("PAWN_B", Position(6+1, 'c')))
    piece_list.append(Piece("PAWN_B", Position(6+1, 'd')))
    piece_list.append(Piece("PAWN_B", Position(6+1, 'e')))
    piece_list.append(Piece("PAWN_B", Position(6+1, 'f')))
    piece_list.append(Piece("PAWN_B", Position(6+1, 'g')))
    piece_list.append(Piece("PAWN_B", Position(6+1, 'h')))


    # Add rooks
    piece_list.append(Piece("ROOK_W", Position(0+1, 'a')))
    piece_list.append(Piece("ROOK_W", Position(0+1, 'h')))
    piece_list.append(Piece("ROOK_B", Position(7+1, 'a')))
    piece_list.append(Piece("ROOK_B", Position(7+1, 'h')))

    # Add knights
    piece_list.append(Piece("KNIGHT_W", Position(0+1, 'b')))
    piece_list.append(Piece("KNIGHT_W", Position(0+1, 'g')))
    piece_list.append(Piece("KNIGHT_B", Position(7+1, 'b')))
    piece_list.append(Piece("KNIGHT_B", Position(7+1, 'g')))

    # Add bishops
    piece_list.append(Piece("BISH_W", Position(0+1, 'c')))
    piece_list.append(Piece("BISH_W", Position(0+1, 'f')))
    piece_list.append(Piece("BISH_B", Position(7+1, 'c')))
    piece_list.append(Piece("BISH_B", Position(7+1, 'f')))

    # Add kings
    piece_list.append(Piece("KING_W", Position(0+1, 'd')))
    piece_list.append(Piece("KING_B", Position(7+1, 'd')))

    # Add queens
    piece_list.append(Piece("QUEEN_W", Position(0+1, 'e')))
    piece_list.append(Piece("QUEEN_B", Position(7+1, 'e')))


def find_movement(frame, boundRectList, cornerlist):
    # print("Detecting Movement")
    for rect in boundRectList:
        x, y, w, h = rect
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        square = find_chess_square(x, y, w, h, cornerlist)  # Convert to chessboard position
        #print(f"{x},{y} -- {w},{h}")
        if square not in piece_list and len(piece_list)<3:
            piece_list.append(square)
    

# Main function to set up video capture and processing loop
def main():

    global move_count
    start_time = time.time()

    cap = cv.VideoCapture(1)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # init_piece_list()

    # cv.namedWindow("Configuration")
    # cv.createTrackbar("Movement Threshold", "Configuration", thresh_slider, thresh_slider_max, on_trackbar)

    # Background subtractor for detecting movement
    bg_subtractor = cv.createBackgroundSubtractorMOG2()
    bg_subtractor.setBackgroundRatio(0.5)

    # Erosion element for noise reduction
    element = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))

    

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video capture!")
            break
        
        if time.time() - start_time >= 4:

            # Resize the frame if necessary
            frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
            frame_resized = cv.resize(frame, (480, 640))
            
            # Crop the chessboard area
            cropped_frame, x_start, y_start = crop_chessboard(frame_resized, extended_corners_full)
            new_grid(frame_resized, 9,9,extended_corners_full)
            

            # Apply background subtraction and erosion for movement detection
            fgmask = bg_subtractor.apply(cropped_frame)
            fgmask = cv.erode(fgmask, element)

        
            bound_rects = detect_movement(fgmask)
            # if bound_rects:
            for rect in bound_rects:
                x, y, w, h = rect
                cv.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 255), 2)

            if len(bound_rects) == 2:
                move_count += 1
            
            if move_count > MOVEMENT_THRESHOLD:
                find_movement(cropped_frame, bound_rects, extended_corners_full)  # Call movement finder

            if not bound_rects:
                move_count = 0

            if not bound_rects and len(piece_list) == 2:
                # print(piece_list)
                cap.release()
                cv.destroyAllWindows()
                return piece_list
            elif not bound_rects and len(piece_list) > 2:
                # print(piece_list)
                cap.release()
                cv.destroyAllWindows()
                return piece_list[:2]
                
                
                
            #cv.circle(cropped_frame, (77-60, 254-242), 8, (231, 255, 0), -1)
            #cv.circle(frame_resized, (115, 254), 8, (231, 255, 0), -1)
            #cv.circle(frame_resized, (160, 254), 8, (231, 255, 0), -1)
            #cv.circle(frame_resized, (204, 254), 8, (231, 255, 0), -1)

            
            
            # Display results
            
            #cv.imshow("Foreground Mask", frame_resized)
            cv.imshow("Foreground Mask", fgmask)
            cv.imshow("Configuration", cropped_frame)

        key = cv.waitKey(10) & 0xFF
        if key == 27:  # 'Esc' to exit
            cap.release()
            cv.destroyAllWindows()
            return ["e0","e0"]


    # Release resources
    cap.release()
    cv.destroyAllWindows()

def extend_corners_horizontally(corners):
    # Extend each row horizontally
    new_corner = []
    j = 0
    while j < 49:
        # Calculate the horizontal spacing for the left extension
        horizontal_spacing_front = corners[0 + j][0][0] + (corners[0 + j][0][0] - corners[1 + j][0][0])
        new_corner.append([horizontal_spacing_front, corners[0 + j][0][1]])

        # Append the original row points
        for i in range(7):
            new_corner.append([corners[i + j][0][0], corners[i + j][0][1]])

        # Calculate the horizontal spacing for the right extension
        horizontal_spacing_back = corners[6 + j][0][0] + (corners[6 + j][0][0] - corners[5 + j][0][0])
        new_corner.append([horizontal_spacing_back, corners[6 + j][0][1]])

        j += 7

    return np.array(new_corner).reshape((7, 9, 2))

def extend_corners_vertically(corners):
    # Extend each column vertically after horizontal extension
    extended_corners = []

    for i in range(9):  # 9 columns after horizontal extension
        new_column = []

        # Calculate vertical spacing for the top extension
        vertical_spacing_top = corners[0][i][1] + (corners[0][i][1] - corners[1][i][1])
        new_column.append([corners[0][i][0], vertical_spacing_top])

        # Append the existing points in the column
        for j in range(7):
            new_column.append([corners[j][i][0], corners[j][i][1]])

        # Calculate the vertical spacing for the bottom extension
        vertical_spacing_bottom = corners[6][i][1] + (corners[6][i][1] - corners[5][i][1])
        new_column.append([corners[6][i][0], vertical_spacing_bottom])

        extended_corners.append(new_column)

    # Transpose extended_corners to match the 9x9 format and flatten the grid
    extended_corners = np.array(extended_corners).transpose((1, 0, 2))
    flat_corners = extended_corners.reshape(-1, 2).tolist()  # Flatten to a list of 81 points

    return flat_corners

def new_grid(frame, row, col, new_corner):

    # Draw vertical lines for each column
    for i in range(col):  # Loop through each column (0 to 8)
        for j in range(row - 1):  # Draw lines within each column
            # Calculate the index of the current and next points in the same column
            start_index = j * col + i
            end_index = (j + 1) * col + i
            # Draw a line between consecutive points in the column
            start_point = (int(new_corner[start_index][0]), int(new_corner[start_index][1]))
            end_point = (int(new_corner[end_index][0]), int(new_corner[end_index][1]))
            cv.line(frame, start_point, end_point, (255, 0, 255), 2)

    # Draw horizontal lines for each row
    for i in range(row):  # Loop through each row (0 to 8)
        for j in range(col - 1):  # Draw lines within each row
            # Calculate the index of the current and next points in the same row
            start_index = i * col + j
            end_index = i * col + (j + 1)
            # Draw a line between consecutive points in the row
            start_point = (int(new_corner[start_index][0]), int(new_corner[start_index][1]))
            end_point = (int(new_corner[end_index][0]), int(new_corner[end_index][1]))
            cv.line(frame, start_point, end_point, (255, 0, 255), 2)

def print_grid_matrix(grid):
    if len(grid) != 81:
        print("Error: The grid does not contain 81 points.")
        return

    # Print the top header for column indices
    print("      ", end="")  # Initial padding for row index
    for col_index in range(9):
        print(f"   {col_index}    ", end="")
    print("\n" + "    " + "-" * 75)  # Divider line

    # Print each row with the row index at the beginning
    for i in range(9):  # 9 rows
        row = grid[i * 9 : (i + 1) * 9]  # Extract a row from the flattened list

        # Print the row index on the left
        print(f"{i} | ", end="")

        # Format each point in the row
        formatted_row = ["({:.1f}, {:.1f})".format(point[0], point[1]) for point in row]
        
        # Join and print the formatted row with spacing
        print("  ".join(formatted_row))

def get_live_calibration_data():
    if NEW_CALIBRATION_DATA == True:
        corners = find_all_chessboard_corners()
        extended_corners_horizontal = extend_corners_horizontally(corners)
        extended_corners_full = extend_corners_vertically(extended_corners_horizontal)
        write_grid_to_file(extended_corners_full)
    else:
        return read_grid_from_file("adjusted.txt")

def transpose_coordinates(coordinates):
    grid = [coordinates[i * 7:(i + 1) * 7] for i in range(7)]
    transposed_grid = [[grid[j][i] for j in range(7)] for i in range(7)]
    
    transposed_coordinates = [coord for row in transposed_grid for coord in row]
    return transposed_coordinates

def rearrange_coordinates(coordinate_list):

    if len(coordinate_list) != 81:
        raise ValueError("The input list must contain exactly 81 coordinates.")

    rearranged_list = []

    # Add the first 9 elements in reverse order
    for i in range(9):
        rearranged_list.append(coordinate_list[8 - i])  # Reverse the first row

    # Add the remaining rows in the specified order
    for i in range(9, 81, 9):
        for j in range(8, -1, -1):  # Add each row in reverse order
            rearranged_list.append(coordinate_list[i + j])

    return rearranged_list



def find_chess_square(x_cord, y_cord, w, h, grid, x_offset=60, y_offset=242):
    
    column_labels = "abcdefgh"
    row_labels = "12345678"
    #column_labels = "hgfedcba"
    #row_labels = "87654321"

    x = x_cord + w // 2
    y = y_cord + h // 2

    # Step 1: Determine the column (letter) based on x-coordinate
    column = None
    for col in range(8):  # Iterate over columns from 0 to 7
        for row in range(9):  # Check each row for column boundaries
            # Get x values for the two consecutive columns in the same row
            x_left = grid[row * 9 + col][0] - x_offset
            x_right = grid[row * 9 + (col + 1)][0] - x_offset
            # Check if x lies between these two column boundaries
            if x_left >= x >= x_right or x_left <= x <= x_right:
                # print("Hi")
                column = column_labels[col]
                break
        if column is not None:
            break
        # print(f"{x_left} - {x_right}")

    
    # Step 2: Determine the row (number) based on y-coordinate
    row = None
    for r in range(8):  # Iterate over rows from 0 to 7
        for c in range(9):  # Check each column for row boundaries
            # Get y values for the two consecutive rows in the same column
            y_top = grid[r * 9 + c][1] - y_offset
            y_bottom = grid[(r + 1) * 9 + c][1] - y_offset

            # Check if y lies between these two row boundaries
            if y_top >= y >= y_bottom or y_top <= y <= y_bottom:
                row = row_labels[7 - r]  # Reverse row index to match chess notation
                break
        if row is not None:
            break

    # Return the combined notation if both column and row are found
    if column is not None and row is not None:
        return f"{column}{row}"
    else:
        return None

def get_still_calibration_data():
    if NEW_CALIBRATION_DATA == True:
        quick_capture()
        
        corners = find_still_chessboard_corners("imgg.png")
        if (corners[0][0][1] - corners[9][0][1] < 3):
            corners = transpose_coordinates(corners)

        print(corners)
        extended_corners_horizontal = extend_corners_horizontally(corners)
        extended_corners_full = extend_corners_vertically(extended_corners_horizontal)
        
        # if (extended_corners_full[0][1] > extended_corners_full[80][1]):
        #     extended_corners_full = extended_corners_full[::-1]

        write_grid_to_file(extended_corners_full)
        print("New Data Aquired and Recorded")
    else:
        print("Existing Data Acquired")
        return read_grid_from_file("grid.txt")

def find_still_chessboard_corners(path):
    repeat = True
    while repeat == True:
        frame = cv.imread(path)

        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        frame_resized = cv.resize(frame, (480, 640))
        # frame_resized = cv.resize(frame, (640, 480))
        while True:
            gray = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)
            found, corners = cv.findChessboardCorners(gray, (7, 7))  # Adjust grid size if needed
            print(found)
            if found:
                cv.drawChessboardCorners(frame_resized, (7, 7), corners, found)
                # cv.imshow("Configuration", frame_resized)
                # key = cv.waitKey(10)
                # if key == 27:  # 'Esc' to exit
                #     # repeat = False
                return corners.tolist()
            else:
                break

def chess_square_to_coordinates(square, grid):

    # Map column letters and row numbers to indices
    column_labels = "abcdefgh"
    row_labels = "12345678"

    if len(square) != 2 or square[0] not in column_labels or square[1] not in row_labels:
        raise ValueError("Invalid square notation. Use format like 'a1' or 'h8'.")

    # Extract the column and row from the square notation
    col_index = column_labels.index(square[0])
    row_index = row_labels.index(square[1])

    # Calculate the coordinates based on the grid layout
    x_left = grid[row_index * 9 + col_index][0]
    y_top = grid[row_index * 9 + col_index][1]

    # Calculate center coordinates
    x_center = x_left + (grid[row_index * 9 + (col_index + 1)][0] - x_left) / 2
    y_center = y_top + (grid[(row_index + 1) * 9 + col_index][1] - y_top) / 2

    return int(x_center), int(y_center) 



def new():

    cap = cv.VideoCapture(0)
    
    if LIVE == True:
        
        while True:
            ret, frame = cap.read()

            frame_resized = cv.resize(frame, (640, 480))

            if not ret:
                print("Failed to capture image")
                cap.release()
                exit()
        
            cropped_frame, x_start, y_start = crop_chessboard(frame_resized, extended_corners_full)
            new_grid(frame_resized, 9,9,extended_corners_full)

            # print_grid_matrix(extended_corners_full)


            #cv.circle(cropped_frame, (152, 151), 3, (0, 255, 0), -1)  # Circle in green

            
            #cv.imshow("Original", frame_resized)
            cv.imshow("Final Cropped", cropped_frame)

            key = cv.waitKey(10)
            if key == 27:  # 'Esc' to exit
                break
            elif key == 13:  # 'Enter' to end)
                break

    else:
        ret, frame = cap.read()
        # frame = cv.imread("Images/img2.png")
        if not ret:
            print("Failed to capture image")
            cap.release()
            exit()

        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        frame_resized = cv.resize(frame, (480, 640))
        # frame_resized = cv.resize(frame, (640, 480))

    
        new_grid(frame_resized, 9,9,extended_corners_full)
        cropped_frame, x_start, y_start = crop_chessboard(frame_resized, extended_corners_full)

        # print_grid_matrix(extended_corners_full)


        x, y = chess_square_to_coordinates("e2", extended_corners_full)
        cv.circle(frame_resized, (x,y), 3, (0, 0, 0), -1)  # Circle in green
        
        x, y = chess_square_to_coordinates("e4", extended_corners_full)
        cv.circle(frame_resized, (x,y), 3, (0, 0, 0), -1)  # Circle in green


        # val = find_chess_square(x,y,x,y,extended_corners_full)


        cv.imshow("Original", cropped_frame)
        cv.waitKey(0)

def map_moves(move):
    origin = move[:2]
    destination = move[2:]
    # print(f"{origin} ---> {destination}")
    return origin, destination



def check_knight(inp):
    inp = inp[:2]
    origin_square = chess.parse_square(inp)
    piece = board.piece_at(origin_square)
    if piece.piece_type == chess.KNIGHT:
        print("Knight Detected")
        return "K"
    else:
        return ""

def check_capture(inp):
    origin = inp[:2]
    destination = inp[2:]

    origin_square = chess.parse_square(origin)
    destination_square = chess.parse_square(destination)

    move = chess.Move(origin_square, destination_square)

    if board.is_capture(move):
        return "X"
    else:
        return ""

def check_castling(inp):
    origin = inp[:2]
    destination = inp[2:]

    origin_square = chess.parse_square(origin)
    destination_square = chess.parse_square(destination)

    move = chess.Move(origin_square, destination_square)

    if board.is_castling(move):
        return "Y"
    else:
        return ""

def check_promotion(inp):
    origin = inp[:2]
    destination = inp[2:]

    origin_square = chess.parse_square(origin)
    destination_square = chess.parse_square(destination)

    move = chess.Move(origin_square, destination_square)

    if move.promotion:
        # If there is a promotion, it will be a piece type (like chess.QUEEN, chess.KNIGHT, etc.)
        return "P"
    else:
        return ""

def check_en_passant(inp, board):
    origin = inp[:2]
    destination = inp[2:]

    origin_square = chess.parse_square(origin)
    destination_square = chess.parse_square(destination)

    move = chess.Move(origin_square, destination_square)

    if move in board.legal_moves and board.is_en_passant(move):
        return "Z"
    else:
        return ""


def quick_capture():
    cap = cv.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
    ret,frame = cap.read() # return a single frame in variable `frame`

    while(True):
        cv.imshow('img',frame) #display the captured image
        if cv.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y' 
            cv.imwrite('imgg.png',frame)
            cv.destroyAllWindows()
            break
        if cv.waitKey(1) == 27 & 0xFF:
            cv.destroyAllWindows()
            break

    cap.release()




def chess_logic():

    #K for knight
    #X for capture
    #Y for Castling
    #Z for En Passan
    #P for Promotoion


    global piece_list, check, check_2
    print("Welcome to Terminal Chess!")
    print(board)
    print()

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            print("White's move")
            print("Enter your move in UCI format (e.g., e2e4): ")
            piece_list = []
            move_input = []
            move1 = ""
            move2 = ""
            move_input = main()
            #print(move_input)
            
            move1 = move_input[1] + move_input[0]
            move2 = move_input[0] + move_input[1]
            #move1 = input("Enter your move in UCI format (e.g., e2e4): ")

            try:
                move = chess.Move.from_uci(move1)
                if move in board.legal_moves:
                    board.push(move)
                    print(board)
                    print()
                else:
                    move = chess.Move.from_uci(move2)
                    if move in board.legal_moves:
                        board.push(move)
                        print(board)
                        print()
                    else:
                        print("Illegal move. Try again.")
                        print(move)
            except ValueError:
                print("Invalid move format. Please try again.")
        else:
            print("Black's move")
            move_input = ""
            move = ""
            destination = ""
            origin = ""
            # PC
            move_input = input("Enter your move in UCI format (e.g., e7e5): ")
            try:
                move = chess.Move.from_uci(move_input)
                if move in board.legal_moves:
                    origin, destination = map_moves(move_input)
                    case = check_knight(move_input)
                    #case = check_capture(move_input)
                    #case = check_castling(move_input)
                    #case = check_promotion(move_input)
                    #case = check_en_passant(move_input)

                    board.push(move)
                    while True:
                        response = com.read_response()
                        if response:
                            print(f"**Arduino** {response}")

                        if response == "Standby for Input..":
                            print("Received all")
                            com.send_move(f"{origin}\n".encode())
                            com.send_move(f"{destination}\n".encode())
                            if case:
                                com.send_move(f"{case}\n".encode())
                            else:
                                com.send_move(f"O\n".encode())
                            break
                    print(board)
                    print()
                else:
                    print("Illegal move. Try again.")
            except ValueError:
                print("Invalid move format. Please try again.")


        if board.is_check():
            print("Check!")

    # Display game result
    if board.is_checkmate():
        print("Checkmate! " + ("White" if board.turn == chess.BLACK else "Black") + " wins!")
    elif board.is_stalemate():
        print("Stalemate!")
    elif board.is_insufficient_material():
        print("Draw due to insufficient material.")
    elif board.is_seventyfive_moves():
        print("Draw due to the 75-move rule.")
    elif board.is_fivefold_repetition():
        print("Draw due to fivefold repetition.")
    elif board.is_variant_draw():
        print("Draw by variant-specific rules.")
    else:
        print("Game over.")


def quick_check():

    cap = cv.VideoCapture(2)

    while True:
        ret, frame = cap.read()
        cv.imshow("xx", frame)
        key = cv.waitKey(10)
        if key == 27:
            break
    cap.release()
    cv.destroyAllWindows()
    

def zfind_chess_square(x_cord, y_cord, w, h, grid):


    # Labels for chess notation columns and rows
    column_labels = "abcdefgh"  # Left to right for chess columns
    row_labels = "12345678"     # Bottom to top for chess rows

    # Adjust for the center of the rectangle
    x = x_cord + w // 2
    y = y_cord + h // 2

    # Find the column by checking x against grid boundaries
    column = None
    for col in range(8):
        x_left = grid[col][0]
        x_right = grid[col + 1][0]
        # Check if x falls within this column range
        if x_left >= x >= x_right or x_left <= x <= x_right:
            column = column_labels[col]
            break

    # Find the row by checking y against grid boundaries
    row = None
    for r in range(8):
        y_top = grid[r * 9][1]
        y_bottom = grid[(r + 1) * 9][1]
        # Check if y falls within this row range
        if y_top >= y >= y_bottom or y_top <= y <= y_bottom:
            row = row_labels[7 - r]  # Reverse for chessboard notation
            break

    # Return the result if both column and row are found
    if column and row:
        return f"{column}{row}"
    else:
        return None

    

if __name__ == "__main__":

    extended_corners_full = get_still_calibration_data()
    board = chess.Board()
    #val = main()

    chess_logic()
    #print(find_chess_square(400,260,12,20,extended_corners_full))
    #cv.circle(frame_resized, (80, 258), 8, (231, 255, 0), -1))



