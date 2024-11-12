import serial
import time

_ser_instance = None


def get_serial_instance(port='/dev/ttyUSB0', baudrate=115200, timeout=1):
# def get_serial_instance(port='/dev/cu.usbmodem14101', baudrate=115200, timeout=1):

    global _ser_instance
    if _ser_instance is None:
        _ser_instance = serial.Serial(port, baudrate, timeout=timeout)
        _ser_instance.flush()
    return _ser_instance
    

def send_move(move):
    ser = get_serial_instance()
    ser.write(move)
    time.sleep(0.1) 



def read_response():
    ser = get_serial_instance()
    line = ser.readline().decode('utf-8').rstrip()
    return line

def get_input():
    while True:
        buffer = input("Enter your input: ").strip()  # Read and trim input
        if buffer == "quit":
            print("Exiting...")
            exit()
        elif len(buffer) != 2 and len(buffer) > 0:
            print("INVALID INPUT!")
        elif len(buffer) == 2:
            if ord(buffer[0].lower()) < 97 or ord(buffer[0].lower()) > 104:
                print("INVALID LETTER!")
            else:
                if buffer[1].isdigit() and 1 <= int(buffer[1]) <= 8:
                    buffer = buffer.lower()
                    # print(f"Valid input: {buffer}")
                    return buffer
                else:
                    print("INVALID NUMBER!")


if __name__ == "__main__":  
    # while True:
    # Read message from Arduino
    response = read_response()
    if response:
        print(f"**Arduino** {response}")
    
    # If Arduino requests input, prompt user
    # if response == "Enter a command:":
    if response == "":
        origin = get_input()         # Get the origin
        destination = get_input()    # Get the destination

        print("Recived all")
        # Send the origin and destination sequentially
        send_move(f"{origin}\n".encode())
        send_move(f"{destination}\n".encode())
        # response=""
    



