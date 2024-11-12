#define DEBUG

// X AXIS
#define MOTOR1_STEP 12
#define MOTOR1_DIR 11

// Y AXIS
#define MOTOR2_STEP 10
#define MOTOR2_DIR 9

#define MOTOR3_STEP 8
#define MOTOR3_DIR 7


#define MAX_X_POSITION 2320
#define MIN_X_POSITION 0

#define MAX_Y_POSITION 2320
#define MIN_Y_POSITION 0


#define INTERVAL 1000

#define MY_RELAY 6
#define MY_SWITCH 5

#define LIM_SWITCH_X 4
#define LIM_SWITCH_Y1 3
#define LIM_SWITCH_Y2 2

#define MY_LED1 A4
#define MY_LED2 A1
#define MY_LED3 A2
#define MY_LED4 A3

#define HOMING_SPEED 500

////////////////////////////////////////////////
#define HALF_SQUARE 139
////////////////////////////////////////////////
#define CENTER_SQUARE_1 194
#define CENTER_SQUARE_2 480
#define CENTER_SQUARE_3 766
#define CENTER_SQUARE_4 1052
#define CENTER_SQUARE_5 1338
#define CENTER_SQUARE_6 1623
#define CENTER_SQUARE_7 1910
#define CENTER_SQUARE_8 2196
////////////////////////////////////////////////
////////////////////////////////////////////////
#define START_EDGE_SQUARE_1 49
#define START_EDGE_SQUARE_2 327
#define START_EDGE_SQUARE_3 605
#define START_EDGE_SQUARE_4 882
#define START_EDGE_SQUARE_5 1160
#define START_EDGE_SQUARE_6 1438
#define START_EDGE_SQUARE_7 1715
#define START_EDGE_SQUARE_8 1993

#define END_EDGE_SQUARE_1 327
#define END_EDGE_SQUARE_2 605
#define END_EDGE_SQUARE_3 882
#define END_EDGE_SQUARE_4 1160
#define END_EDGE_SQUARE_5 1438
#define END_EDGE_SQUARE_6 1715
#define END_EDGE_SQUARE_7 1993
#define END_EDGE_SQUARE_8 2271

struct Coordinates{
  short x;
  short y;
};

struct Pieces{
  String name;
  Coordinates coordinate;
};

Coordinates initialLocation, finalLocation;


// static unsigned long previousMillis = 0;  // will store last time LED was updated

bool finished = false;
String thePiece = "";
String origin = "";
String destination = "";
char specialCase;


Pieces piece[16];


bool x_flag = false;
bool y_flag = false;


