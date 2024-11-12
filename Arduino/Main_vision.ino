#include "Squares_m.h"
#include <AccelStepper.h>



AccelStepper stepper1(1, MOTOR1_STEP, MOTOR1_DIR);  // (Type of driver: with 2 pins, STEP, DIR)
AccelStepper stepper2(1, MOTOR2_STEP, MOTOR2_DIR); // (Type of driver: with 2 pins, STEP, DIR)
AccelStepper stepper3(1, MOTOR3_STEP, MOTOR3_DIR); // (Type of driver: with 2 pins, STEP, DIR)

enum MovementState {HOMING, WAITING, GETTING_LOCATIONS, MOVING, COMPLETED};
MovementState currentState = HOMING;


void setup() {
  Serial.begin(115200);
  
  pinMode(MY_SWITCH, INPUT_PULLUP);
  pinMode(MY_RELAY, OUTPUT);

  pinMode(LIM_SWITCH_X, INPUT_PULLUP);
  pinMode(LIM_SWITCH_Y1, INPUT_PULLUP);
  pinMode(LIM_SWITCH_Y2, INPUT_PULLUP);

  pinMode(MY_LED1, OUTPUT);


  stepper1.setMaxSpeed(1000);
  stepper1.setAcceleration(1000);

  stepper2.setMaxSpeed(1000);
  stepper2.setAcceleration(1000);

  stepper3.setMaxSpeed(1000);
  stepper3.setAcceleration(1000);

  delay(1000);

  Serial.println("");
  

}


void loop() {


  stateMachine();
  // digitalWrite(6, LOW);
  // delay(2000);
  // digitalWrite(6, HIGH);
  // delay(2000);



}

inline void stateMachine(){
  switch (currentState) {
    case HOMING:
      // Serial.println("Initiating Homing");
      if (!x_flag || !y_flag){
        homing();
      }
      else if (x_flag && y_flag){
        currentState = WAITING;
      }
      break;
    case WAITING:
      // Wait until the conditions are met to move to the next state
      Serial.println("Standby for Input..");
      if (initialLocation.x == 0 && finalLocation.x == 0) {
        currentState = GETTING_LOCATIONS;
      }
      break;

    case GETTING_LOCATIONS:
      
      getLocations();
      if (initialLocation.x != 0 && finalLocation.x != 0) {
        // Serial.println("Origin command received: " + origin);
        // Serial.println("Destination command received: " + destination);
        // Serial.println("Special Case command received: " + specialCase);
        origin = "";
        destination = "";
        // specialCase = "";
        // Serial.println("Ack");
        currentState = MOVING;  
      }
      break;

    case MOVING:
      if (specialCase == 'K'){
        finished = knightException(finalLocation, initialLocation);
      }
      else if (specialCase == 'X'){
        finished = captureException(finalLocation, initialLocation);
      }
      else {
        finished = movePiece();  // Move the crane and set `finished` when done
      }

      if (finished) {
        currentState = COMPLETED;  // Movement completed, change state
      }
      break;

    case COMPLETED:
      // Serial.println("Completed..");
      initialLocation.x = 0;
      finalLocation.x = 0;
      specialCase = "";
      finished = false;  // Reset for the next move
      thePiece = "";
      currentState = WAITING;  // Start over
      break;
  }
}


  
bool homing(){
  if (digitalRead(LIM_SWITCH_Y1) && !y_flag){
    stepper2.setSpeed(-HOMING_SPEED);
    stepper3.setSpeed(-HOMING_SPEED);
    stepper2.runSpeed();
    stepper3.runSpeed();
  }
  else if (!digitalRead(LIM_SWITCH_Y1) && !y_flag){
    stepper2.setCurrentPosition(0);
    stepper3.setCurrentPosition(0);
    y_flag = true;
    return true;
  }

  if (digitalRead(LIM_SWITCH_X) && !x_flag){
    stepper1.setSpeed(-HOMING_SPEED);
    stepper1.runSpeed();
  }
  else if (!digitalRead(LIM_SWITCH_X) && !x_flag){
    stepper1.setCurrentPosition(0);
    x_flag = true;
    return true;
  }
}
