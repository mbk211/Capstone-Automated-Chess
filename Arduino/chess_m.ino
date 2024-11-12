
String getInput() {
  String buffer;
  
  while (true) {
    buffer = Serial.readStringUntil('\n');
    buffer.trim();
    if (buffer.length() == 2) {  // Valid input length check
      return buffer;
    } else {
      Serial.println("Invalid command received.");
    }
  }
}
char specialMove() {
  
  // K for Knight
  // X for Capture
  // Y for Castling
  // Z for En Passant
  // P for Promotion

  String buffer;
  
  while (true) {
    // Read the incoming serial data until a newline character is found
    buffer = Serial.readStringUntil('\n');
    buffer.trim(); // Remove any leading and trailing whitespace

    if (buffer.length() == 1) {
      char command = buffer.charAt(0);
      command = toupper(command); 
      if (command == 'K' || command == 'X' || command == 'Y' || command == 'Z' || command == 'P' || command == 'O') {
        return command; 
      }
      else{
        return "";
      }
    }
  }
}


void getLocations(){


  if (origin == "" && Serial.available() > 0){
    origin = getInput();
    initialLocation = getCoordinates(origin);
  }
  if (origin != "" && destination == "" && Serial.available() > 0) {
    destination = getInput();
    finalLocation= getCoordinates(destination);
    specialCase = specialMove();

    // Serial.println(origin);
    // Serial.println(destination);
    // Serial.println(specialCase);
}
  }


Coordinates getCoordinates(String buffer) {
  Coordinates coordinate;
  switch (buffer.charAt(0)) {
    case 'a': coordinate.x = CENTER_SQUARE_1; break;
    case 'b': coordinate.x = CENTER_SQUARE_2; break;
    case 'c': coordinate.x = CENTER_SQUARE_3; break;
    case 'd': coordinate.x = CENTER_SQUARE_4; break;
    case 'e': coordinate.x = CENTER_SQUARE_5; break;
    case 'f': coordinate.x = CENTER_SQUARE_6; break;
    case 'g': coordinate.x = CENTER_SQUARE_7; break;
    case 'h': coordinate.x = CENTER_SQUARE_8; break;
    default: coordinate.x = 0; break;  // Invalid input
  }
  switch (buffer.charAt(1)) {
    case '1': coordinate.y = CENTER_SQUARE_1; break;
    case '2': coordinate.y = CENTER_SQUARE_2; break;
    case '3': coordinate.y = CENTER_SQUARE_3; break;
    case '4': coordinate.y = CENTER_SQUARE_4; break;
    case '5': coordinate.y = CENTER_SQUARE_5; break;
    case '6': coordinate.y = CENTER_SQUARE_6; break;
    case '7': coordinate.y = CENTER_SQUARE_7; break;
    case '8': coordinate.y = CENTER_SQUARE_8; break;
    default: coordinate.y = 0; break;  // Invalid input
  }
  return coordinate;
}

void printPath(Coordinates first, Coordinates last){
  Serial.print("\n(");
  Serial.print(first.x);
  Serial.print(",");
  Serial.print(first.y);
  Serial.print(")");
  Serial.print(" --------> ");
  Serial.print("(");
  Serial.print(last.x);
  Serial.print(",");
  Serial.print(last.y);
  Serial.println(")\n");
}




inline void magnetOn(){
  digitalWrite(MY_RELAY, HIGH);
}
inline void magnetOff(){
  digitalWrite(MY_RELAY, LOW);
}

// void homing(){
//   while(true){
//     stepper1.moveTo(0);
//     stepper2.moveTo(0);
//     stepper3.moveTo(0);
//     stepper1.run();
//     stepper2.run();
//     stepper3.run();
//     if (stepper1.currentPosition() == 0 && stepper2.currentPosition() == 0 && stepper3.currentPosition() ==0){
//       break;
//     }
//   }
// } 

bool moveMotors(Coordinates pos){// NON-BLOCKING
  while(true){
    stepper1.moveTo(pos.x);
    stepper2.moveTo(pos.y);
    stepper3.moveTo(pos.y);
    stepper1.run();
    stepper2.run();
    stepper3.run();
    if (pos.x == stepper1.currentPosition() && pos.y == stepper2.currentPosition() && pos.y == stepper3.currentPosition()){
      return true;
    }
  }


}

bool movePiece(){
  
  if (moveMotors(initialLocation)){
    // delay(300);
    magnetOn();
    // PORTD |= (1 << MY_RELAY);
    if (moveMotors(finalLocation)){
      delay(300);
      magnetOff();
      // PORTD &= ~(1 << MY_RELAY);
      delay(300);
      return true;
    }
  }
}


void clearLocations(){
  initialLocation.x = 0;
  initialLocation.y = 0;
  finalLocation.x = 0;
  finalLocation.y = 0;
}

String checkSquare(Coordinates location){
  for (int i = 0; i < 16; i++){
    if (location.x == piece[i].coordinate.x && piece[i].coordinate.y == location.y) {
      return piece[i].name;  // Return the name of the piece at the coordinates
    }
  }
  return "";
}

void pieceInit(){

  //left to right

  piece[0] = {"Pawn1", {CENTER_SQUARE_1, CENTER_SQUARE_2}};
  piece[1] = {"Pawn2", {CENTER_SQUARE_2, CENTER_SQUARE_2}};
  piece[2] = {"Pawn3", {CENTER_SQUARE_3, CENTER_SQUARE_2}};
  piece[3] = {"Pawn4", {CENTER_SQUARE_4, CENTER_SQUARE_2}};
  piece[4] = {"Pawn5", {CENTER_SQUARE_5, CENTER_SQUARE_2}};
  piece[5] = {"Pawn6", {CENTER_SQUARE_6, CENTER_SQUARE_2}};
  piece[6] = {"Pawn7", {CENTER_SQUARE_7, CENTER_SQUARE_2}};
  piece[7] = {"Pawn8", {CENTER_SQUARE_8, CENTER_SQUARE_2}};
  
  //1 is left
  //2 is right

  // (Letter, Number)
  piece[8] = {"Knight1", {CENTER_SQUARE_2, CENTER_SQUARE_1}};  // Knight at (1, 0)
  piece[9] = {"Knight2", {CENTER_SQUARE_7, CENTER_SQUARE_1}};  // Knight at (1, 0)

  piece[10] = {"Bishop1", {CENTER_SQUARE_3, CENTER_SQUARE_1}};  // Bishop at (2, 0)
  piece[11] = {"Bishop2", {CENTER_SQUARE_6, CENTER_SQUARE_1}};  // Bishop at (2, 0)

  piece[12] = {"Rook1", {CENTER_SQUARE_1, CENTER_SQUARE_1}};    // Rook at (0, 0)
  piece[13] = {"Rook2", {CENTER_SQUARE_8, CENTER_SQUARE_1}};    // Rook at (0, 0)

  piece[14] = {"Queen", {CENTER_SQUARE_5, CENTER_SQUARE_1}};    // Queen at (0, 0)

  piece[15] = {"King", {CENTER_SQUARE_4, CENTER_SQUARE_1}};    // Rook at (0, 0)
}

void updatePiece(){
  for (int i = 0; i < 16; i++) {
    if (piece[i].name == thePiece) {
      piece[i].coordinate.x = finalLocation.x;
      piece[i].coordinate.y = finalLocation.y;
    }
  }
  Serial.println("Board Updated!");
}


bool printBoard(){

  String board[8][8];
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      board[i][j] = "  ";  // Empty space
    }
  }
  for (int i = 0; i < 16; i++) {
    int x = mapCoordinate(piece[i].coordinate.x);
    int y = mapCoordinate(piece[i].coordinate.y);
    board[x][y] = mapName(piece[i].name);  // Place the piece at the correct position
  }
  Serial.println("  -------------------------");
  for (int i = 0; i < 8; i++) {
    Serial.print("|");
    for (int j = 0; j < 8; j++) {
      Serial.print(board[j][i]);
      Serial.print("|");
    }
    Serial.println();  // New line after each row
    Serial.println("  -------------------------");
  }
  return true;
}

int mapCoordinate(short coord) {
  switch (coord) {
    case 194:
      return 0;  // Map to board index 0
    case 480:
      return 1;  // Map to board index 1
    case 766:
      return 2;  // Map to board index 2
    case 1052:
      return 3;  // Map to board index 3
    case 1338:
      return 4;  // Map to board index 4
    case 1623:
      return 5;  // Map to board index 5
    case 1910:
      return 6;  // Map to board index 6
    case 2196:
      return 7;  // Map to board index 7
    default:
      return -1;  // Invalid coordinate
  }
}

String mapName(String name) {
  if (name == "Pawn1") {
    return "p1";  
  } else if (name == "Pawn2") {
    return "p2";  
  } else if (name == "Pawn3") {
    return "p3";  
  } else if (name == "Pawn4") {
    return "p4";  
  } else if (name == "Pawn5") {
    return "p5";  
  } else if (name == "Pawn6") {
    return "p6";  
  } else if (name == "Pawn7") {
    return "p7";  
  } else if (name == "Pawn8") {
    return "p8";  
  } 
  
  else if (name == "Knight1") {
    return "K1";  
  } else if (name == "Knight2") {
    return "K2";  
  } else if (name == "Bishop1") {
    return "B1";  
  } else if (name == "Bishop2") {
    return "B2";  
  } else if (name == "Rook1") {
    return "R1";  
  } else if (name == "Rook2") {
    return "R2";  
  } else if (name == "King") {
    return "K ";  
  } else if (name == "Queen") {
    return "Q ";  
  } else {
    return "";  // Invalid name
  }
}


bool knightException(Coordinates destination, Coordinates origin){
  Coordinates diff;
  int direction; 

  moveMotors(origin);
  magnetOn();
  delay(300);

  diff.x = abs(destination.x - origin.x);
  diff.y = abs(destination.y - origin.y);
  
  
  if (diff.x < diff.y){
    if (destination.x < origin.x){
      moveMotors({destination.x + HALF_SQUARE, origin.y});
      moveMotors({destination.x + HALF_SQUARE, destination.y});
    }
    else{
      moveMotors({destination.x - HALF_SQUARE, origin.y});
      moveMotors({destination.x - HALF_SQUARE, destination.y});
    }
    moveMotors({destination.x, destination.y});
  }
  else if (diff.y < diff.x){
    if (destination.y < origin.y){
      moveMotors({origin.x, destination.y + HALF_SQUARE});
      moveMotors({destination.x, destination.y + HALF_SQUARE});
    }
    else{
      moveMotors({origin.x, destination.y - HALF_SQUARE});
      moveMotors({destination.x, destination.y - HALF_SQUARE});
    }
    moveMotors({destination.x, destination.y});
  }
  else{
    magnetOff();
    return false;
  }

  // moveMotors(pos.s);
  magnetOff();
  return true;
}

bool captureException(Coordinates destination, Coordinates origin){
  moveMotors(destination);
  magnetOn();
  delay(300);

  Coordinates grave = {MAX_Y_POSITION,destination.y};
  moveMotors(grave);
  delay(300);
  magnetOff();

  return movePiece();

}
