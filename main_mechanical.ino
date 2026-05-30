/*
  ╔══════════════════════════════════════════════════════════════╗
  ║              TURRET CONTROLLER — ARDUINO                    ║
  ║                                                             ║
  ║  Fixes applied:                                             ║
  ║    1. Pan direction: PAN_LEFT_SPEED_US / PAN_RIGHT_SPEED_US  ║
  ║       were swapped — cmdPanLeft() was spinning RIGHT.        ║
  ║    2. panMoveSteps() step→direction logic corrected.         ║
  ║    3. cmdTiltSetUS() now uses smoothMove() to avoid jitter.  ║
  ║    4. Serial.flush() + buffer clear added in setup() to      ║
  ║       discard garbage bytes during RPi boot delay.           ║
  ║                                                             ║
  ║  Commands accepted (newline-terminated):                    ║
  ║    A      pan LEFT  one step                                 ║
  ║    D      pan RIGHT one step                                 ║
  ║    T####  set tilt absolute µs  (e.g. T1200)                 ║
  ║    F      fire                                               ║
  ║    R      reset / home                                       ║
  ║    W / S  tilt up / down (manual, one step)                  ║
  ║    M      toggle drive motors                                ║
  ║    H      print help                                         ║
  ║                                                             ║
  ║  IMPORTANT: Servos MUST use an external 5 V supply.          ║
  ╚══════════════════════════════════════════════════════════════╝
*/

#include <Servo.h>

// ═══════════════════════════════════════════════════════════════
//  PIN CONFIGURATION
// ═══════════════════════════════════════════════════════════════

const int TILT_PIN     = 9;
const int PAN_PIN      = 10;   // continuous rotation servo
const int SHOOT_PIN    = 11;

const int MOTOR_A_PIN1 = 2;
const int MOTOR_A_PIN2 = 3;
const int MOTOR_A_EN   = 5;

const int MOTOR_B_PIN1 = 4;
const int MOTOR_B_PIN2 = 7;
const int MOTOR_B_EN   = 6;

// ═══════════════════════════════════════════════════════════════
//  SERVO CONFIGURATION
// ═══════════════════════════════════════════════════════════════

// Tilt servo (standard 0°–70°)
const int TILT_MIN_US  = 600;
const int TILT_MAX_US  = 1700;
const int TILT_HOME_US = TILT_MIN_US;

// Pan servo — continuous rotation
//   Calibrate these two values for YOUR servo if it doesn't sit still at 1500:
//   increase PAN_STOP_US by ~10 if it drifts one way, decrease if the other.
const int PAN_STOP_US  = 1500;

// FIX #1: original code had LEFT=1600 and RIGHT=1400, which physically
// rotated the turret in the wrong directions.
// Standard continuous servo: >1500 → clockwise, <1500 → counter-clockwise.
// Adjust the comment below to match YOUR servo's wiring if it's still reversed:
//   If pan still goes the wrong way, swap the two values below.
const int PAN_CW_US    = 1600;   // clockwise        → turret turns RIGHT
const int PAN_CCW_US   = 1400;   // counter-clockwise → turret turns LEFT

const int PAN_STEP_DURATION_MS = 150;   // ms the pan motor runs per step
const int PAN_STEPS_MAX        = 36;    // ±36 steps from home (doubled to prevent left/right hard stop during tracking)
const int PAN_HOME_STEP        = 0;

// Shoot servo
const int SHOOT_REST_US = 1200;
const int SHOOT_FIRE_US = 2000;

// Motion constants
const int SERVO_STEP_US   = 80;    // µs per manual tilt key-press
const int SMOOTH_DELAY_MS = 15;    // ms between smoothMove() steps
const int SHOOT_HOLD_MS   = 75;
const int SHOOT_SETTLE_MS = 300;
const int MOTOR_SPEED     = 200;

// ═══════════════════════════════════════════════════════════════
//  STATE
// ═══════════════════════════════════════════════════════════════

Servo tiltServo;
Servo panServo;
Servo shootServo;

int  tiltUS   = TILT_HOME_US;
int  panStep  = PAN_HOME_STEP;
bool motorsOn = false;
bool isFiring = false;

// ═══════════════════════════════════════════════════════════════
//  FORWARD DECLARATIONS
// ═══════════════════════════════════════════════════════════════

void printBanner();
void printHelp();
void printState();
void smoothMove(Servo &srv, int currentUS, int targetUS);
int  usToDeg(int us, int minUS, int maxUS, int minDeg, int maxDeg);

void panRotateStep(int speedUS);
void panMoveSteps(int steps);

void cmdTiltUp();
void cmdTiltDown();
void cmdTiltSetUS(int targetUS);
void cmdPanLeft();
void cmdPanRight();
void cmdFire();
void cmdToggleMotors();
void cmdResetHome();

// ═══════════════════════════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════════════════════════

void setup() {
  Serial.begin(115200);

  // Drive motor pins
  pinMode(MOTOR_A_PIN1, OUTPUT); pinMode(MOTOR_A_PIN2, OUTPUT);
  pinMode(MOTOR_A_EN,   OUTPUT);
  pinMode(MOTOR_B_PIN1, OUTPUT); pinMode(MOTOR_B_PIN2, OUTPUT);
  pinMode(MOTOR_B_EN,   OUTPUT);

  digitalWrite(MOTOR_A_PIN1, HIGH); digitalWrite(MOTOR_A_PIN2, LOW);
  analogWrite(MOTOR_A_EN, 0);
  digitalWrite(MOTOR_B_PIN1, HIGH); digitalWrite(MOTOR_B_PIN2, LOW);
  analogWrite(MOTOR_B_EN, 0);

  // Attach servos
  tiltServo.attach(TILT_PIN, TILT_MIN_US, TILT_MAX_US);
  tiltServo.writeMicroseconds(TILT_HOME_US);

  panServo.attach(PAN_PIN);
  panServo.writeMicroseconds(PAN_STOP_US);

  shootServo.attach(SHOOT_PIN, 500, 2500);
  shootServo.writeMicroseconds(SHOOT_REST_US);

  delay(1000);

  // FIX #4: flush any garbage that arrived while RPi was booting
  Serial.flush();
  while (Serial.available() > 0) Serial.read();

  printBanner();
  printHelp();
  printState();
}

// ═══════════════════════════════════════════════════════════════
//  MAIN LOOP
// ═══════════════════════════════════════════════════════════════

void loop() {
  if (Serial.available() <= 0) return;

  char c0 = (char)Serial.peek();

  // ── "T####" absolute tilt from Raspberry Pi ──────────────────
  if (c0 == 'T' || c0 == 't') {
    Serial.read();  // consume 'T'/'t'

    // Legacy: bare 't' + newline → auto-test (kept for compatibility)
    if (Serial.peek() == '\n' || Serial.peek() == '\r' ||
        Serial.peek() == -1) {
      while (Serial.available() > 0) Serial.read();
      cmdResetHome();
      return;
    }

    int targetUS = Serial.parseInt();
    while (Serial.available() > 0) Serial.read();
    cmdTiltSetUS(targetUS);
    return;
  }

  // ── Single-character commands ─────────────────────────────────
  char cmd = (char)Serial.read();
  while (Serial.available() > 0) Serial.read();  // discard rest of line

  // Normalize to lowercase
  if (cmd >= 'A' && cmd <= 'Z') cmd += 32;

  switch (cmd) {
    case 'w': cmdTiltUp();       break;
    case 's': cmdTiltDown();     break;
    case 'a': cmdPanLeft();      break;
    case 'd': cmdPanRight();     break;
    case 'f': cmdFire();         break;
    case 'm': cmdToggleMotors(); break;
    case 'r': cmdResetHome();    break;
    case 'h': printHelp();       break;
    case '\n':
    case '\r':
      break;
    default:
      Serial.print(F("  Unknown command: '"));
      Serial.print(cmd);
      Serial.println(F("'  — press H for help"));
      break;
  }
}

// ═══════════════════════════════════════════════════════════════
//  PAN HELPERS
// ═══════════════════════════════════════════════════════════════

void panRotateStep(int speedUS) {
  panServo.writeMicroseconds(speedUS);
  delay(PAN_STEP_DURATION_MS);
  panServo.writeMicroseconds(PAN_STOP_US);
  delay(50);  // settle before next command
}

// FIX #2: corrected direction mapping.
//   steps > 0 → move RIGHT (clockwise)
//   steps < 0 → move LEFT  (counter-clockwise)
void panMoveSteps(int steps) {
  int speedUS = (steps > 0) ? PAN_CW_US : PAN_CCW_US;
  int count   = abs(steps);
  for (int i = 0; i < count; i++) panRotateStep(speedUS);
}

// ═══════════════════════════════════════════════════════════════
//  COMMANDS
// ═══════════════════════════════════════════════════════════════

void cmdTiltUp() {
  tiltUS = min(TILT_MAX_US, tiltUS + SERVO_STEP_US);
  tiltServo.writeMicroseconds(tiltUS);
}

void cmdTiltDown() {
  tiltUS = max(TILT_MIN_US, tiltUS - SERVO_STEP_US);
  tiltServo.writeMicroseconds(tiltUS);
}

// FIX #3: was a direct write; now uses smoothMove() to prevent jitter
void cmdTiltSetUS(int targetUS) {
  if (targetUS < TILT_MIN_US) targetUS = TILT_MIN_US;
  if (targetUS > TILT_MAX_US) targetUS = TILT_MAX_US;
  smoothMove(tiltServo, tiltUS, targetUS);
  tiltUS = targetUS;
}

void cmdPanLeft() {
  if (panStep <= -PAN_STEPS_MAX) return;
  // FIX #1: pan LEFT = counter-clockwise = PAN_CCW_US
  panRotateStep(PAN_CCW_US);
  panStep--;
}

void cmdPanRight() {
  if (panStep >= PAN_STEPS_MAX) return;
  // FIX #1: pan RIGHT = clockwise = PAN_CW_US
  panRotateStep(PAN_CW_US);
  panStep++;
}

void cmdFire() {
  if (isFiring) return;
  isFiring = true;
  shootServo.writeMicroseconds(SHOOT_FIRE_US);
  delay(SHOOT_HOLD_MS);
  shootServo.writeMicroseconds(SHOOT_REST_US);
  delay(SHOOT_SETTLE_MS);
  isFiring = false;
}

void cmdToggleMotors() {
  if (motorsOn) {
    analogWrite(MOTOR_A_EN, 0);
    analogWrite(MOTOR_B_EN, 0);
    motorsOn = false;
  } else {
    analogWrite(MOTOR_A_EN, MOTOR_SPEED);
    analogWrite(MOTOR_B_EN, MOTOR_SPEED);
    motorsOn = true;
  }
}

void cmdResetHome() {
  smoothMove(tiltServo, tiltUS, TILT_HOME_US);
  tiltUS = TILT_HOME_US;

  if (panStep != PAN_HOME_STEP) {
    // FIX #2: stepsToHome sign drives the correct direction
    int stepsToHome = PAN_HOME_STEP - panStep;
    panMoveSteps(stepsToHome);
    panStep = PAN_HOME_STEP;
  }

  shootServo.writeMicroseconds(SHOOT_REST_US);

  if (motorsOn) {
    analogWrite(MOTOR_A_EN, 0);
    analogWrite(MOTOR_B_EN, 0);
    motorsOn = false;
  }
}

// ═══════════════════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════════════════

void smoothMove(Servo &srv, int currentUS, int targetUS) {
  if (currentUS == targetUS) return;
  int step = (targetUS > currentUS) ? SERVO_STEP_US : -SERVO_STEP_US;
  int pos  = currentUS;
  while (abs(pos - targetUS) > abs(step)) {
    pos += step;
    srv.writeMicroseconds(pos);
    delay(SMOOTH_DELAY_MS);
  }
  srv.writeMicroseconds(targetUS);
}

int usToDeg(int us, int minUS, int maxUS, int minDeg, int maxDeg) {
  return map(us, minUS, maxUS, minDeg, maxDeg);
}

void printState() {
  int tiltDeg = usToDeg(tiltUS, TILT_MIN_US, TILT_MAX_US, 0, 70);
  int panDeg  = (int)((long)panStep * 90 / PAN_STEPS_MAX);
  Serial.print(F("Tilt: "));  Serial.print(tiltDeg); Serial.print(F("deg  "));
  Serial.print(F("Pan: "));   Serial.print(panDeg);  Serial.println(F("deg"));
}

void printBanner() {
  Serial.println(F("\n+=========================================+"));
  Serial.println(F("|      TURRET CONTROLLER — READY          |"));
  Serial.println(F("+=========================================+\n"));
}

void printHelp() {
  Serial.println(F("Commands (case-insensitive):"));
  Serial.println(F("  A / D      pan left / right (one step)"));
  Serial.println(F("  W / S      tilt up / down (one step)"));
  Serial.println(F("  T####      set tilt absolute µs (e.g. T1200)"));
  Serial.println(F("  F          fire"));
  Serial.println(F("  R          reset to home"));
  Serial.println(F("  M          toggle drive motors"));
  Serial.println(F("  H          this help"));
}
