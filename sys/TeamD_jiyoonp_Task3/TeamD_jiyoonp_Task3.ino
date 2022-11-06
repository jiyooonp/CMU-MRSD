// Jiyoon Park (jiyoonp)
//all the parameters
int count = 0;

const byte interruptPin = 2;
int state = 0;

int sensorPin = A0;   
double sensorValue = 0;  

const int buttonPin0 = 2;    
int buttonState0 = 0; 

const int buttonPin1 = 3;     
int buttonState1 = 0; 
int buttonState1_prev = 0;

const int ledRPin = 9;     
const int ledGPin = 10;    
const int ledBPin = 11;    
int ledOn = 0;

String input;
char color;
int value;

void setup() {
  
  pinMode(ledRPin, OUTPUT);
  pinMode(ledGPin, OUTPUT);
  pinMode(ledBPin, OUTPUT);
  pinMode(buttonPin0, INPUT_PULLUP);

  pinMode(buttonPin1, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(buttonPin0), my_interrupt_handler0, FALLING);
  attachInterrupt(digitalPinToInterrupt(buttonPin1), my_interrupt_handler1, CHANGE);

  Serial.begin(9600);
}

void loop() {
  count +=1;
  // if (count%100000==0){
  //   Serial.println(state);
  // }
  if (state == 0){
    // if (count%100000==0){
    //     Serial.print("ledOn");
    //   }

    if (ledOn==0){
      digitalWrite(ledBPin, HIGH);
    }
    else{
      digitalWrite(ledBPin, LOW);
    }

    digitalWrite(ledRPin, HIGH);
    digitalWrite(ledGPin, HIGH);
    
    buttonState1 = digitalRead(buttonPin1); 
    if (buttonState1 == LOW){
      buttonState1_prev = 0;    
    }
    
  }
  else if(state == 1){
    digitalWrite(ledRPin, HIGH);
    digitalWrite(ledGPin, HIGH);
    sensorValue = analogRead(sensorPin)*255.0/1023.0;
    // Serial.println(sensorValue);
    analogWrite(ledBPin, sensorValue);
  }
  else{
    if(Serial.available()){

      input = Serial.readStringUntil('\n');
      Serial.print("You typed: " );
      Serial.println(input);
      
      color = input.charAt(0);
      value = input.substring(1).toDouble()/1;
      
      if ((value>=0)&(value<=255)){
        switch(color){
          case 'r':
            analogWrite(ledRPin, 255 - value);
            break;
          case 'g':
            analogWrite(ledGPin, 255 - value);
            break;
          case 'b':
            analogWrite(ledBPin, 255 - value);
            break;
          default:
            break;
        }
      }
    }
  }

}

void my_interrupt_handler0()
{
  static unsigned long last_interrupt_time = 0;
  unsigned long interrupt_time = millis();
  
  // If interrupts come faster than 700ms, assume it's a bounce and ignore
  if (interrupt_time - last_interrupt_time > 200)
  {
    if (state ==2){
      state = 0;
    }
    else{
      state +=1;

    }
    
  }
  else{
    Serial.println("bounced");
  }
  last_interrupt_time = interrupt_time;
  Serial.print("Current State:");
  Serial.println(state);
  digitalWrite(ledRPin, HIGH);
  digitalWrite(ledGPin, HIGH);
  digitalWrite(ledBPin, HIGH);
}
void my_interrupt_handler1()
{
  static unsigned long last_interrupt_time = 0;
  unsigned long interrupt_time = millis();
  
  if (interrupt_time - last_interrupt_time > 50)
  {
    buttonState1 = digitalRead(buttonPin1); 
    if (buttonState1 == LOW){
      buttonState1_prev = 0;    
    }

    if (buttonState1 != buttonState1_prev){
      Serial.println("Changed Values");
      buttonState1_prev = buttonState1;
      if (ledOn == 0){
        ledOn = 1;
        buttonState1_prev = buttonState1;
      }
      else{
        ledOn = 0;
      }
    }
  }
  else{
    Serial.println("bounced");
  }
  last_interrupt_time = interrupt_time;
}