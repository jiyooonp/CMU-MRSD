%% Jiyoon Park (jiyoonp@andrew.cmu.edu) MEC Problem 1
%% Question 1
%% part a

A = [0 1 0; 0 0 1; 1 5 7;];
B = [1;0;0;];
eig(A)


%% part b 

C = [B A*B A*A*B];
rank(C)

%% part c

X0 = [0; 1; 0;];
A = [0 1 0; 0 0 1; 1 5 7;];
C = [0 1 3];
y=[];
t = linspace(0,2);

for k=t
    At = A.*k;
    E = expm(At);
    y(end+1) = C*E*X0;
end

figure
grid on 
plot(t, y)
xlabel('t') 
ylabel('y(t)') 

%% part d
A = [0 1 0; 0 0 1; 1 5 7;];
B = [1;0;0;];
p = [-1+j -1-j -2];
K = place(A,B,p)


%% part e 
A = [0 1 0; 0 0 1; 1 5 7;];
B = [1;0;0;];
p = [-1+1j -1-1j -2];
K = place(A,B,p);

C = [0 1 3];
x0 = [0;1;0];

BK = K.*B;
ABK = A-BK;

t = linspace(0,10);

y = [];
for a = t
    y(end+1) = C*forced(ABK, x0, a);
end

figure
grid on 
plot(t, y)
xlabel('t') 
ylabel('y(t)') 

%% Question 2

%% part c
A = [0 0 1 0;
     0 0 0 1;
     0 1 -3 0;
     0 2 -3 0;]

eA = eig(A)

%% part d
figure
grid on 

A = [0 0 1 0;
     0 0 0 1;
     0 1 -3 0;
     0 2 -3 0;];
B = [0;0;1;1;];
Q = [1 0 0 0; 0 5 0 0; 0 0 1 0; 0 0 0 5;];
R = 10;
K = lqr(A, B, Q, R);

time = 0:0.01:30;
x0 = [0; 0.1; 0; 0;];
[t1, y1] = ode45(@(t, x) func1(t, x, K), time, x0);
subplot(4,1,1);
plot(t1, y1)

x0 = [0; 0.5; 0; 0;];
[t1, y1] = ode45(@(t, x) func1(t, x, K), time, x0);
subplot(4,1,2);
plot(t1, y1)

x0 = [0; 1.0886; 0; 0;];
[t1, y1] = ode45(@(t, x) func1(t, x, K), time, x0);
subplot(4,1,3);
plot(t1, y1)

x0 = [0; 1.1; 0; 0;];
[t1, y1] = ode45(@(t, x) func1(t, x, K), time, x0);
subplot(4,1,4);
plot(t1, y1)


%% part e
figure
grid on 
A = [0 0 1 0;
     0 0 0 1;
     0 1 -3 0;
     0 2 -3 0;];
B = [0;0;1;1;];
Q = [1 0 0 0; 0 5 0 0; 0 0 1 0; 0 0 0 5;];
R = 10;
K = lqr(A, B, Q, R);

time = 0:0.01:30;
x0 = [0; 0.1; 0; 0;];
[t1, y1] = ode45(@(t, x) func2(t, x, K), time, x0);
subplot(4,1,1);
plot(t1, y1)

x0 = [0; 0.5; 0; 0;];
[t1, y1] = ode45(@(t, x) func2(t, x, K), time, x0);
subplot(4,1,2);
plot(t1, y1)

x0 = [0; 1.0886; 0; 0;];
[t1, y1] = ode45(@(t, x) func2(t, x, K), time, x0);
subplot(4,1,3);
plot(t1, y1)

x0 = [0; 1.1; 0; 0;];
[t1, y1] = ode45(@(t, x) func2(t, x, K), time, x0);
subplot(4,1,4);
plot(t1, y1)

%% part g
figure
grid on 
A = [0 0 1 0;
     0 0 0 1;
     0 1 -3 0;
     0 2 -3 0;];
B = [0;0;1;1;];
Q = [1 0 0 0; 
     0 5 0 0; 
     0 0 1 0; 
     0 0 0 5;];
R = 10;
K = lqr(A, B, Q, R);
C = [39.3700787 0 0 0];
x0=[0;0;0;0];

T = 0.01;
time = 0:0.01:200;
y = 20*square(2*pi*T*time);

E = -inv(C*inv(A-B*K)*B);


[t1, y1] = ode45(@(t, x) func3(t, x, y, E, K), time, x0);

plot(t1, y1)

figure
plot(time, y)
hold on
grid on
plot(t1(:,1), C*y1')

%% part h
figure
grid on 
A = [0 0 1 0;
     0 0 0 1;
     0 1 -3 0;
     0 2 -3 0;];
B = [0;0;1;1;];

T = 0.01;
time = 0:0.01:200;
y = 20*square(2*pi*T*time);
C = [39.3700787 0 0 0];
x0=[0;0;0;0];


Q = [1 0 0 0; 
     0 5 0 0; 
     0 0 1 0; 
     0 0 0 5;];
R = 10;
K = lqr(A, B, Q, R);
E = -inv(C*inv(A-B*K)*B);

[t1, y1] = ode45(@(t, x) func3(t, x, y, E, K), time, x0);

Q1 =[1 0 0 0; 
     0 5 0 0; 
     0 0 1 0; 
     0 0 0 5;];
R1 = 2;
K1 = lqr(A, B, Q1, R1);
E1 = -inv(C*inv(A-B*K1)*B);

[t2, y2] = ode45(@(t, x) func3(t, x, y, E1, K1), time, x0);

Q1 =[20 0 0 0; 
     0 5 0 0; 
     0 0 20 0; 
     0 0 0 5;];
R2 = 2;
K2 = lqr(A, B, Q1, R2);
E2 = -inv(C*inv(A-B*K2)*B);

[t3, y3] = ode45(@(t, x) func3(t, x, y, E2, K2), time, x0);

plot(time, y)
hold on
grid on
subplot(3,1,1);
plot(time, y)
hold on
plot(t1(:,1), C*y1')
title('R = 10 | Q = 1 5 1 5')
subplot(3,1,2);
plot(time, y)
hold on
plot(t2(:,1), C*y2')
title('R = 2 | Q = 1 5 1 5')
subplot(3,1,3);
plot(time, y)
hold on
plot(t3(:,1), C*y3')
title('R = 2 | Q = 20 5 20 5')


%% functions

function Y = forced(A, x0, t)
    At = A*t;
    E = expm(At);
    Y = E*x0;
end

function out1 = func1(t, x, K)
    A = [0 0 1 0;
         0 0 0 1;
         0 1 -3 0;
         0 2 -3 0;];
    B = [0;0;1;1;];
    Q = [1 0 0 0; 0 5 0 0; 0 0 1 0; 0 0 0 5;];
    R = 10;
    K = lqr(A, B, Q, R);
    U = -K*x;
    out1 = A*x+B*U;
end

function out2 = func2(t, x, K)
    
    U = -K*x;
    out2 = [x(3);
            x(4);
            (U-x(4)^2*sin(x(2))-3*x(3)+cos(x(2))*sin(x(2)))/(2-cos(x(2))^2);
            (U*cos(x(2))-x(4)^2*cos(x(2))*sin(x(2))-3*x(3)*cos(x(2))+2*sin(x(2)))/(2-cos(x(2))^2)];    

end

function out3 = func3(t, x, y, E, K)
    y = 20*square(2*pi*0.01*t);
    V = E*y;
    U = V-K*x;
    out3 = [x(3);
            x(4);
            (U-x(4)^2*sin(x(2))-3*x(3)+cos(x(2))*sin(x(2)))/(2-cos(x(2))^2);
            (U*cos(x(2))-x(4)^2*cos(x(2))*sin(x(2))-3*x(3)*cos(x(2))+2*sin(x(2)))/(2-cos(x(2))^2)];    
end
