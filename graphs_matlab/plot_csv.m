%GA

% Elites changing

% dim 5
elitesGA5_fig = figure;

plot(originalga(1,:), originalga(2,:));
hold on;
plot(elitesga(1,:),elitesga(2,:));
hold on;
plot(elitesga(3,:),elitesga(4,:));
hold off;

title('Fitness = f(Elites) --> dimension: 5')
xlabel('Generations') 
ylabel('Fitness') 
legend('Elites: 4','Elites: 8','Elites: 16')

% dim 10
elitesGA10_fig = figure;

plot(originalga(3,:), originalga(4,:));
hold on;
plot(elitesga(5,:),elitesga(6,:));
hold on;
plot(elitesga(7,:),elitesga(8,:));
hold off;

title('Fitness = f(Elites) --> dimension: 10')
xlabel('Generations') 
ylabel('Fitness') 
legend('Elites: 4','Elites: 8','Elites: 16')



% Mutation changing

% dim 5
mutationGA5_fig = figure;

plot(originalga(1,:), originalga(2,:));
hold on;
plot(mutationga(1,:),mutationga(2,:));
hold on;
plot(mutationga(3,:),mutationga(4,:));
hold off;

title('Fitness = f(Mutation) --> dimension: 5')
xlabel('Generations') 
ylabel('Fitness') 
legend('Mutation: 5%','Mutation: 10%','Mutation: 20%')

% dim 10
mutationGA10_fig = figure;

plot(originalga(3,:), originalga(4,:));
hold on;
plot(mutationga(5,:),mutationga(6,:));
hold on;
plot(mutationga(7,:),mutationga(8,:));
hold off;

title('Fitness = f(Mutation) --> dimension: 10')
xlabel('Generations') 
ylabel('Fitness') 
legend('Mutation: 5%','Mutation: 10%','Mutation: 20%')



% Max absolute mutation changing

% dim 5
maxabsMutGA5_fig = figure;

plot(originalga(1,:), originalga(2,:));
hold on;
plot(maxabsga(1,:),maxabsga(2,:));
hold on;
plot(maxabsga(3,:),maxabsga(4,:));
hold off;

title('Fitness = f(MaxAbsMut) --> dimension: 5')
xlabel('Generations') 
ylabel('Fitness') 
legend('MaxAbsMut: 0.1','MaxAbsMut: 0.4','MaxAbsMut: 0.8')

% dim 10
maxabsMutGA10_fig = figure;

plot(originalga(3,:), originalga(4,:));
hold on;
plot(maxabsga(5,:),maxabsga(6,:));
hold on;
plot(maxabsga(7,:),maxabsga(8,:));
hold off;

title('Fitness = f(MaxAbsMut) --> dimension: 10')
xlabel('Generations') 
ylabel('Fitness') 
legend('MaxAbsMut: 0.1','MaxAbsMut: 0.4','MaxAbsMut: 0.8')




%************************************************************%


%PSO



% Inertia changing

% dim 5
inertiaPSO5_fig = figure;

plot(originalpso(1,:), originalpso(2,:));
hold on;
plot(inertiapso(1,:),inertiapso(2,:));
hold on;
plot(inertiapso(3,:),inertiapso(4,:));
hold off;

title('Fitness = f(Inertia) --> dimension: 5')
xlabel('Generations') 
ylabel('Fitness') 
legend('Inetria: 0.00','Inertia: 0.37','Inertia: 0.74')

% dim 10
inertiaPSO10_fig = figure;

plot(originalpso(3,:), originalpso(4,:));
hold on;
plot(inertiapso(5,:),inertiapso(6,:));
hold on;
plot(inertiapso(7,:),inertiapso(8,:));
hold off;


title('Fitness = f(Inertia) --> dimension: 10')
xlabel('Generations') 
ylabel('Fitness') 
legend('Inetria: 0.00','Inertia: 0.37','Inertia: 0.74')


% Personal factor changing

% dim 5
PersFactPSO5_fig = figure;

plot(originalpso(1,:), originalpso(2,:));
hold on;
plot(personalfactorpso(1,:),personalfactorpso(2,:));
hold on;
plot(personalfactorpso(3,:),personalfactorpso(4,:));
hold off;

title('Fitness = f(Personal factor) --> dimension: 5')
xlabel('Generations') 
ylabel('Fitness') 
legend('Personal factor: 0.5','Personal factor: 1.0','Personal factor: 1.5')

% dim 10
PersFactPSO10_fig = figure;

plot(originalpso(3,:), originalpso(4,:));
hold on;
plot(personalfactorpso(5,:),personalfactorpso(6,:));
hold on;
plot(personalfactorpso(7,:),personalfactorpso(8,:));
hold off;

title('Fitness = f(Personal factor) --> dimension: 10')
xlabel('Generations') 
ylabel('Fitness') 
legend('Personal factor: 0.5','Personal factor: 1.0','Personal factor: 1.5')


% Social factor changing

% dim 5
SocFactPSO5_fig = figure;

plot(originalpso(1,:), originalpso(2,:));
hold on;
plot(socialfactorpso(1,:),socialfactorpso(2,:));
hold on;
plot(socialfactorpso(3,:),socialfactorpso(4,:));
hold off;

title('Fitness = f(Social factor) --> dimension: 5')
xlabel('Generations') 
ylabel('Fitness') 
legend('Social factor: 0.5','Social factor: 1.0','Social factor: 1.5')

% dim 10
SocFactPSO10_fig = figure;

plot(originalpso(3,:), originalpso(4,:));
hold on;
plot(socialfactorpso(5,:),socialfactorpso(6,:));
hold on;
plot(socialfactorpso(7,:),socialfactorpso(8,:));
hold off;

title('Fitness = f(Social factor) --> dimension: 10')
xlabel('Generations') 
ylabel('Fitness') 
legend('Social factor: 0.5','Social factor: 1.0','Social factor: 1.5')


