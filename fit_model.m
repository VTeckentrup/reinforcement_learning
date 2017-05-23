function L = fit_model(x,D)

% Model Parameters, original units
%alpha = x(1);
%beta = x(2);

% Model Parameters, transformed
alpha = 1/(1+exp(-x(1)));
beta  = exp(x(2)); 

% Parse Data
rew = D(:,2);
choice = D(:,1);
 
%Model Initial Values
L = 1; % initial value sum of squared error
%trials = 1:size(data,1);
 
% This model is looping through every trial. Obviously this isn't
% necessary for this specific model, but it is for more dynamic models that
% change with respect to time such as RL models.

RPE = zeros(length(choice),2);
Q = zeros(length(choice),2);

for t = 1:length(choice)

    RPE(t,choice(t,1)) = rew(t,1) - Q(t,choice(t,1));
    
    if t < length(choice)
        Q(t+1,1) = Q(t,1) + alpha * RPE(t,1);
        Q(t+1,2) = Q(t,2) + alpha * RPE(t,2);
    end
    
    %Calculate loglikelihood
    L = L + log( exp(Q(t,choice(t,1)) .* beta) / (exp(Q(t,1) .* beta) + exp(Q(t,2) .* beta)));
    %L=L*(1/(2+exp(a*D(i)+b)));
end

L=-L;

%Output trial by trial results - saved as obj.trial
%trialout = [ones(t,1)*data(1,1) (1:t)', obssx, predsx(1:t)'];
 
end