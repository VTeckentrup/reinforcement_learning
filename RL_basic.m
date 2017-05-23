
%RL simulation containing a basic Q Learning approach
%contact: nils.kroemer@uni-tuebingen.de

sim_par.alpha = 0.4; %learning rate of the simulated agent
sim_par.beta = 1.5; %inverse temperature of the simulated agent
sim_par.n_trials = 200; %number of trials
sim_par.p_win = .8; %probability of winning for the good option
sim_par.p_lose = .2; %probability of losing for the good option
sim_par.min_flip = 10; %minimum number of trials before another reversal can occur
sim_par.n_part = 100; %number of simulated participants

for i_p = 1:sim_par.n_part

    %initialize variables
    RPE = zeros(sim_par.n_trials,2);
    Q = zeros(sim_par.n_trials,2);
    choice_p = zeros(sim_par.n_trials,2);
    reward = zeros(sim_par.n_trials,1);
    choice = zeros(sim_par.n_trials,1);
    good_opt = zeros(sim_par.n_trials,1);
    count_flip = 1;
    count_rev = 0;

    for i_t = 1:sim_par.n_trials

        %choice
        %1 = OptA, 2 = OptB
        if i_t == 1
            choice(i_t,1) = randi(2,1); %first choice is random
        else
            %[Q_max, best_opt] = max(Q(i_t,:));
            %choice(i_t,1) = best_opt;

            OptA = exp(Q(i_t,1)*sim_par.beta);
            OptB = exp(Q(i_t,2)*sim_par.beta);
            choice_p(i_t,1) = OptA / (OptA + OptB);
            choice_p(i_t,2) = OptB / (OptA + OptB);

            [Q_max, pref_opt] = max(choice_p(i_t,:));

            if rand < choice_p(i_t,1)
               choice(i_t,1) = 1;
            else
               choice(i_t,1) = 2;
            end

    %        choice(i_t,1) = pref_opt;

        end

        if i_t > 10
            count_chA = sum(choice(i_t-9:i_t) == 1);
        else
            count_chA = 5;
        end

        %assign one the options as the good option
        if i_t == 1
            good_opt(i_t,1) = randi(2,1);

        %flip good and bad if accuracy is higher than 7 out of 10
        elseif count_flip > sim_par.min_flip && count_chA>7 && good_opt(i_t-1,1) == 1
            good_opt(i_t,1) = 2;
            count_flip = 1; %resets flip counter
            count_rev = count_rev + 1;
        elseif count_flip > sim_par.min_flip && count_chA<3 && good_opt(i_t-1,1) == 2
            good_opt(i_t,1) = 1;
            count_flip = 1; %resets flip counter
            count_rev = count_rev + 1;
        else %if not, keep the good option
            good_opt(i_t,1) = good_opt(i_t-1,1);
            count_flip = count_flip + 1; %counts number of trials after each flip
        end

        %calculate outcome "reward" based on p_win and p_lose
        if choice(i_t,1) == good_opt(i_t,1) %good option
            reward(i_t,1) = (double(rand < sim_par.p_win) - 0.5) * 2;
        else
            reward(i_t,1) = (double(rand < sim_par.p_lose) - 0.5) * 2;
        end

        %compute reward prediction errors, RPEs, according to delta rule,
        %single update
        RPE(i_t,choice(i_t,1)) = reward(i_t,1) - Q(i_t,choice(i_t,1));

        %update Q values according to the learning rate, alpha, and the RPEs
        if i_t < sim_par.n_trials
            Q(i_t+1,1) = Q(i_t,1) + sim_par.alpha * RPE(i_t,1);
            Q(i_t+1,2) = Q(i_t,2) + sim_par.alpha * RPE(i_t,2);
        end

    end

    %set initial values for fitting
    x0 = [sim_par.alpha, sim_par.beta];
    D = [choice, reward]; %concatenate data vector
    
    %options = optimset('Display','off');
    options = optimoptions(@fminunc,'Algorithm','quasi-newton','Display','off');
    [xout,fval,exitflag,output] = fminunc(@(x)fit_model(x,D),x0,options);

    %print output
    %fprintf('\n a=%.2f, b=%.2f, L=%.2f\n',xout(1),xout(2),fval);

%store output
%original units
%eval_fit(i_p,:) = [sim_par.alpha, sim_par.beta, xout(:,1), xout(:,2), fval, sum(reward)/sim_par.n_trials, count_rev*100/sim_par.n_trials];

%transformed units
eval_fit(i_p,:) = [sim_par.alpha, sim_par.beta, 1./(1+exp(xout(:,1))), exp(xout(:,2)), fval, sum(reward)/sim_par.n_trials, count_rev*100/sim_par.n_trials];

end

fig_handle = figure('Position', [100, 100, 900, 700]);
subplot(2,2,1)       % add first plot in 2 x 1 grid
%histogram(eval_fit(:,1)-eval_fit(:,3))
ecdf(eval_fit(:,1)-eval_fit(:,3),'bounds','on');
title('ecdf of simulated vs. estimated alpha');

%figure
%histogram(eval_fit(:,2)-eval_fit(:,4))
subplot(2,2,2)       
ecdf(eval_fit(:,2)-eval_fit(:,4),'bounds','on');
title('ecdf of simulated vs. estimated beta');

%figure
subplot(2,2,3)       
ecdf(eval_fit(:,5),'bounds','on');
title('ecdf of the obtained fit');

%figure
subplot(2,2,4)  
ecdf(eval_fit(:,6),'bounds','on');
title('ecdf of win per trial');

figure
ecdf(eval_fit(:,7),'bounds','on');
title('ecdf of normalized reversal counts');
