%2024 FINAL CURRENCY UNION + PRICE RIGIDITY 


 
    %CURRENCY UNION + PRICE 
    % Figure 5 of my paper: Incomplete pass-through and welfare: Currency union vs Inflation targeting
    % Welfare analysis of a NK with incomplete pass-through and external habit formation in a CURRENCY UNION FOR SOUTH AMERICA
    % THE MAIN RESULT ON WELFARE IS WITH A PRODUCTIVITY SHOCK Z 
    close all;
 
    var        
    pi_h  
    mc 
    pi_f   
    psi_f  
    pi 
    delta_x 
    y  
    z  
    x  
    c  
    q   
    delta_psi_f   
    delta_s   
    pi_star    
    y_star   
    i  
    i_star  
    delta_q    
    y_gap   
    y_flex   
    z_star  
    x_flex  
    y_star_flex  
    mc_star     
    y_gap_star  
 
 
    s
    p     
 
 
 
 
    
    n
    a
    ;
 
    varexo
    eps_pih  
    eps_pif  
    eps_z  
    eps_pistar   
    eps_star_i  
    eps_star_z  
    eps_q   
 
 
 
    eps_a   
 
    ;
 
 
    parameters 
    beta  
    gamma 
    phi 
    sigma 
    eta 
    rho_istar
    rho_i
 
    rho_z  
    h 
    psi_pi 
    psi_y 
    theta_h 
    theta_f 
 
 
    phie
    phiy
    phip
 
 
 
 
 
 
   // theta_w 
    //eps_w 
    //uw 
 
 
    rho_a       
    alpha
 
    cu % Currency Union parameter. If cu=0, then CU 
    ;
 
    gridphie = 20 ;     % number of parameter grid points 
    gridthew = 40 ; 
    gridthep = 40 ; 
 
    phies   = linspace(0.001,0.99,gridphie);  % default range for exchange rate coefficient    % use for 3D welfare analysis
    thews   = linspace(0.001,0.99,gridthew);  % default range for wage calvo parameter
    //thews   = linspace(0.77,0.99,gridthew);  % default range for wage calvo parameter
   // thews   = linspace(0.77,0.99,gridthew);  % default range for wage calvo parameter
    theps   = linspace(0.001,0.99,gridthep);  % default range for price calvo parameter
 
    thewbench = 0.8 ;
    thewpos = round(thewbench*gridthew);
    thepbench = 0.8 ;
    theppos = round(thepbench*gridthep);
 
 
    // Calibrations
 
    % Values 
    phies = [0.9999 0] ;    % 0 = Currency Union
    phip = 100;
    phiy = 0;
 
 
 
 
    beta = 0.99;
    gamma = 0.124;
 
    theta_h = 0.9637;
    theta_f = 0.975;
 
   // theta_w = 0.8 ;
    //eps_w = 4.3 ;
    //uw = 0.000000000001; 
    //uw = 0.000000000001; 
 
 
 
 
    h = 0.99;
    phi = 3;
    sigma = 0.09;
    eta = 0.83;
    rho_istar = 0.921;
    rho_i = 0.921;
 
    rho_z = 0.92;
    psi_pi = 1.08;
    psi_y = 0.75;
 
 
    phie = 0.999;
    phiy = 0;
    phip = 100;
 
 
    rho_a = 0.74 ;
    alpha =  0.26 ;
 
 
    
 
    lamp   =   ((1-beta*theta_h)*(1-theta_h)/theta_h)*((1-phi)/(1-phi+phi*eta)) ;
 
                                                            % start of loops
    for i1 = 1:length(thews)                                    % loop 1 (thew)
    completion = i1/length(thews)
    //kappa_f = thews(i1);  
    //psi_pi = thews(i1);
    //theta_f = thews(i1);   % Final results with these parameter ON FOR WELFARE ANALYSIS 
    //theta_f = thews(i1);   % Final results with these parameter ON FOR WELFARE ANALYSIS 
    //h = thews(i1);   % Parameter for external habit
    theta_f = thews(i1);   % Parameter for external habit
 
    //theta_w = thews(i1);   % Final results with these parameter ON FOR WELFARE ANALYSIS 
    //psi_pi = thews(i1);
    aux=phies;
    //end
 
 
    for i2 = 1:2                               
    cu=1;                 % 0 = Currency Union
 
    if i2 ==1
    cu=1;                 % 1 = Independent 
    elseif i2 ==2
    cu=0;                % 0 =  Currency Union
    end
 
 
% Loop for i3
for i3 = 1:7
veps_pih = 0;
veps_pif = 0;
veps_z = 0;
veps_pistar = 0;
veps_star_i = 0;
veps_star_z = 0;
veps_q = 0;
 
 
veps_a = 0;
 
if i3 ==1
veps_pih=0.01;
elseif i3 ==2
veps_pif=0.01;
elseif i3 ==3
veps_z=0.01;
elseif i3 ==4
veps_pistar=0.01;
elseif i3 ==5
veps_star_i=0.01;
elseif i3 ==6
veps_star_z=0.01;
elseif i3 ==7
veps_q=0.01;
elseif i3 ==8
veps_a = 0.01;
end
 
 
    model_local_variable 
    A B   C D 
    kappa_h kappa_f  
   // lambda_w
    ;
 
 
    model(linear);
 
    #kappa_h=((1-theta_h)*(1-beta*theta_h))/theta_h;
    #kappa_f=((1-theta_f)*(1-beta*theta_f))/theta_f;
 
    //% wage rigidity (down)
    //#lambda_w = ((1-theta_w)*(1-beta*theta_w))/theta_w*(1+eps_w*phi);
    //#lambda_w = ((1-theta_w)*(1-beta*theta_w))/theta_w;
 
    #A=(h*sigma*(phi*gamma*eta*(1-gamma)+1)) / (sigma*(phi*gamma*eta*(2-gamma)+1)+(1-h)*(1-gamma)^2*phi) ;
    #B = (sigma*(1+phi)) / (sigma*(phi*gamma*eta*(2-gamma)+1)+(1-h)*(1-gamma)^2*phi);
    #C = ((1+phi)*(1-h)) / (phi*(1-h)+sigma); 
    #D = (sigma*h) / (phi*(1-h)+sigma);
 

    y_star_flex = C*z_star + D*y_star_flex(-1);
 
    y_flex=(1+phi/phi) * (z-z_star) - x_flex/phi +  y_star_flex;
    x_flex= A*x_flex(-1) + B*(z-h*z(-1) - (z_star-h*z_star(-1))) ;
 
    [name='Equation (3), Domestic Inflation'] 
    pi_h = beta*pi_h(+1) + kappa_h*mc + eps_pih ;
 
 
    [name='Equation (3), Imported Inflation'] 
    pi_f = beta*pi_f(+1) + kappa_f*psi_f + eps_pif ;
 
 
    [name='Equation (3), CPI inflation'] 
    pi = pi_h + gamma*delta_x;
 
 
    [name='Equation (6), marginal costs'] 
    mc = phi*y - (1+phi)*z + gamma*x + sigma*(1-h)^-1*(c-h*c(-1));
 
 
    [name='Equation (4), Real exchange rate'] 
    q =  (1 - gamma)*x + psi_f ;
 
 
    [name='Equation (5), Definition l.o.p gap'] 
    delta_psi_f = cu*delta_s + pi_star - pi_f;
 
 
    [name='Equation (6), Terms of trade'] 
    delta_x = pi_f - pi_h;
 
 
    [name='Equation (7), Link equation'] 
    (c-h*c(-1)) = (y_star-h*y_star(-1)) + 1/sigma*(1-h) * ((1-gamma)*x + psi_f);
 
 
    [name='Equation (13) Uncovered interest parity condition']
    (i - pi(+1)) - (i_star - pi_star(+1)) = delta_q(+1) + eps_q;
 
 
    [name='Equation (15), Goods market clearing'] 
    (1-gamma)*c = y - gamma*eta*(2-gamma)*x - gamma*eta*psi_f - eta*y_star;
 
 
    [name='Below (12), Monetary Policy'] 
    //i = phip*pi+phiy*y_gap + phie*q  ;             
    i = rho_i * i(-1) + (1 - rho_i) * (psi_pi * pi + psi_y * y_gap )  ;             
 
 
 
    [name='Equation (*), CPI inflation definition'] 
    pi = p - p(-1) ;
 
 
    [name='Below (12), Output gap']
    y_gap = y - y_flex;
 
 
    [name='Above Equation (6), Production shock']
    z = rho_z*z(-1) + eps_z;
 
 
    [name='Above Equation (6), Foreign output']
    y_star - h*y_star(-1) = (y_star(+1) - h*y_star) - 1/sigma*(1-h) * (i_star - pi_star(+1));
 
 
    [name='Above Equation (15), Foreign inflation']
    pi_star = beta*pi_star(+1) + kappa_h*mc_star + eps_pistar;
 
 
    [name='Above Equation (6), Foreign Marginal cost']
    mc_star = phi*y_star - (1+phi)*z_star + sigma*(1-h)^-1 * (y_star-h*y_star(-1)) ;
 
 
    [name='Below (12), Foreign Monetary Policy'] 
    i_star = rho_istar*i_star(-1) + (1-rho_istar)*(psi_pi*pi_star + psi_y*y_gap_star) + eps_star_i;
 
 
    [name='Below (12), Foreign Output gap']
    y_gap_star = y_star - y_star_flex;
 
 
    [name='Below (12), Foreign Productivity Shock']
    z_star = rho_z*z_star(-1) + eps_star_z;
 
 
    [name='Below (12), delta_q']
    delta_q = q  - q(-1);
 
 
    [name='Below (12), delta_x']
    delta_x = x  - x(-1);
 
 
    [name='Below (12), delta_s']
    //cu*delta_s = s  - s(-1);
    delta_s = s  - s(-1);
 
 
    [name='Below (12), delta_psi_f']
    delta_psi_f = psi_f   - psi_f (-1);
 
 
 
    
 
    [name='Equation (c), c'] 
    n = (1 / 1-alpha) * ( y - a )  ;
 
 
    
    [name='Equation (e), e'] 
    a = rho_a * a(-1) + eps_a ;
 
 
    end;
 
    options_.noprint = 1;
    steady;
 
 
    shocks;
    var eps_pih = veps_pih; 
    var eps_pif = veps_pif; 
    var eps_z = veps_z;  
    var eps_pistar = veps_pistar; 
    var eps_star_i = veps_star_i; 
    var eps_star_z = veps_star_z; 
    var eps_q = veps_q; 
    
 
    var eps_a = veps_a; 
 
    end;
 
stoch_simul(irf=10, nofunctions,nograph,noprint)  pi, y_gap, delta_psi_f ;
 
//stoch_simul(irf=10, nofunctions,nograph,noprint)  pi, y_gap, delta_psi_f  u_w;
 
//stoch_simul(irf=10, nofunctions,nograph,noprint)  pi, y_gap, delta_psi_f  pi_w;
 
  
% ***********
% * welfare loss analysis *
% ***********
 
vdp = oo_.var(1,1);
vygap = oo_.var(2,2);
vpsi_fgap = oo_.var(3,3);
//vdw = oo_.var(4,4);
 
 
weight_dp = 0.33 ;
weight_ygap = 0.33;
weightdw_psifgap = 0.33;
 
 
 
//weight_dp = 0.25 ;
//weight_ygap = 0.25;
//weightdw_psifgap = 0.25;
//weight_dw = 0.25 ;
 
 
 
//weightn = 0.5*(1-nu)*(1+phi)*(1-alf) ;
//weight_dp = 0.5*(1-gamma)*eta/lamp ;
//weightdw = 0.5*(1-nu)*epsw*(1-alf)/lamw ;
 
 
lossdp = weight_dp * vdp ;
lossygap = weight_ygap * vygap ;
losspsi_fgap = weightdw_psifgap * vpsi_fgap ;
//lossdw = weight_dw * vdw ;
 
 
loss = lossdp + lossygap + losspsi_fgap  ;
 
//loss = lossdp + lossygap + losspsi_fgap + lossdw  ;
 
 
 
wl_loss(i1,i2,i3)= 100*loss ;
wl_lossdp(i1,i2,i3)= 100*lossdp ;
wl_lossygap(i1,i2,i3)= 100*lossygap ;
wl_losspsi_fgap(i1,i2,i3)= 100*losspsi_fgap ;
 
 
 //wl_lossdw(i1,i2,i3)= 100*lossdw ;
 
 
% Plot welfare 
 
mark1 = 'b-o';
mark2 = 'r-d';
mark3 = 'k-s';
mark4 = 'm-*';
 
 
// legend('inflation targeting', 'currency union'); ORIGINAL IMPORTANT !!!!
 
// sorry fa, but i guess yoou have to re-do the codes forr the graphs :) 
 
 
 
 
end;
end;
end;
 
 
 
figure(4)
subplot(2,2,1)
plot(thews,wl_loss(:,1,1),mark1,    thews,wl_loss(:,2,1),mark2)
ylabel('Welfare loss','FontSize',10)
xlabel('\theta_F','FontSize',10)
title('Domestic Inflation Shock','FontSize',10)
legend('South American Indep.', 'South American Union');
 
subplot(2,2,2)
plot(thews,wl_loss(:,1,2),mark1,    thews,wl_loss(:,2,2),mark2)
ylabel('Welfare loss','FontSize',10)
xlabel('\theta_F','FontSize',10)
title('Imported Inflation Shock','FontSize',10)
legend('South American Indep.', 'South American Union');
 
subplot(2,2,3)
plot(thews,wl_loss(:,1,3),mark1,    thews,wl_loss(:,2,3),mark2)
ylabel('Welfare loss','FontSize',10)
xlabel('\theta_F','FontSize',10)
title('Productivity Shock','FontSize',10)
legend('South American Indep.', 'South American Union');
 
subplot(2,2,4)
plot(thews,wl_loss(:,1,4),mark1,    thews,wl_loss(:,2,4),mark2)
ylabel('Welfare loss','FontSize',10)
xlabel('\theta_F','FontSize',10)
title('Foreign Inflation Shock','FontSize',10)
legend('South American Indep.', 'South American Union');
 
 
figure(5)
subplot(2,2,1)
plot(thews,wl_loss(:,1,5),mark1,    thews,wl_loss(:,2,5),mark2)
ylabel('Welfare loss','FontSize',10)
xlabel('\theta_F','FontSize',10)
title('Foreign Int. Rate Shock','FontSize',10)
legend('South American Indep.', 'South American Union');
 
subplot(2,2,2)
plot(thews,wl_loss(:,1,6),mark1,    thews,wl_loss(:,2,6),mark2)
ylabel('Welfare loss','FontSize',10)
xlabel('\theta_F','FontSize',10)
title('Foreign Productivity Shock','FontSize',10)
legend('South American Indep.', 'South American Union');
 
subplot(2,2,3)
plot(thews,wl_loss(:,1,7),mark1,    thews,wl_loss(:,2,7),mark2)
ylabel('Welfare loss','FontSize',10)
xlabel('\theta_F','FontSize',10)
title('RER Shock','FontSize',10)
legend('South American Indep.', 'South American Union');
 
 
 
% ***********
% * welfare components*
% ***********
 
figure(6)
plot(thews,wl_loss(:,1,1)/wl_loss(thewpos,1,1),mark1,    thews,wl_lossdp(:,1,1)/wl_loss(thewpos,1,1),mark2,    thews,wl_lossygap(:,1,1)/wl_loss(thewpos,1,1),mark3,    thews,wl_losspsi_fgap(:,1,1)/wl_loss(thewpos,1,1),mark4)
ylabel('Welfare loss','FontSize',14)
xlabel('\theta_F','FontSize',14)
title('Domestic Inflation Shock','FontSize',10)
legend('total','price inflation','output gap','lop gap');
 
figure(7)
plot(thews,wl_loss(:,1,2)/wl_loss(thewpos,1,2),mark1,    thews,wl_lossdp(:,1,2)/wl_loss(thewpos,1,2),mark2,    thews,wl_lossygap(:,1,2)/wl_loss(thewpos,1,2),mark3,    thews,wl_losspsi_fgap(:,1,2)/wl_loss(thewpos,1,2),mark4)
ylabel('Welfare loss','FontSize',14)
xlabel('\theta_F','FontSize',14)
title('Imported Inflation Shock','FontSize',10)
legend('total','price inflation','output gap','lop gap');
 
 
figure(8)
plot(thews,wl_loss(:,2,1)/wl_loss(thewpos,2,1),mark1,    thews,wl_lossdp(:,2,1)/wl_loss(thewpos,2,1),mark2,    thews,wl_lossygap(:,2,1)/wl_loss(thewpos,2,1),mark3,    thews,wl_losspsi_fgap(:,2,1)/wl_loss(thewpos,2,1),mark4)
ylabel('Welfare loss','FontSize',14)
xlabel('\theta_F','FontSize',14)
title('Domestic Inflation Shock','FontSize',10)
legend('total','price inflation','output gap','lop gap');
 
figure(9)
plot(thews,wl_loss(:,2,2)/wl_loss(thewpos,2,2),mark1,    thews,wl_lossdp(:,2,2)/wl_loss(thewpos,2,2),mark2,    thews,wl_lossygap(:,2,2)/wl_loss(thewpos,2,2),mark3,    thews,wl_losspsi_fgap(:,2,2)/wl_loss(thewpos,2,2),mark4)
ylabel('Welfare loss','FontSize',14)
xlabel('\theta_F','FontSize',14)
title('Imported Inflation Shock','FontSize',10)
legend('total','price inflation','output gap','lop gap');
 
 
