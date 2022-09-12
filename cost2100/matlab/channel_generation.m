% Choose a Network type out of 
% {'IndoorHall_5GHz','SemiUrban_300MHz','Indoor_CloselySpacedUser_2_6GHz','SemiUrban_CloselySpacedUser_2_6GHz','SemiUrban_VLA_2_6GHz'}
% to parameterize the COST2100 model
Network = 'IndoorHall_5GHz';
% In COST2100, # links = # BSs x # MSs
% Set Link type to `Multiple' if you work with more than one link
% Set Link type to `Single' otherwise
Link = 'Single';
% Choose an Antenna type out of
% {'SISO_omni', 'MIMO_omni', 'MIMO_dipole', 'MIMO_measured', 'MIMO_Cyl_patch', 'MIMO_VLA_omni'}
Antenna = 'SISO_omni';
% ...and type of channel: {'Wideband','Narrowband'}.
Band = 'Wideband';

scenario = 'LOS'; % {'LOS'} only LOS is available
freq = [-10e6 10e6]+5.3e9; % [Hz}
snapRate = 1000; % Number of snapshots per s
snapNum = 1000; % Number of snapshots
MSPos  = [10 5  0]; % [m]
MSVelo = -[1 0 0]; % [m/s]
BSPosCenter  = [10 10 0]; % Center position of BS array [x, y, z] [m]
BSPosSpacing = [0 0 0]; % Inter-position spacing (m), for large arrays
BSPosNum = 1; % Number of positions at each BS site, for large arrays

tic
%% Get the MPCs from the COST 2100 channel model
[...
    paraEx,...       % External parameters
    paraSt,...       % Stochastic parameters
    link,...         % Simulated propagation data for all links [nBs,nMs]
    env...           % Simulated environment (clusters, clusters' VRs, etc.)
] = cost2100...
(...
    Network,...      % Model environment
    scenario,...     % LOS or NLOS
    freq,...         % [starting freq., ending freq.]
    snapRate,...     % Number of snapshots per second
    snapNum,...      % Total # of snapshots
    BSPosCenter,...  % Center position of each BS
    BSPosSpacing,... % Position spacing for each BS (parameter for physically very-large arrays)
    BSPosNum,...     % Number of positions on each BS (parameter for physically very-large arrays)
    MSPos,...        % Position of each MS
    MSVelo...        % Velocity of MS movements
    );         
toc

%% Ccombine propagation data with antenna patterns
% Construct the channel data
% The following is example code
% End users can write their own code
h_need = [];
for i = 1:50
    delta_f = (freq(2)-freq(1))/256;
    h_omni = create_IR_omni(link,freq,delta_f,Band);
    %switch Band
        %case 'Wideband'
            %H_omni = fft(h_omni,[],2);                        
            %figure;
            %mesh((freq(1):delta_f:freq(2))*1e-6,1:size(H_omni,1),10*log10(abs(H_omni)));
            %xlabel('Frequency [MHz]')
            %ylabel('Snapshots')
            %zlabel('Power [dB]')
            %title('Frequency response for the SISO channel')

            %figure;
            %plot(1:snapNum, pow2db(mean(abs(H_omni).^2, 2)));
            %xlabel('Snapshots')
            %ylabel('Power [dB]')
            %title('Frequency response for the SISO channel')
        %case 'Narrowband'
            %figure,plot(1:size(h_omni,2),10*log10(abs(h_omni)))                        
            %xlabel('Snapshots')
            %ylabel('Power [dB]')
            %title('Impulse response for the SISO channel')
    %end

    h_need(:, i) = h_omni(:, 1);
end