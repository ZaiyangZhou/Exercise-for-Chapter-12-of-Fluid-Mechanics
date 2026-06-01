%% 生成近岸海域有效波高回归数据集（500样本）
clear; clc;
rng(42);   % 固定随机种子，保证可复现

n = 500;

%% 1. 基础环境变量
wind_speed = min(max(9.5 + 4.0 .* randn(n,1), 1.0), 22.0);          % m/s
wind_dir = 360 .* rand(n,1);                                        % deg
air_pressure = min(max(1010 + 6.5 .* randn(n,1), 995), 1025);       % hPa
tide_level = min(max(0.45 + 0.45 .* randn(n,1), -0.6), 1.6);        % m

surface_current = 0.12 + 0.045 .* wind_speed + 0.08 .* randn(n,1);
surface_current = min(max(surface_current, 0.05), 1.25);            % m/s

current_dir = mod(wind_dir + 25 .* randn(n,1), 360);                % deg
depth = 8 + (28 - 8) .* rand(n,1);                                  % m

%% 2. 风流夹角影响
angle_diff = abs(wind_dir - current_dir);
angle_diff = min(angle_diff, 360 - angle_diff);
dir_factor = cosd(angle_diff);   % [-1, 1]

%% 3. 构造目标变量 Hs_m（有效波高）
hs = 0.12 ...
    + 0.055 .* (wind_speed .^ 1.18) ...
    + 0.018 .* (1015 - air_pressure) ...
    + 0.22 .* max(tide_level, -0.2) ...
    + 0.35 .* surface_current .* (0.55 + 0.45 .* dir_factor) ...
    + 0.015 .* sqrt(depth) ...
    + 0.06 .* sind(wind_dir - 135) ...
    + 0.08 .* randn(n,1);

hs = min(max(hs, 0.15), 4.5);

%% 4. 保留三位小数
wind_speed = round(wind_speed, 3);
wind_dir = round(wind_dir, 3);
air_pressure = round(air_pressure, 3);
tide_level = round(tide_level, 3);
surface_current = round(surface_current, 3);
current_dir = round(current_dir, 3);
depth = round(depth, 3);
hs = round(hs, 3);

%% 5. 生成表格并保存
T = table(wind_speed, wind_dir, air_pressure, tide_level, ...
          surface_current, current_dir, depth, hs, ...
          'VariableNames', {'wind_speed', 'wind_dir', ...
                            'air_pressure', 'tide_level', ...
                            'surface_current', 'current_dir', ...
                            'depth', 'Hs'});

writetable(T, 'coastal_wave_mlp_dataset_500.csv');

disp(T(1:10,:));
disp('数据集已保存为 coastal_wave_mlp_dataset_500.csv');