%% MATLAB版 MLP 回归训练：近岸海域有效波高预测
clear; clc; close all;
rng(42);

%% 1. 读取数据
T = readtable('coastal_wave_mlp_dataset_500.csv');

% 输入特征和目标变量
X = T{:, {'wind_speed','wind_dir','air_pressure', ...
          'tide_level','surface_current','current_dir','depth'}};
Y = T.Hs;

%% 2. 划分训练集 / 测试集
cv = cvpartition(size(X,1), 'HoldOut', 0.2);
idxTrain = training(cv);
idxTest  = test(cv);

XTrain = X(idxTrain, :);
YTrain = Y(idxTrain, :);
XTest  = X(idxTest, :);
YTest  = Y(idxTest, :);

%% 3. 特征标准化（只用训练集统计量）
mu = mean(XTrain, 1);
sigma = std(XTrain, 0, 1);
sigma(sigma == 0) = 1;

XTrainStd = (XTrain - mu) ./ sigma;
XTestStd  = (XTest - mu) ./ sigma;

%% 4. 训练 MLP 回归模型
% LayerSizes 可改成 [20 10]、[50 30 10] 等试试
mdl = fitrnet( ...
    XTrainStd, YTrain, ...
    'LayerSizes', [50 30], ...
    'Activations', 'relu', ...
    'Standardize', false, ...
    'Lambda', 1e-4, ...
    'IterationLimit', 5000);

%% 5. 预测
YPredTrain = predict(mdl, XTrainStd);
YPredTest  = predict(mdl, XTestStd);

%% 6. 评价指标函数
calcMAE  = @(y, yhat) mean(abs(y - yhat));
calcRMSE = @(y, yhat) sqrt(mean((y - yhat).^2));
calcR2   = @(y, yhat) 1 - sum((y - yhat).^2) / sum((y - mean(y)).^2);

% 训练集指标
MAE_train  = calcMAE(YTrain, YPredTrain);
RMSE_train = calcRMSE(YTrain, YPredTrain);
R2_train   = calcR2(YTrain, YPredTrain);

% 测试集指标
MAE_test  = calcMAE(YTest, YPredTest);
RMSE_test = calcRMSE(YTest, YPredTest);
R2_test   = calcR2(YTest, YPredTest);

%% 7. 输出结果
fprintf('===== 训练集结果 =====\n');
fprintf('MAE  = %.4f\n', MAE_train);
fprintf('RMSE = %.4f\n', RMSE_train);
fprintf('R^2  = %.4f\n\n', R2_train);

fprintf('===== 测试集结果 =====\n');
fprintf('MAE  = %.4f\n', MAE_test);
fprintf('RMSE = %.4f\n', RMSE_test);
fprintf('R^2  = %.4f\n', R2_test);

%% 8. 画图：测试集真实值 vs 预测值
figure('Color','w');
scatter(YTest, YPredTest, 45, 'filled', 'MarkerFaceAlpha', 0.75);
hold on;
xmin = min([YTest; YPredTest]);
xmax = max([YTest; YPredTest]);
plot([xmin xmax], [xmin xmax], 'r--', 'LineWidth', 1.5);
grid on;
xlabel('真实波高 Hs\_m');
ylabel('预测波高 Hs\_m');
title('测试集：真实值 vs 预测值');
legend('样本点', '理想预测线', 'Location', 'best');

%% 9. 画图：测试集样本顺序对比
figure('Color','w');
plot(YTest, 'b-o', 'LineWidth', 1.2, 'MarkerSize', 4);
hold on;
plot(YPredTest, 'r-*', 'LineWidth', 1.2, 'MarkerSize', 4);
grid on;
xlabel('测试样本编号');
ylabel('波高 Hs\_m');
title('测试集预测对比');
legend('真实值', '预测值', 'Location', 'best');

%% 10. 残差图
residuals = YTest - YPredTest;
figure('Color','w');
scatter(YPredTest, residuals, 40, 'filled', 'MarkerFaceAlpha', 0.75);
hold on;
yline(0, 'r--', 'LineWidth', 1.5);
grid on;
xlabel('预测波高 Hs\_m');
ylabel('残差');
title('测试集残差图');