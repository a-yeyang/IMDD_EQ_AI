function plotspectrum(x, fs,title1)
    % plot_dual_sided_spectrum 用于绘制信号 x 的双边频谱
    % 输入:
    %   x  - 输入信号
    %   fs - 采样率（Hz）
    N = length(x);             % 信号长度
    X = fft(x);                % 计算 FFT
    X_shifted = fftshift(X);   % 将零频率移到中心
    f = (-N/2:N/2-1) * (fs/N);   % 构造双边频谱频率轴
    semilogy(f, abs(X_shifted)/N);
    fontSize = 14;             % 统一字体大小（可自定义）
    xlabel('f','FontSize', fontSize);
    ylabel('dB','FontSize', fontSize);
    title(title1,'FontSize', fontSize + 2);
    grid on;
   
end