function save1(Rx, filename)
    receiversignal = Rx.band.E;
    filename = fullfile('D:\IMDD_EQ_AI\vpi_data.txt');
    save(filename, 'receiversignal', '-ascii');
end