folder = 'HR_generated/';
arr_file = dir([folder, '*.png']);

filename = 'filename_list_generated.txt';
fid = fopen(filename, 'w+');

num_arr_file = size(arr_file,1);

for i = 1:num_arr_file
    name = arr_file(i).name;
    
    if i ~= num_arr_file
        file_name = sprintf('%s\n', name);
    else
        file_name = sprintf('%s', name);
    end
    fwrite(fid, file_name);
end
fclose(fid);