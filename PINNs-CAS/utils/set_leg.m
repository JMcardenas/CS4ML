if samp_num == 1
    % CAS
    name   = [arch_name 'CAS' '-' actname ];
    marker = marker_list(1,:);
else
    % MC
    name   = [arch_name 'MC' '-' actname];
    marker = marker_list(2,:);
end

line_width = 1;

if act_num == 1
    % 5x50-Tanh
    face_color = default_color(1,:);
    edge_color = default_color(1,:);
    
elseif act_num == 2
    % 10x100-Tanh
    face_color = default_color(2,:);
    edge_color = default_color(2,:);
    
elseif act_num == 3
    % 5x50-ReLU
    face_color = default_color(3,:);
    edge_color = default_color(3,:);
    
elseif act_num == 4
    % 10x100-ReLU
    face_color = default_color(4,:);
    edge_color = default_color(4,:);
end