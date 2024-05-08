close all
clear all
clc

filename = "result(model_GRU_221108).txt";

data = textread(filename);

gesture_num = 36;
N = size(data,1);
label = data(:,1);
predicted = data(:,2);
matched = data(:,3);
matched_type = data(:,4);
matched_link = data(:,5);


disp("accuracy(all): " + num2str(100*sum(matched)/N) + "%")
disp("accuracy(type): " + num2str(100*sum(matched_type)/N) + "%")
disp("accuracy(link): " + num2str(100*sum(matched_link)/N) + "%")

%%
accuracies = zeros(gesture_num,2);
for i = 1:N
   accuracies(label(i)+1,2) = accuracies(label(i)+1,2) + 1;
   if matched(i)
      accuracies(label(i)+1,1) = accuracies(label(i)+1,1) + 1; 
   end
end

disp(compose("\n" + "accuracy table:"))
table_str = "";
for i = 1:7
   table_str = table_str + num2str(100*accuracies(i+7*4,1)/accuracies(i+7*4,2)) + "\t";
   table_str = table_str + num2str(100*accuracies(i+7*3,1)/accuracies(i+7*3,2)) + "\t";
   table_str = table_str + num2str(100*accuracies(i+7*0,1)/accuracies(i+7*0,2)) + "\t";
   table_str = table_str + num2str(100*accuracies(i+7*1,1)/accuracies(i+7*1,2)) + "\t";
   table_str = table_str + num2str(100*accuracies(i+7*2,1)/accuracies(i+7*2,2)) + "\t";
   
   temp1 = accuracies(i+7*0,1) + accuracies(i+7*1,1) + accuracies(i+7*2,1) + accuracies(i+7*3,1) + accuracies(i+7*4,1);
   temp2 = accuracies(i+7*0,2) + accuracies(i+7*1,2) + accuracies(i+7*2,2) + accuracies(i+7*3,2) + accuracies(i+7*4,2);
   table_str = table_str + num2str(100*temp1/temp2) + "\n";
end
table_str = table_str + num2str(100*sum(accuracies(29:35,1))/sum(accuracies(29:35,2))) + "\t";
table_str = table_str + num2str(100*sum(accuracies(22:28,1))/sum(accuracies(22:28,2))) + "\t";
table_str = table_str + num2str(100*sum(accuracies(1:7,1))/sum(accuracies(1:7,2))) + "\t";
table_str = table_str + num2str(100*sum(accuracies(8:14,1))/sum(accuracies(8:14,2))) + "\t";
table_str = table_str + num2str(100*sum(accuracies(15:21,1))/sum(accuracies(15:21,2))) + "\t";
table_str = table_str + num2str(100*sum(matched)/N) + "\n";
disp(compose(table_str))

%%
matched_type_matrix = zeros(5,6);
case_type = zeros(5);
for i = 1:N
    type = floor(label(i)/7);
    
    case_type(type+1) = case_type(type+1) + 1;
    matched_type_matrix(type+1, floor(predicted(i)/7)+1) = matched_type_matrix(type+1, floor(predicted(i)/7)+1) + 1;
end

disp(compose("\n" + "accuracy table(type):"))
type_table_str = "";
order_disp = [5, 4, 1, 2, 3, 6];
for idx = 1:5
    i = order_disp(idx);
    for idx2 = 1:6
        j = order_disp(idx2);
        type_table_str = type_table_str + num2str(100*matched_type_matrix(i,j)/case_type(i)) + "\t";
    end
    type_table_str = type_table_str + "\n";
end
disp(compose(type_table_str))
