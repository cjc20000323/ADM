
with open('../file/statistics_ag_50.txt', 'r') as f:
    content = f.read()
    data = eval(content)

print(data)
print(data['actual_normal_sum'])
print(data['actual_abnormal_sum'])
print(data['judge_normal_sum'])
print(data['judge_abnormal_sum'])
print(data['both_normal_sum'])
print(data['both_abnormal_sum'])
print(data['both_to_actual_normal'])
# print(data['both_to_judge_normal'])
print(data['both_to_actual_abnormal'])
print(data['both_to_judge_abnormal'])
print(data['both_to_actual'])
print(data['both_to_judge'])

with open('../file/min_max_loss_ag_50.txt', 'r') as f:
    content = f.read()
    data = eval(content)

print(data.keys())