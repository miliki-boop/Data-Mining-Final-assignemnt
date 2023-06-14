input_file = 'test_key.txt'
output_file = 'output.txt'

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 去除空行
lines = [line.strip() for line in lines if line.strip()]

with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
