
with open ('geo_data.txt', 'r', encoding='utf-8') as g:
    with open ('geo_data2.txt', 'w', encoding='utf-8') as g2:
        for line in g.readlines():
            temp_lines = line.split(';')
            new_line = temp_lines[1] + '\t' + temp_lines[2]
            g2.write(new_line + '\n')
