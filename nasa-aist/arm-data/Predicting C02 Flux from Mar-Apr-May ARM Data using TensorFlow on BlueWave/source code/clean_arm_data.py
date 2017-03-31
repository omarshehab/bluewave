import csv

with open('ARM4mDec2002Jul2015OklahomaV2_edit.csv', 'rb') as inp, open('ARM4mDec2002Jul2015OklahomaV2_edit2.csv', 'wb') as out:
    writer = csv.writer(out)
    num_col=len(next(csv.reader(inp))) # Read first line and count columns
    inp.seek(0)
    flag = 0
    count = 1
    for row in csv.reader(inp):
       count = count + 1
       for column in range(0, num_col):
         if row[column] == "y" or row[column] == "-9999.0":
            flag = 1
            break   
       if flag == 1:
            flag = 0
            continue
       if count > 29 and count < 36:
          print str(row)
       writer.writerow(row)
