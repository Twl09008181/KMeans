import matplotlib.pyplot as plt


with open("speed.txt",'r') as result:
    lines = result.readlines()
    speedRecord = {}
    for i in range(1,len(lines),2):
        line1, line2 = lines[i].split(" "),lines[i+1].split(" ")
        label = line1[0]+line1[1]
        if label not in speedRecord:
            speedRecord[label] = []
        speedRecord[label].append(float(line2[2]))

    for i, label in enumerate(speedRecord):
        plt.plot([1,2,4,8,16], speedRecord[label], label=label)

        

        

#plt.axis([1,16,0,20])
plt.xlabel("core")
plt.ylabel("time(s)")

plt.xticks([1,2,4,8,16])

plt.legend()
plt.show()
