import xml.etree.ElementTree as ET
import operator
from pprint import pprint

def hasNumbers(inputString):
	return any(char.isdigit() for char in inputString)

data = ['traindata_part_b_iobes.xml']

dict_pos = {}
dict_iobes = {}
dict_ = {}
count = 0

for dat in data:
	tree = ET.parse(dat)
	for n in tree.findall(".//token[@type]"):
		count += 1
		types = n.get('type').replace('-', ' ').split(' ')

		if len(types) > 1:
			if types[1] not in dict_pos:
				dict_pos[types[1]] = 1
			else:
				dict_pos[types[1]] += 1
		
		if types[0] not in dict_iobes:
			dict_iobes[types[0]] = 1
		else:
			dict_iobes[types[0]] += 1

		if n.get('type') not in dict_:
			dict_[n.get('type')] = 1
		else:
			dict_[n.get('type')] += 1

for k, v in dict_pos.iteritems():
	dict_pos[k] = v * 100 / count
dict_pos = sorted(dict_pos.items(), key=lambda kv: kv[1])
print "dict_pos:", len(dict_pos), dict_pos, "\n"

for k, v in dict_iobes.iteritems():
	dict_iobes[k] = v * 100 / count
dict_iobes = sorted(dict_iobes.items(), key=lambda kv: kv[1])
print 'dict_iobes:', len(dict_iobes), dict_iobes, "\n"

for k, v in dict_.iteritems():
	dict_[k] = v * 100 / count
dict_ = sorted(dict_.items(), key=lambda kv: kv[1])
print len(dict_), dict_

print "Total token", count
