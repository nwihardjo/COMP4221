import xml.etree.ElementTree as ET
import operator
from pprint import pprint

def hasNumbers(inputString):
	return any(char.isdigit() for char in inputString)

tree = ET.parse('traindata_part_a_iobes.xml')
root = tree.getroot()


dict = {}
count = 0

for n in tree.findall(".//token[@type]"):
	count += 1
	if n.get("type") not in dict:
		dict[n.get("type")] = 1
	else:
		dict[n.get("type")] += 1

print dict
print "Total token", count
