import xml.etree.ElementTree as ET
import operator
from pprint import pprint

def hasNumbers(inputString):
	return any(char.isdigit() for char in inputString)

tree = ET.parse('traindata.xml')
root = tree.getroot()


dict = {}
for n in tree.findall(".//nonterminal[@value]"):
	if n.get("value") not in dict:
		dict[n.get("value")] = 1
	else:
		dict[n.get("value")] += 1

print "Most frequent POS tag", max(dict.items(), key=operator.itemgetter(1))[0], "having occurence of", max(dict.items(), key=operator.itemgetter(1))[1]

dict_ = {}
clash = []
number = []
with_dot = []
email_and_links = []
for n in tree.findall(".//nonterminal[@value]"):
	if "alt." in n.text and len(n.text) > 1:
		with_dot.append(n.text)

	if "@" in n.text or "://" in n.text:
		email_and_links.append(n.text)

	if n.text not in dict_:
		dict_[n.text] = n.get("value")
	elif dict_[n.text] != n.get("value"):
		clash.append([n.text, dict_[n.text], n.get("value")])
	if hasNumbers(n.text) == True:
		number.append(n.text)

#pprint(clash)
print "Word having two different POS tags:", len(clash), "out of", len(dict_), "words"

#pprint(number)
print "Token which contain any number: ", len(number)

#pprint(with_dot)
print "Token which have weird dot etc: ", len(with_dot)

print "Token with email or any url : ", len(email_and_links)
