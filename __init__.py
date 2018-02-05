import analyse
import image
import simulate
import phase

def help():
	import os
	print("spipy software includes modules : ")
	for f in os.listdir(__file__.split('__init__.py')[0]):
		if '.' in f:
			continue
		else:
			print("    |- " + f)
			eval(f).help()