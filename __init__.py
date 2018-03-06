import analyse
import image
import simulate
import phase
import merge

def help():
	import os
	print("spipy software includes packages-modules-functions : ")
	for f in os.listdir(__file__.split('__init__.py')[0]):
		if '.' in f:
			continue
		else:
			print("    |- " + f)
			try:
				eval(f).help()
			except:
				pass
