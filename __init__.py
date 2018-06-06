import analyse
import image
import simulate
import phase
import merge

def help():
	import os
	print("spipy software includes packages-modules-functions : ")
	dir_name = os.path.dirname(__file__)
	for f in os.listdir(__file__.split('__init__.py')[0]):
		if (not os.path.isdir(os.path.join(dir_name,f))) or ('.' in f):
			continue
		else:
			print("    |- " + f)
			try:
				eval(f).help()
			except:
				pass
