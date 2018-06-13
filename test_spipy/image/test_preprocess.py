import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mimage
from spipy.image import preprocess
import copy
import sys

if __name__ == "__main__":
	data = np.load("test_adu.npy")
	artif = np.load("artifacts.npy")
	artif = np.where(artif==1)
	artif = np.vstack((artif[0], artif[1])).T
	mask = np.load("mask.npy")

	# test preprocess.hit_find
	print("\n(1) test preprocess.hit_finding")
	# simulate background
	background = np.random.poisson(1,data.shape[1:])
	hits = preprocess.hit_find(dataset=data, background=background, radii_range=[10, 100], mask=mask, cut_off=10)
	print("Predicted hit pattern index: " + str(np.where(hits==1)[0]))

	# test preprocess.fix_artifacts
	print("\n(2) test preprocess.fix_artifact")
	ref = copy.deepcopy(data[0])
	data = preprocess.fix_artifact(dataset=data, estimated_center=np.array(data[0].shape)/2, artifacts=artif, mask=mask )
	plt.subplot(1,2,1)
	plt.imshow(np.log(1+np.abs(ref)))
	plt.title('Before fix')
	plt.subplot(1,2,2)
	plt.imshow(np.log(1+np.abs(data[0])))
	plt.title('After fix')
	plt.show()

	print("\n(3) test preprocess.adu2photon")
	adu, newdata = preprocess.adu2photon(dataset=data, mask=mask, photon_percent=0.1, nproc=1, transfer=True, force_poisson=False)
	plt.imshow(np.log(1+newdata[0]))
	plt.show()

	print("\n(4) test preprocess.fix_artifact_auto")
	pl = mimage.imread('fix_art_auto.png')
	plt.imshow(pl)
	plt.show()
	sys.exit(0)

	#test_adu.npy")
	data = np.load("Your-data-path")
	ref = copy.deepcopy(data)
	newdata = preprocess.fix_artifact_auto(dataset=data, estimated_center=np.array(data[0].shape)/2, njobs=1, mask=mask, vol_of_bins=50)
	for watch in np.random.choice(data.shape[0],10,replace=False):
		plt.subplot(1,2,1)
		plt.imshow(np.log(1+np.abs(ref[watch])))
		plt.title('Before fix')
		plt.subplot(1,2,2)
		plt.imshow(np.log(1+np.abs(newdata[watch])))
		plt.title('After fix')
		plt.show()