import numpy as np
import h5py
import matplotlib.pyplot as plt
from spipy.image import preprocess

if __name__ == "__main__":
	data = np.load("test_adu.npy")
	artif = np.load("artifacts.npy")
	artif = np.where(artif==1)
	artif = np.vstack((artif[0], artif[1])).T
	mask = np.load("mask.npy")

	# test preprocess.fix_artifacts
	print("\n(1) test preprocess.fix_artifact")
	
	preprocess.fix_artifact(dataset=data, estimated_center=np.array(data[0].shape)/2, artifacts=artif, mask=mask )
	plt.imshow(np.log(1+data[5]))
	plt.show()

	print("\n(2) test preprocess.adu2photon")
	adu, newdata = preprocess.adu2photon(dataset=data, photon_percent=0.9, nproc=2, transfer=True, force_poisson=True)
	plt.imshow(np.log(1+newdata[5]))
	plt.show()
