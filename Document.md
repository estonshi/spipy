# <center>Document</center>

<center> spipy : Python tools for Single Particle Imaging analysis and reconstruction</center>

***
### 1. Architecture

* [analyse](#analyse)
	* [q](#q)
	* [saxs](#saxs)
	* [orientation](#ori)
	* [SH_expan](#sh)
* [image](#image)
	* [radp](#radp)
	* [quat](#quat)
	* [classify](#class)
	* [preprocess](#pre)
* [merge](#merge)
	* [emc](#emc)
* [phase](#phase)
	* [phase2d](#phase2d)
	* [phase3d](#phase3d)
* [simulate](#simulate)
	* [sim](#sim)

**Example** : 

~~~python
from spipy.analyse import saxs
import spipy.image as spiimage
~~~

</br>
***
### 2. Install

> The package need Anaconda2 installed in advance

**Download**

```shell
git clone https://github.com/GeekGitman/spipy.git
```

**Make file**

```shell
./make_all.sh
```

</br>
***<span id="example"></span>
### 3. Help and examples

**You can refer to *help* function of spipy or its modules to get immediate help. For example :** 

```python
"Anaconda Ipython 2.7"
>> import spipy
>> spipy.help()
>> spipy.phase.help()
>> spipy.phase.help("phase2d")
>> spipy.phase.phase2d("help")
```

**In the *'test_spipy'* directory, you can find examples about how to use functions in every modules. For example :**

```shell
cd ./test_spipy/image
python test_classify.py
```

</br>
***
### 4. Modules
> **varialbles that begin with '#' are optional, their default values are behind '/'**

<span id="analyse"></span>
<span id='q'></span>
#### 4.1.1 spipy.analyse.q

```text
+ :numpy.1darray = cal_q (detd:float, lamda:float, det_r:int, pixsize:int)

+ :numpy.1darray = cal_r (qlist:numpy.1darray, detd:float, lamda:float, det_r:int, pixsize:int)
```

--
* **cal_q : calculate frequency q values, also known as k'-k**
	* detd : distance between detector and sample, unit=mm
	* lamda : wave length, unit=Angstrom
	* det_r : the highest resolution (radius on detector) that you want to calculate to, unit=pixels
	* pixsize : physical length of a detector pixel, unit=mm

--
* **cal_r : calculate radius list on detector corresponding to a q-list**
	* qlist : input q list, numpy.ndarray, shape=(Nq,)

<span id='saxs'></span>
#### 4.1.2 spipy.analyse.saxs

```text
+ :(numpy.ndarray, ...) = grid (input_patt:numpy.ndarray)

+ :numpy.1darray = frediel_search (pattern:numpy.2darray, estimated_center:(int, int))

+ :numpy.1darray = inten_profile_vaccurate (dataset:numpy.3darray, *exp_param:[...])

+ :numpy.1darray = inten_profile_vfast (dataset:numpy.3darray, *exp_param:[...])

+ :numpy.2darray = cal_saxs (data:numpy.3darray)
```

--
* **grid : return x and y indices of a pattern/volume**
	* input_patt : input pattern or volume. If it is a pattern, then returned value consists two 2d-arrays, one is the x coordinates of every pixels, and the other is y coordinates. If input is a scattering volume, then returned value consists 3 3d-arrays.

--
* **frediel_search : return the center point (zero frequency) of a pattern,** numpy.array([Cx,Cy])
	* pattern : input pattern. Check there is no nan or inf values.
	* estimated_center : an estimated center, (Cx',Cy')

--
* **inten\_profile\_vaccurate : return the averaged intensity radial profile.** shape=(2,Nq), the first colum is q value while the second one is radial intensities. **This function processes every image so it is more accurate, but PRETTY SLOW**
	* dataset : patterns, shape=(Nd,Nx,Ny)
	* *exp\_param : for calculating q value, please give all experiment parameters required in spipy.analyse.q.cal_q

--
* **inten\_profile\_vfast : fast calculation of intensity profile, RECOMENDED**

--
* **cal_saxs : return averging pattern of a pattern set**
	* data : input pattern set, shape=(Nd,Nx,Ny)

<span id='ori'></span>
#### 4.1.3 spipy.analyse.orientation

```text
+ :[numpy.2darray, numpy.2darray] = Sphere_randp (algo:str, radius:float, num:int)

+ :[numpy.2darray, numpy.2darray] = proc_Hammer (qlist:numpy.2darray, data:numpy.1darray)

+ :void = draw_Hammer (logi_lati:numpy.2darray, #save_dir:str/None) 

+ :void = draw_ori_Df (ori_bin:str, q_level:int)
```
--
* **Sphere_randp : return random/uniform distributed points on a pherical surface**, shape=[np.numpy(Np,3), np.numpy(Np,2)]
	* algo : algorithm, string "random" or "uniform"
	* radius : radius of the sphere, unit=pixels
	* num : numbers of points you want to generate

--
* **proc\_Hammer : transfer quaternions to Aitoff-Hammer projection coordinates**, return [ numpy.array( [ logitute[..], latitute[..], value[..] ] ), numpy.array( [ x[..], y[..], value[..] ] ) ]
	* qlist : quaternion list, numpy.array( [ w[..], qx[..], qy[..], qz[..] ] ); or xyz coordinates, numpy.array( [ x[..], y[..], z[..] ] ). That is, shape=(Nq, 4) or (Nq,3)
	* **[Notice] w = cos(the/2) , q? = q? * sin(theta/2). BE CAREFUL of the order!**
	* data : the corresponding value for every quaternion (orientation), this function do not care what these values really mean. shape=(Nq,1)

--
* **draw\_Hammer : draw figure to show distribution of quaternions with corresponding values**
	* logi\_lati : Logitute, latitute and value, shape=(Nq,3) ( The first output of *proc\_Hammer* )
	* #save_dir : string, the path if you want to save figure. default is None

--
* **draw\_ori\_Df : draw figure to show orientation distribution of DragonFly output**
	* ori\_bin : string, path of output *'orientation_xxx.bin'* file from spipy.merge.emc
	* q\_level : the *'num_div'* parameter used in spipy.merge.emc

<span id='sh'></span>
#### 4.1.4 spipy.analyse.SH_expan

```text
+ :numpy.1darray = sp_hamonics (data:{'volume':numpy.3darray, 'mask':numpy.3darray}, r:float, #L:int/10)
```
--
* **sp\_harmonics : spherical expansion of a 3D-intensity**, return Cl, shape=(L,)
	* data : { 'volume' : intensity voxel data, 'mask' : mask voxel data }, if no mask please set 'mask' : None
	* r : radius of the shell you want to expand, unit=pixels
	* #L : level of hamonics, recommended >=10

<span id="image"></span>
<span id='radp'></span>
#### 4.2.1 spipy.image.radp

```text
+ :numpy.1darray = radial_profile_2d (data:numpy.2darray, center:(float, float))

+ :numpy.1darray = radial_profile_3d (data:numpy.3darray, center:(float, float, float))

+ :numpy.2darray = shells_2d (rads:[float, ...], data_shape:(int, int), center:(float, float))

+ :numpy.2darray = shells_3d (rads:[float, ...], data_shape:(int, int, int), center:(float, float, float))

+ :numpy.2darray = radp_norm_2d (ref_Iq:numpy.1darray, data:numpy.2darray, center:(float, float))

+ :numpy.3darray = radp_norm_3d (ref_Iq:numpy.1darray, data:numpy.3darray, center:(float, float, float))
```

--
* **radial\_profile\_2d : return intensity radial profile of a 2d pattern**, shape=(Nr,), where Nr is the pixel length at the pattern corner
	* data : input pattern, shape=(Nx,Ny)
	* center : zero frequency point of the pattern, unit=pixels

--
* **radial\_profile\_3d : return intensity radial profile of a 3d field**, shape=(Nr,), where Nr is the pixel length at the volume corner
	* data : input pattern, shape=(Nx,Ny,Nz)
	* center : zero frequency point of the scattering volume, unit=pixels

--
* **shells\_2d : return xy indices of a pattern which forms a shell/circle at radius=rads**, shape=[ shell1( numpy.array( x[..], y[..] ), shape=(N1,2) ) , shell2 , ... , shelln ]
	* rads : radius of output shell, list, [ r1, r2, ..., rn ], unit=pixels
	* data_shape : size of your pattern, (size\_x, size\_y), unit=pixels
	* center : zero frequency point of the pattern, unit=pixels

--
* **shells\_3d : return xyz indices of a field which forms a spherical shell/surface at radius=rads**, shape=[ shell1( numpy.array( x[..], y[..], z[..] ), shape=(N1,3) ) , shell2 , ... , shelln ]
	* rads : radius of output shells, list, [ r1, r2, ..., rn ], unit=pixels
	* data_shape : size of your volume, (size\_x, size\_y, size\_z), unit=pixels
	* center : zero frequency point of the scattering volume, unit=pixels

--
* **radp\_norm\_2d : normalize pattern intensities by comparing intensity radial profile**, return normalized pattern
	* ref\_Iq : reference intensity radial profile, shape=(Nr,)
	* data : pattern that need to be normalized, shape=(Nx,Ny)
	* center : zero frequency point of the pattern, unit=pixels
	* **[Notice] The program do not require accurate match of shapes between 'ref\_Iq' and 'data' , but the start point of 'ref\_Iq' should be exactly the center of 'data'**

--
* **radp\_norm\_3d : normalize volume intensities by comparing intensity radial profile**, return normalized pattern
	* ref\_Iq : reference intensity radial profile, shape=(Nr,)
	* data : volume that need to be normalized, shape=(Nx,Ny,Nz)
	* center : zero frequency point of the scattering volume, unit=pixels
	* **[Notice] The program do not require accurate match of sizes between 'ref\_Iq' and 'data' , but the start point of 'ref\_Iq' should be exactly the center of 'data'**

<span id='quat'></span>
#### 4.2.2 spipy.image.quat

* In this modules, quaternion's format is numpy.array( [w, qx, qy, qz] ), where w = cos(theta/2) , q? = ? * sin(theta/2).

```text
+ :numpy.1darray = invq (q:numpy.1darray)

+ :numpy.1darray = quat_mul (q1:numpy.1darray, q2:numpy.1darray)

+ :numpy.1darray = conj (q:numpy.1darray)

+ :numpy.2darray = quat2rot (q:numpy.1darray)

+ :numpy.1darray = rot2quat (rot:numpy.2darray)

+ :numpy.1darray = rotv (vector:np.1darray, q:np.1darray)

+ :numpy.1darray = Slerp (q1:numpy.1darray, q2:numpy.1darray, t=float)
```

--
* **invq : calculate the inverse/reciprocal of a quaternion**, return q<sup>-1</sup>
	* q : input quaterion (w, qx, qy, qz)

--
* **quat_mul : multiply two quaternions**, return q1 * q2

--
* **conj : conjugate quaternion**, return q*

--
* **quat2rot : transfer quaternion to 3D rotation matrix**, return 3x3 matrix

--
* **rot2quat : transfer 3D rotation matrix to quaternion**, return quaternion

--
* **rotv : rotate a 3D vector using a quaternion**, return new vector nump.array( [x', y', z'] )
	* vector : input vector to rotate, numpy.array( [x, y, z] )
	* q : quaternion

--
* **Slerp : linear interpolation on spherical surface between two quaternions**, return new quaternion
	* q1 & q2 : two input quaternions
	* t : interpolation weight from q1 to q2, 0~1

<span id='class'></span>
#### 4.2.3 spipy.image.classify

```text
+ :[numpy.2darray, numpy.1darray] = cluster_fSpec (dataset:numpy.3darray, #low_filter:float/0.3, #decomposition:str/'SVD', #ncomponent:int/2, #nneighbors:int/10, #LLEmethod:str/'Standard')

+ :cluster_fTSNE (dataset:numpy.3darray, #low_filter:float/0.3, #no_dims:int/2, #perplexity:int/50, #use_pca:bool/True, #initial_dims:int/50, #max_iter:int/500, #theta:float:0.5, #randseed:int/-1, #verbose:bool/False)
```
--
* **cluster\_fSpec ： single-nonsingle hits clustering using linear/non-linear decomposition and spectural clustering**, return [ data_decomp{ numpy.array, shape=(Nd, ncomponent) }, label{ numpy.array, shape=(Nd, ) }], where 'data\_decomp' is the data after decomposition and 'label' is predicted label of clustering methods.
	* dataset : raw dataset, int/float, shape=(Nd, Nx, Ny)
	* #low_filter : float 0~1, the percent of area at the center part of fft intensity pattern that is used for clustering, default=0.3
	* #decomposition : decoposition method, choosen from 'LLE' (Locally Linear Embedding), 'SVD' (Truncated Singular Value Decomposition) and 'SpecEM' (Spectral Embedding), default='SVD'
	* #ncomponent : number of components left after decomposition, default=2
	* #nneighbors : number of neighbors to consider for each point, considered only when decomposition method is 'LLE', default=10
	* #LLEmethod : LLE method, choosen from 'standard' (standard locally linear embedding algorithm), 'modified' (modified locally linear embedding algorithm), 'hessian' (Hessian eigenmap method) and 'ltsa' (local tangent space alignment algorithm), default='standard'
	* **[NOTICE] The input dataset is not recommended to contain more than 5k patterns, but it's also neccessary to have more than 500 ones.
You can split the original dataset into several parts and use multi-processors to deal with them.**
	
--
* **cluster\_fTSNE : single-nonsingle hits clustering using t-SNE decomposition and KNN clustering**, return [ data_decomp{ numpy.array, shape=(Nd, ncomponent) }, label{ numpy.array, shape=(Nd, ) }], where 'data\_decomp' is the data after decomposition and 'label' is predicted label of clustering methods.
	* dataset : raw dataset, int/float, shape=(Nd, Nx, Ny)
	* #low_filter : float 0~1, the percent of area at the center part of fft intensity pattern that is used for clustering, default=0.3
	* #no_dims : number of components left after decomposition, default=2
	* #perplexity : perlexity value to evaluate P(i|j) in t-SNE, POSITIVE,  default=50
	* #use_pca : whether to use PCA to generate initiate features, default=True
	* #initial_dims : output dimensions of inititate PCA, POSITIVE, ignored if use_pca=False, default=50
	* #max_iter : max times of iterations, default=1000, suggested >500
	* #theta : the speed vs accuracy trade-off parameter, theta=1 means highest speed with lowest accuracy, default=0.5
	* #randseed : 1) if it is >=0, then use it as initiate value's generating seed; 2) else <0 then use current time as random seed, default=-1
	* #verbose : print details while running ? default=True
	* **[NOTICE] The input dataset is not recommended to contain more than 5k patterns, but it's also neccessary to have more than 500 ones.
You can split the original dataset into several parts and use multi-processors to deal with them.**

<span id='pre'></span>
#### 4.2.4 spipy.image.preprocess

```text
+ :void = fix_artifact (dataset:numpy.3darray, estimated_center:(int, int), artifacts:numpy.2darray, #mask:numpy.2darray/None)

+ :float or [float, numpy.3darray] = adu2photon (dataset:numpy.3darray, #photon_percent:float/0.9, #nproc:int/2, #transfer:bool/True, #force_poisson:bool/False)
```
--
* **fix\_artifact : reduces artifacts of dataset (adu values), patterns in the dataset should share the same artifacts**, NO RETURN, to save RAM, your input dataset is modified directly
	* dataset : FLOAT adu patterns, shape=(Nd,Nx,Ny)
	* estimated_center : estimated pattern center, (Cx,Cy)
	* artifacts : where artifacts locate in pattern (indices), shape=(Na,2), the first colum is x coordinates of artifacts and second colum is y coordinates
	* #mask : mask area of patterns, a 0/1 binary array where 1 means masked, shape=(Nx,Ny), default=None
	* **[NOTICE] This function cannot reduce backgroud noise, try preprocess.adu2photon instead**

--
* **adu2photon : evaluate adu value per photon and transfer adu patterns to photon patterns**, return adu : float **or** [adu : float, data_photonCount : numpy.ndarray, shape=(Nd,Nx,Ny)]
	* dataset : patterns of adu values, shape=(Nd, Nx, Ny)
	* #photon_percent : estimated percent of pixels that has photons, default=0.1
	* #nproc : number of processes running in parallel, default=2
	* #transfer : Ture -> evaluate adu unit and transfer to photon, False -> just evaluate, default=True
	* #force_poisson : whether to determine photon numbers at each pixel according to poisson distribution, default=False, ignored if transfer=False

<span id="simulate"></span>
<span id='sim'></span>
#### 4.3.1 spipy.simulate.sim

```text
+ work_dir:str

+ config_default:dict

+ :void = generate_config_files (pbd_file:str, #workdir:str/None, #params:dict/{})

+ :void = run_simulation()
```
--
* **work\_dir** : directory you project locates

--
* **config\_default** : default configuration parameters, a dict
	* **keys** : [reference 1](#ref1), or refer to [example codes](#example)
		* "parameters|detd" : distance between sample and detector [unit : mm]
		* "parameters|lambda" : wave length of laser [unit : angstrom]
		* "parameters|detsize" : detector size in width/height [unit : pixel]
		* "parameters|pixsize" : pixel size of detector [unit : mm]
		* "parameters|stoprad" : radius of a circle region at the center of pattern that to be masked out [unit : pixel]
		* "parameters|polarization" : correction due to incident beam polarization, value from 'x', 'y' or 'none'
		* "make\_data|num\_data" : how many patterns do you want to generate
		* "make\_data|fluence" : laser fluence [unit : photons/mm<sup>2</sup>], usually 1e10 ~ 1e14 is reasonable for most proteins

--
* **generate\_config\_files : configure simulation parameters, make project dir and copy neccessary files to your work_dir**, NO return
	* pdb_file : path of your pdb file used in simulation [/..../xx.pdb]
	* #workpath : choose a path to set up your project, ABSOLUTE PATH ! default is current dir
	* #name : give your project a name, default is None, program will choose a name for you
	* #params : parameters dict, { "section\_1|param\_1": value_1, ... }, for default program will use 'config\_default'

--
* **run\_simulation : start simulation after configuing**, NO return
	* NO input

<span id="merge"></span>
<span id='emc'></span>
#### 4.4.1 spipy.merge.emc

```text
+ config_essential:dict

+ config_advanced:dict

+ :void = new_project (data_path:str, inh5:str, #path:str/None, #name:str/None)

+ :void = config (params:dict)

+ :void = run (num_proc:int, num_thread:int, iters:int, #nohup:bool/False, #resume:bool/False, #cluster:bool/True)

+ :void = use_project (project_path:str)
```
--
* **config\_essential** : dict of important parameters
	* **keys** : [reference [1]](#ref1), or refer to [example codes](#example)
		* 'parameters|detd' : distance between sample and detector [unit : mm]
		* 'parameters|lambda' : wave length of laser [unit : angstrom]
		* 'parameters|detsize' : detector size in width/height [unit : pixel]
		* 'parameters|pixsize' : pixel size of detector [unit : mm]
		* 'parameters|stoprad' : radius of a circle region at the center of pattern that will not be used in orientation recovery, but will be merged to final scattering volume [unit : pixel]
		* 'parameters|polarization' : correction due to incident beam polarization, value from 'x', 'y' or 'none'
		* 'emc|num_div' : level that used to generate quaternions, also known as n, where M<sub>rot</sub>=10(n+5n<sup>3</sup>)
		* 'emc|need\_scaling' : whether need to scale patterns' intensities
		* 'emc|beta' : beta value of emc method
		* 'emc|beta\_schedule' : for example, if 'emc|beta\_schedule' = '1.414 10', that means for every 10 iterations, beta = beta * 1.414
		
--
* **config\_advanced** : dict of advanced parameters
	* **keys** : [reference [1]](#ref1), or refer to [example codes](#example)
		* 'parameters|ewald\_rad' : radius of ewald sphere, used to control oversampling rate
		* 'make\_detector|in\_mask\_file' : path of mask file that operate on input patterns, masked pixels will not be used for both orientation recovery and merging. **Usually, most experiments need mask file**
		* 'emc|sym_icosahedral' : bool (0/1), whether to force icosahedral symmetry in orientation recovery
		* 'emc|selection' : value chosen from 'even', 'odd' and 'None', where 'even' / 'odd' means only patterns whose index is even / odd will be used. 'None' means all patterns will be used
		* 'emc|start\_model\_file' : path of file that store initiate model which will be used at the start of emc program

--
* **new\_project : create a new project at your given path**, NO return
	* data_path : path of your dataset file, MUST be h5 file
	* inh5 : path of patterns inside h5 file, patterns should be stored in a numpy.ndarray, shape=(Nd,Nx,Ny)
	* #path : create work directory at your give path, set None to use current dir, default=None
	* #name : give a name to your project, set None to let program choose one for you, default=None
	
--
* **config : edit configure file**, NO return
	* params : dict, parameters that you want to modified. refer to [example code](#example)

--
* **run : start emc**, NO return
	* num_proc : int, how many processes to run in parallel
	* num_thread : int, how many threads in each process
	* iters : int, how many reconstruction iterations
	* #nohup : bool, whether run in the background, default=False
	* #resume : bool, whether run from previous break point, default=False
	* #cluster : bool, whether you will submit jobs using job scheduling system, if True, the function will only generate a command file at your work path without submitting it, and ignore nohup value; if False, the program will run directly. default=True
	* **[Notice] As this program costs a lot of memories, use as less processes and much threads as possible. Recommended strategy : num\_proc * num\_thread ~ number of cores in all of your CPUs. Let one cluster node support 1~2 processes. (Mentioned, using too many processes may cause low precision in merging result)**

--
* **use\_project : switch to a existing project**, NO return
	* project_path : str, the path of project directory that you want to switch to

<span id="phase"></span>
<span id="phase2d"></span>
#### 4.5.1 spipy.phase.phase2d

```text
+ :void = new_project (data_mask_path:str, #path:str/None, #name:str/None)

+ :void = config (params:dict)

+ :void = run (#num_proc:int/1, #nohup:bool/False)

+ :void = use_project (project_path:str)
```

* **All parameters**, a dict :
	* **important** : 
		* 'input|shape' : input pattern shape, default='123,123'
		* 'input|padd\_to\_pow2' : add padding to make the size of pattern to be a 2<sup>n</sup> number, default=True
		* 'input|inner\_mask' : pixels whose radius<inner\_mask are allowed to float while phasing, default=5
		* 'input|outer\_mask' : pixels whose radius>outer\_mask are set to zero,
	default=64
		* 'input|outer\_outer\_mask' : pixels whose radius are between outer\_mask and outer\_outer\_mask are allowed to float, default=None
		* 'input|mask_edges' : a switch, whether allow pixels between outer\_mask and outer\_outer\_mask to float, default=True
		* 'phasing|repeats' : how many times of independent phasing, default=40
		* 'phasing|iters' : schedual iterations for 1 phasing loop, default='300ERA 200DM 300ERA' which means '300 times error reduction algorithm -> 200 times different map algorithm -> 300 times error reduction algorithm'
		* 'phasing\_parameters|support\_size' : set restriction to the number of pixels inside final support of shrinkwrap process, default=200 (pixels)
	* **optional** (for most time you don't need to change them):
		* 'input|subtract\_percentile' : subtract the ? th percentile value for all pixels, default=None
		* 'input|spherical\_support' : radius of spherical support that added at the initiation, default=None
		* 'phasing\_parameters|background' : evaluate background while phasing, default=True 

	
--
* **new\_project : create a new project at your given path**, NO return
	* data\_mask\_path : [data\_path, user\_mask\_path] , *data\_path* is the path of pattern while *user\_mask\_path* is the path of mask file
	* #path : create work directory at your give path, set None to use current dir, default=None
	* #name : give a name to your project, set None to let program choose one for you, default=None
	* **[Notice] Your original intensity file should be 2D array '.npy' or '.mat' or '.bin', mask file must be 'npy'. Leave data\_mask\_path[1] to None if you don't have user mask**

--
* **config : edit configure file**, NO return
	* params : dict, parameters that you want to modified. refer to [example code](#example)

--
* **run : start phasing**, NO return
	* no_hup : whether run in background, default=False

--
* **show\_result : show pahsing result**, calling 'show\_result.py' in your work_dir, NO return
	* #outpath : IF you move output.h5 to another folder (not your project folder), please give me its path, default is None
	* #exp\_param : list detd, lambda, det\_r, pix\_size in a string. Used to calculate q value. e.g. '200,2.5,128,0.3', default=None

--
* **use_project : switch to an existing project**, NO return
	* project_path : path of the project directory that you want to use

<span id="phase3d"></span>
#### 4.5.2 spipy.phase.phase3d

```text
+ :void = new_project (data_path:str, #path:str/None, #name:str/None)

+ :void = config (params:dict)

+ :void = run (#num_proc:int/1, #nohup:bool/False, #cluster:bool/True)

+ :void = use_project (project_path:str)

+ :void = show_result (#outpath:str/None, #exp_param:str/None)
```
--
* **All parameters**
	* 'input|shape' : default='120,120,120'
	* 'phasing\_parameters|support\_size' : default=2000
	* **Others the same with [phase2d parameters](#phase2d)**

--
* **new_project : create a new project at your given path**, NO return
	* data_path : path of your original 3d scattering intensity data, file that contains 3D voxel data should be **'.npy'** or **'.mat'**, or Dragonfly output **'.bin'**
	* #path : create work directory at your give path, set None to use current dir, default=None
	* #name : give a name to your project, set None to let program choose one for you, default=None

--
* **config : edit configure file**, NO return
	* params : dict, parameters that you want to modified. refer to [example code](#example)

--
* **run : start phasing**, NO return
	* #num_proc : this function maps multi-processes automatically, so you need to figure out how many processes to run in parallel, default=1
	* #nohup : whether run in background, default=False
	* #cluster : bool, whether you will submit jobs using job scheduling system, if True, the function will only generate a command file at your work path without submitting it, and ignore nohup value; if False, the program will run directly. default=True

--
* **show\_result : show pahsing result**, calling 'show\_result.py' in your work_dir, NO return
	* #outpath : IF you move output.h5 to another folder (not your project folder), please give me its path, default is None
	* #exp\_param : list detd, lambda, det\_r, pix\_size in a string. Used to calculate q value. e.g. '200,2.5,128,0.3', default=None

--
* **use_project : switch to an existing project**, NO return
	* project_path : path of the project directory that you want to use
