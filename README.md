# jet_images

- Pythia needs to be compiled with python3
https://pythia.org/latest-manual/PythonInterface.html
	- warning, [https://root-forum.cern.ch/t/include-code-h-and-python-3-11/52425](https://root-forum.cern.ch/t/include-code-h-and-python-3-11/52425) , it won't work with python3.11 > downgrade to 3.10
- Need Madgraph5_aMC@NLO
		- You will find madgraph5 package on the following page:  https://launchpad.net/madgraph5.
	    For this program, you just need to untar it.
	    To check if mg5 is correctly install you directly try to run it by doing:
	    ./bin/mg5_aMC
      
Run z01... to make jet images
Run z02... to train and results
