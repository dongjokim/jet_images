# jet_images

1. Pythia needs to be compiled with python3
 - download pythia from https://pythia.org/ and untar it
  https://pythia.org/latest-manual/PythonInterface.html
	- warning, [https://root-forum.cern.ch/t/include-code-h-and-python-3-11/52425](https://root-forum.cern.ch/t/include-code-h-and-python-3-11/52425) , it won't work with python3.11 > downgrade to 3.10
 - go to the pythia directroy and issue the command
 	./configure --with-python-config=python3-config
	and make
	and make install
	
2. Need Madgraph5_aMC@NLO
		- You will find madgraph5 package on the following page:  		https://launchpad.net/madgraph5.
	    For this program, you just need to untar it.
	    To check if mg5 is correctly install you directly try to run it by doing:
	    ./bin/mg5_aMC
3. Make a soft link of your pythia directory with "ln -sf ~/softwares/pythia8309 .
4. Run z01... to make jet images
5. Run z02... to train and results
