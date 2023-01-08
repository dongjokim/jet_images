import sys, os
import matplotlib.pyplot as plt
import numpy as np

# find pythia path so you can import it
pythia_path = 'pythia8308/'
cfg = open(pythia_path+"examples/Makefile.inc")
lib = pythia_path+"lib"
print(lib)
for line in cfg:
    if line.startswith("PREFIX_LIB="): lib = line[11:-1]; break
print(lib)
sys.path.insert(0, lib)
import pythia8



def write_mg_cards(PTRANGE, delta=10, nevents=200000, extra=''):
    """
    Write two Madgraph cards for production of ttbar and jj events.
    Apply generator-level cut on the particle momenta given by PTRANGE,
    allowing for a tolerance delta.
    If argument extra is specified, append it to output directories
    """
    with open('generate_tt.mg5','w') as f:
        f.write("""
    generate p p > t t~
    output jets_tt{3}
    launch
    madspin=none
    done
    set nevents {2}
    set pt_min_pdg {{ 6: {0} }}
    set pt_max_pdg {{ 6: {1} }}
    decay t > w+ b, w+ > j j
    decay t~ > w- b~, w- > j j
    done    
    """.format(PTRANGE[0]-delta, PTRANGE[1]+delta, nevents, extra))

    with open('generate_qcd.mg5','w') as f:
        f.write("""
    generate p p > j j
    output jets_qcd{3}
    launch
    done
    set nevents {2}
    set ptj {0}
    set ptjmax {1}
    done    
    """.format(PTRANGE[0]-delta, PTRANGE[1]+delta, nevents, extra))

def extend_jet_phi(phi, jet_phi):
    """
    If a jet center is close to either 0 or 2*pi, its constituents could be on the other side
    of the periodicity line. This takes care of this problem by remapping phi to be either
    above 2*pi or below zero.
    """
    if abs(jet_phi + np.pi)<1.: # phi close to -pi
        return phi-2*np.pi*int(abs(phi-np.pi) <1-abs(jet_phi + np.pi))
    elif abs(jet_phi - np.pi)<1.: # phi close to pi
        return phi+2*np.pi*int(abs(-phi-np.pi) < 1-abs(jet_phi - np.pi)) 
    else: 
        return phi

def make_image_leading_jet(leading_jet, leading_jet_constituents):
    """ 
    Jet and constituents are passed as pythia vec4 objects.
    Restricts image grid to within a DeltaR=1.2 range around jet center.
    Returns pT-weighted histogram, and tuple with histogram grid.
    """
    jet_phi = leading_jet.phi()
    jet_eta = leading_jet.eta()
    ### redefine grid to only be Delta R=1 around jet center
    yedges = [phi for phi in phiedges if abs(phi-jet_phi)<=1.2+(phiedges[1]-phiedges[0])]
    xedges = [eta for eta in etaedges if abs(eta-jet_eta)<=1.2+(etaedges[1]-etaedges[0])]
    
    jet_constituents = np.array([ [c.pT(), c.eta(), extend_jet_phi(c.phi(), jet_phi) ] for c in leading_jet_constituents ])

    histo, xedges, yedges =  np.histogram2d(jet_constituents[:,1], jet_constituents[:,2], bins=(xedges,yedges), weights=jet_constituents[:,0])
    
    ### transpose to have eta=x, phi=y
    return histo.T, (xedges, yedges)
    
def make_image_event(all_jets, all_constituents):
    """ 
    Jets are passed as pythia vec4 objects
    Returns list of pT-weighted histogram, and tuple with histogram grids, covering full (eta,phi) ranges
    """
    
    out=[]
    for i in range(len(all_jets)):
        jet_phi = all_jets[i].phi()
        
        jet_constituents = np.array([ [c.pT(), c.eta(), extend_jet_phi(c.phi(), jet_phi) ] for c in all_constituents[i] ])

        histo, xedges, yedges =  np.histogram2d(jet_constituents[:,1], jet_constituents[:,2],bins=(etaedges,phiedges),weights=jet_constituents[:,0])
    
        ### append to output (transpose to have eta=x, phi=y)
        out.append(histo.T)
    
    return out, (xedges, yedges)

def run_pythia_get_images(lhe_file_name, PTRANGE = [500., 700.], PTRANGE2=None, nevents=10**6, plot_first_few=True):
    """
    Take an LHE file, run pythia on it, outputs images.
    For each event, cluster jets, check if the two highest pT jets are in PTRANGE and PTRANGE2,
    and make 2D histograms of the leading jet and of the whole event.
    If plot_first_few=True, plot both images for first 5 events
    """
    
    # unzip LHE file if it is zipped
    if lhe_file_name.endswith('gz') and not os.path.isfile(lhe_file_name.split('.gz')[0]): 
        os.system('gunzip < {} > {}'.format(lhe_file_name, lhe_file_name.split('.gz')[0]))
    lhe_file_name = lhe_file_name.split('.gz')[0]
    if not os.path.isfile(lhe_file_name): raise Exception('no LHE file')
    
    PTRANGE2 = PTRANGE if PTRANGE2 is None else PTRANGE2
    
    pythia = pythia8.Pythia()
    ### read LHE input file
    pythia.readString("Beams:frameType = 4")
    pythia.readString("Beams:LHEF = "+lhe_file_name)

    pythia.init()
    ### define jet parameters: antikt, R, pT_min, Eta_max
    slowJet = pythia8.SlowJet(-1, 1.0, 20, 2.5)

    # outputs: lists of leading jet and full detector images, and (pT, eta, phi) of each jet
    leading_jet_images, all_jet_images = [], []
    jetpep=[]

    iplot=0
    ### Begin event loop. Generate event. Skip if error or file ended. Print counter
    for iEvent in range(0,nevents):
        if not pythia.next(): continue

        print('{}\r'.format(iEvent//10*10)),

        ### Cluster jets. List first few jets. Excludes neutrinos by default
        slowJet.analyze(pythia.event)

        njets = len([j for j in range(0,slowJet.sizeJet()) if slowJet.p(j).pT()> PTCUT])

        jetpep.append([[slowJet.p(j).pT(), slowJet.p(j).eta(), slowJet.p(j).phi()] for j in range(0, njets)])
        jet_list = [ slowJet.p(j) for j in range(0, njets)] # if PTRANGE2[0]<=slowJet.p(j).pT()<=PTRANGE2[1]]
        jet_constituent_list=[ [ pythia.event[c].p() for c in slowJet.constituents(j)] for j in range(0, njets)]

        ### at least two high-pT large R jets in the right range
        if njets<2: continue 
        if not (PTRANGE[0] < jetpep[iEvent][0][0] < PTRANGE[1] and PTRANGE2[0] < jetpep[iEvent][1][0] < PTRANGE2[1]): continue

        hh, (xx, yy) = make_image_leading_jet(jet_list[0], jet_constituent_list[0])
        hh1, _ = make_image_event(jet_list, jet_constituent_list)
        
        leading_jet_images.append(hh)
        all_jet_images.append(hh1)
        
        if plot_first_few and iplot<5:
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))    
            ax1.pcolor(xx,yy, hh, cmap=cmap)
            ax2.pcolor(etaedges, phiedges, sum(hh1), cmap=cmap)
            for j in jet_list:
                ax2.add_artist(plt.Circle((j.eta(),j.phi()),1, color='r', fill=False) )
            for h in [-np.pi, np.pi]: 
                ax2.axhline(h, ls='--', lw=1, c='gray')
            iplot+=1

    return leading_jet_images, all_jet_images, np.array(jetpep)


def pad_image(image, max_size = (16,22)):
    """
    Simply pad an image with zeros up to max_size.
    """
    size = np.shape(image)
    px, py = (max_size[0]-size[0]), (max_size[1]-size[1])
    image = np.pad(image, (list(map(int,((np.floor(px/2.), np.ceil(px/2.))))), list(map(int,(np.floor(py/2.), np.ceil(py/2.))))), 'constant')
    return image

def normalize(histo, multi=255):
    """
    Normalize picture in [0,multi] range, with integer steps. E.g. multi=255 for 256 steps.
    """
    return (histo/np.max(histo)*multi).astype(int)
    
    
########################################################
#############        Run everything        #############
#############      one run by default      #############
########################################################
etaedges = np.arange(-3,3+0.01,0.12)
phiedges = np.arange(-np.pi*4/3,np.pi*4/3+0.01,np.pi/18.)
cmap = plt.get_cmap('gray_r')
PTCUT = 50.
nevents = 50000


if __name__ == "__main__":

    outdir = 'images_out/'
    if not os.path.isdir(outdir): os.system('mkdir {}'.format(outdir))
    cwd = os.getcwd()


    ### write madgraph cards, run madgraph   -- PTRANGE=500-700
    write_mg_cards([500,700], nevents=nevents)

    os.system('cd madgraph; bin/mg5_aMC  {}'.format(os.path.join(cwd, 'generate_tt.mg5')))
    os.system('cd madgraph; bin/mg5_aMC  {}'.format(os.path.join(cwd, 'generate_qcd.mg5')))

    ### run pythia
    lhe_file_name = 'madgraph/jets_tt/Events/run_01_decayed_1/unweighted_events.lhe.gz'
    leading_jet_images, all_jet_images, jetpep = run_pythia_get_images(lhe_file_name, PTRANGE=[500,700], PTRANGE2=[450,700], plot_first_few=False)

    np.savez_compressed(outdir+'tt_leading_jet.npz', leading_jet_images)
    np.savez_compressed(outdir+'tt_all_jets.npz', all_jet_images)
    np.savez_compressed(outdir+'tt_jetpep.npz', jetpep)

    lhe_file_name = 'madgraph/jets_qcd/Events/run_01/unweighted_events.lhe.gz'
    leading_jet_images, all_jet_images, jetpep = run_pythia_get_images(lhe_file_name, PTRANGE=[500,700], PTRANGE2=[450,700], plot_first_few=False)

    np.savez_compressed(outdir+'qcd_leading_jet.npz', leading_jet_images)
    np.savez_compressed(outdir+'qcd_all_jets.npz', all_jet_images)
    np.savez_compressed(outdir+'qcd_jetpep.npz', jetpep)

    # one can just do another run at another pT range by copying the commands above, e.g. write_mg_cards([800,900], nevents=nevents)
    