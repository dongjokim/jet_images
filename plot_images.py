import sys, os
import numpy as np
# define the function
import matplotlib.pyplot as plt
import matplotlib as mpl

# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


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

cmap = plt.get_cmap('gray_r')

outdir = 'images_out/'
leading_jet_images = np.load(outdir+'tt_leading_jet.npz')['arr_0']
all_jet_images = np.load(outdir+'tt_all_jets.npz')['arr_0']
jetpep = np.load(outdir+'tt_jetpep.npz')['arr_0']

leading_jet_images0 = np.load(outdir+'qcd_leading_jet.npz')['arr_0']
all_jet_images0 = np.load(outdir+'qcd_all_jets.npz')['arr_0']
jetpep0 = np.load(outdir+'qcd_jetpep.npz')['arr_0']

#print(leading_jet_images0)

std_jet_images0 = list(map(pad_image, leading_jet_images0))
std_jet_images = list(map(pad_image, leading_jet_images))
#print(std_jet_images0)

logscale = True
logscale = dict(norm=mpl.colors.LogNorm()) if logscale else {}

for idx in range(5):
	fig, axes = plt.subplots(1,2, figsize=(12,3))
	for iax, ax in enumerate(axes):
		im = ax.imshow([std_jet_images0,std_jet_images][iax][idx], cmap=cmap, **logscale)
		#im = ax.imshow(all_jet_images[iax][idx],cmap=cmap, **logscale)
		plt.colorbar(im, ax=ax)
		ax.set(title='{} jet: $p_T=${:.0f} GeV'.format(['QCD','top'][iax], [jetpep0,jetpep][iax][idx][0][0]))

	plt.show()
