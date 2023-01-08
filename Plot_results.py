import sys, os
import numpy as np
# define the function
import matplotlib.pyplot as plt
import matplotlib as mpl
import keras
# save np.load
np_load_old = np.load
cmap = plt.get_cmap('gray_r')

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

############ Load data 
outdir = 'images_out/'
data0 = np.load(outdir+'qcd_leading_jet.npz')['arr_0']
data1 = np.load(outdir+'tt_leading_jet.npz')['arr_0']

print('We have {} QCD jets and {} top jets'.format(len(data0), len(data1)))

x_data = np.concatenate((data0, data1))
# pad and normalize images
x_data = list(map(pad_image, x_data))
x_data = list(map(normalize, x_data))

y_data = np.array([0]*len(data0)+[1]*len(data1))

np.random.seed(0) # for reproducibility
x_data, y_data = np.random.permutation(np.array([x_data, y_data]).T).T

# the data coming out of previous commands is a list of 2D arrays. We want a 3D np array (n_events, xpixels, ypixels)
x_data = np.stack(x_data)

print(x_data.shape, y_data.shape)

# reshape for tensorflow: x_data.shape + (1,) = shortcut for (x_data.shape[0], 16, 22, 1)
x_data = x_data.reshape(x_data.shape + (1,)).astype('float32')
x_data /= 255.

y_data = keras.utils.to_categorical(y_data, 2)

print(x_data.shape, y_data.shape)

n_train = 50000
(x_train, x_test) = x_data[:n_train], x_data[n_train:]
(y_train, y_test) = y_data[:n_train], y_data[n_train:]
print('We will train+validate on {0} images, leaving {1} for cross-validation'.format(n_train,len(x_data)-n_train))

print(x_test.shape)

model_dir='trained_models_low_pt/'
history_logi, history_mlp, history_cnn = np.load(model_dir+'training_histories.npz')['arr_0']
model0 = keras.models.load_model(model_dir+'logi.h5')
model1 = keras.models.load_model(model_dir+'mlp.h5')
model_cnn = keras.models.load_model(model_dir+'cnn.h5')
#print(history_logi)

predictions0 = model0.predict(x_test)
predictions1 = model1.predict(x_test)
predictions_cnn = model_cnn.predict(x_test)

fig, axes = plt.subplots(1,2,figsize=(10,4))

nn=20
axes[1].scatter(nn, 1.*sum(np.argmax(predictions0, axis=1) == np.argmax(y_test, axis=1))/ len(y_test),c='b')
axes[1].scatter(nn, 1.*sum(np.argmax(predictions1, axis=1) == np.argmax(y_test, axis=1))/ len(y_test),c='y')
axes[1].scatter(nn, 1.*sum(np.argmax(predictions_cnn, axis=1) == np.argmax(y_test, axis=1))/ len(y_test),c='g')
for i, history in enumerate([history_logi, history_mlp, history_cnn]):
	axes[1].plot(history['accuracy'][:nn], c=plt.get_cmap("tab10")(i), label='train')
	axes[1].plot(history['val_accuracy'][:nn], c=plt.get_cmap("tab10")(i), ls ='--', label='validation')
	axes[0].plot(history['loss'][:nn], c=plt.get_cmap("tab10")(i), label=['logistic','small MLP', 'CNN', 'CNN v2'][i])
	axes[0].plot(history['val_loss'][:nn], c=plt.get_cmap("tab10")(i), ls='--')
for iax in range(2):
	axes[iax].set(xlabel='epoch', ylabel=['loss','accuracy'][iax])

axes[0].legend()
axes[1].legend(['train','validation'])
fig.show()
fig.savefig('figs/training_history.png')


logscale = True
logscale = dict(norm=mpl.colors.LogNorm()) if logscale else {}

fig, axes = plt.subplots(1,5,figsize=(20,2.5))
for i in range(2):
	im = axes[i].imshow(x_test[predictions_cnn.argmin(axis=0)[i],:,:,0], cmap=cmap, **logscale)
	plt.colorbar(im, ax=axes[i])
	axes[i].set(title='most {}-like jet'.format(['top','qcd'][i]))
axes[2].imshow(x_test[abs(predictions_cnn-0.5).argmin(axis=0)[0],:,:,0], cmap=cmap, **logscale)
plt.colorbar(im, ax=axes[2])
axes[2].set(title='most uncertain jet');

for iax, i in enumerate((predictions_cnn - y_test).argmin(axis=0)):
	im = axes[iax+3].imshow(x_test[i,:,:,0], cmap=cmap, **logscale)
	plt.colorbar(im, ax=axes[iax+3])
	axes[iax+3].set_title('Output: {}\nLabel = {}    --FAIL--'.format(predictions_cnn[i], y_test[i]))

fig.tight_layout()
fig.savefig('figs/cnn_jet_sample.png', dpi=150)
