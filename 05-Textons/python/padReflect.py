import numpy as np
import ipdb

def padReflect(im,r):
	impad = np.zeros(np.array(im.shape)+2*r);
	impad[r:-r,r:-r] = im # middle
	impad[:r,r:-r] = np.flipud(im[:r,:]) # top
	impad[-r:,r:-r] = np.flipud(im[-r:,:]); # bottom
	impad[r:-r,:r] = np.fliplr(im[:,:r]); # left
	impad[r:-r,-r:] = np.fliplr(im[:,-r:]); # right
	impad[:r,:r] = np.flipud(np.fliplr(im[:r,:r])); # top-left
	impad[:r,-r:] = np.flipud(np.fliplr(im[:r,-r:])); # top-right
	impad[-r:,:r] = np.flipud(np.fliplr(im[-r:,:r])); # bottom-left
	impad[-r:,-r:] = np.flipud(np.fliplr(im[-r:,-r:])); # bottom-right
	return impad
