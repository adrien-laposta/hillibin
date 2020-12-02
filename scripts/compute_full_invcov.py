import numpy as np
import astropy.io.fits as fits
import os

data_path = '../modules/data/Hillipop/'
binning_path = data_path + 'Binning/'
spectra_path = data_path + 'Data/NPIPE/spectra/'
cov_path = data_path + 'Data/NPIPE/pclcvm/fll/'

# save paths for binning & covariance matrix
save_bin_cov = data_path + 'Data/NPIPE/pclcvm/invcov_bin_TTEETEETsims.fits'
save_binning = data_path + 'Binning_rte/binningsims.dat'

nmap = 6
nfreq = 3
nxfreq = nfreq * (nfreq +1) // 2
nxspec = nmap * (nmap - 1) // 2
frequencies = [100, 100, 143, 143, 217, 217]

b1 = np.arange(50, 800, step = 30)
b2 = np.arange(np.max(b1)+60, 1200, step = 60)
b3 = np.arange(np.max(b2)+100, 9000, step = 100)
binning = np.concatenate((b1, b2, b3))
#binning = np.arange(50, 9000, step = 30)

# Save the binning in a .dat file
np.savetxt(save_binning, binning)

# Modes to use in the full covariance matrix
modes = ["TT", "EE", "TE"]
lmax = 2500

def set_multipole_range():

    lmins = []
    lmaxs = []
    for hdu in [0, 1, 3, 4]:

        data = fits.getdata(os.path.join(binning_path, "binning.fits"), hdu + 1)
        lmins.append(np.array(data.field(0), int))
        lmaxs.append(np.array(data.field(1), int))

    lmax = np.max([max(l) for l in lmaxs])

    return(lmax, lmins, lmaxs)


def bin_matrix(x, binning):

    N = len(x)
    B = []
    for i in range(len(binning) - 1):

        line = np.zeros(N)
        for j in range(N):

            if x[j] >= binning[i] and x[j] < binning[i+1]:

                size = binning[i+1] - binning[i]
                line[j] = 1 / size

        B.append(line)

    B = np.array(B)

    return(B)


def set_lists(nmap, nfreq, frequencies):

    xspec2map = []
    list_xfq = []
    for m1 in range(nmap):

        for m2 in range(m1 + 1, nmap):

            xspec2map.append((m1, m2))

    list_fqs = []
    for f1 in range(nfreq):

        for f2 in range(f1, nfreq):

            list_fqs.append((f1, f2))

    freqs = list(np.unique(frequencies))
    xspec2xfreq = []
    for m1 in range(nmap):

        for m2 in range(m1 + 1, nmap):

            f1 = freqs.index(frequencies[m1])
            f2 = freqs.index(frequencies[m2])
            xspec2xfreq.append(list_fqs.index((f1, f2)))

    return(xspec2xfreq)


def select_covblock(cov, block = "TTTT"):

    modes = ["TT", "EE", "BB", "TE"]
    blocks = []
    for i, m1 in enumerate(modes):

        line = []
        for j, m2 in enumerate(modes):

            line.append(m1 + m2)

        blocks.append(line)

    blocks = np.array(blocks)

    i, j = np.where(blocks == block)
    i, j = int(i), int(j)
    N = len(cov)
    n = len(modes)
    L = int(N / n)

    return(cov[i * L:(i+1) * L, j*L:(j+1) * L])


def read_covmat(xf1, xf2, binning, block, nmap, nfreq, frequencies, lmax):

    fname = cov_path + "fll_NPIPE_detset_TEET_{}_{}.fits".format(xf1, xf2)
    covmat = fits.getdata(fname, 0).field(0)
    print("Reading covariance matrix {}_{}{}".format(block, xf1, xf2))
    N = int(np.sqrt(len(covmat)))
    covmat = covmat.reshape(N, N)
    covmat = select_covblock(covmat, block = block)
    covmat = covmat[:lmax+1,:lmax+1]
    N = len(covmat)

    ell = np.arange(N)
    id_binning = np.where(binning < np.max(ell))
    binning = binning[id_binning]
    B = bin_matrix(ell, binning)
    bcovmat = B.dot(covmat.dot(B.T))
    #bcovmat *= 1e24 ## K to muK
    b_ell = B.dot(ell)

    modes = ["TT", "EE", "TE", "ET"]
    mode1 = block[:2]
    mode2 = block[2:]
    id1 = modes.index(mode1)
    id2 = modes.index(mode2)
    lmin1 = set_multipole_range()[1][id1][set_lists(nmap, nfreq, frequencies).index(xf1)]
    lmax1 = set_multipole_range()[2][id1][set_lists(nmap, nfreq, frequencies).index(xf1)]
    lmin2 = set_multipole_range()[1][id2][set_lists(nmap, nfreq, frequencies).index(xf2)]
    lmax2 = set_multipole_range()[2][id2][set_lists(nmap, nfreq, frequencies).index(xf2)]
    idrow = np.where((b_ell >= lmin1) & (b_ell <= lmax1))
    idcol = np.where((b_ell >= lmin2) & (b_ell <= lmax2))
    min1, max1 = np.min(idrow), np.max(idrow)
    min2, max2 = np.min(idcol), np.max(idcol)

    bcovmat = bcovmat[min1:max1+1, min2:max2+1]

    return(bcovmat)


def produce_cov_pattern(nxfreq, modes):

    vec = []
    for mode in modes:

        for i in range(nxfreq):

            vec.append((mode,str(i)))

    cov = []
    for doublet1 in vec:

        line = []
        for doublet2 in vec:

            m1, xf1 = doublet1
            m2, xf2 = doublet2
            line.append((m1+m2, xf1+xf2))

        cov.append(line)

    return(cov)


def compute_full_covmat(nxfreq, modes, binning, nmap, nfreq, frequencies, lmax):

    print("Creating pattern for the full covariance matrix ...")
    pattern_cov = produce_cov_pattern(nxfreq, modes)
    full_cov = []
    print("Starting to fill the matrix ...")
    for i in range(len(pattern_cov)):

        line = []
        for j in range(len(pattern_cov)):

            mode, xfreq_couple = pattern_cov[i][j]
            xf1, xf2 = xfreq_couple
            xf1, xf2 = int(xf1), int(xf2)
            m1, m2 = mode[:2], mode[2:]
            try:
                line.append(
                    read_covmat(xf1, xf2, binning, mode,
                                nmap, nfreq, frequencies, lmax)
                               )
            except:
                line.append(
                    read_covmat(xf2, xf1, binning, m2 + m1,
                                nmap, nfreq, frequencies, lmax).T)

        full_cov.append(line)

    full_cov = np.block(full_cov)
    print("End of computation.")

    return(full_cov)


# Save covariance and inv covariance to .fits file for the likelihood

fcov = compute_full_covmat(nxfreq, modes, binning, nmap, nfreq, frequencies, lmax)
print(np.shape(fcov))
fileout = "fullcov.fits"
col = [fits.Column(name = 'fullcov', format = 'D', array = fcov.reshape(len(fcov) * len(fcov)))]
hdulist=fits.BinTableHDU.from_columns(col)
hdulist.writeto(fileout, overwrite=True)
invfcov = np.linalg.inv(fcov)
col = [fits.Column(name = 'invfullcov', format = 'D', array = invfcov.reshape(len(invfcov) * len(invfcov)))]
hdulist=fits.BinTableHDU.from_columns(col)
hdulist.writeto(save_bin_cov, overwrite=True)
