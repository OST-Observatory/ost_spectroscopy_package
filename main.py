import numpy as np
from astropy.io import fits, ascii as asc
from astropy.table import Table
from astroquery.nist import Nist
from glob import glob
import scipy.signal as sig
import scipy.stats as stat
import matplotlib.pyplot as plt
from photutils.aperture import RectangularAperture, aperture_photometry
import astropy.units as u
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from matplotlib import rcParams
from scipy.spatial import cKDTree


# matplotlib plot style parameters
rcParams["text.usetex"]     = False
rcParams['font.family']     = 'serif'
rcParams['font.serif']      = 'Palatino'
rcParams["xtick.bottom"] = True
rcParams["xtick.top"] = True
rcParams["xtick.direction"] = "in"
rcParams["ytick.left"] = True
rcParams["ytick.right"] = True
rcParams["ytick.direction"] = "in"

method = "median"

config = {}

DADOS_ST8_specs = {
    "order_width" : 130.,
    "Norders"      : 3,
}

def flatten_list(list2d):
    """A utility function that reduces the dimansion of a list by one. Good for nested sequences

    Args:
        list2d (List): A multi-dimensional list to be squeezed.

    Returns:
        List: The flattened version of the inuput list.
    """     
    result = []
    for list1d in list2d: result += list(list1d)
    return result



def create_masterfile(directory_list, method=np.median, output="", datatype=np.float64, **kwargs) -> np.ndarray:
    """A utility function that takes a LIST of directories that are looked up for respective fits files. All of the files are taken and according to the averaging method to be used, taken the mean/median of. The method needs to be a function that provides an "axis" option like the np.mean or np.median function. They can be used alternatively as well.
    - You can/have to specify a mask to filter for the fits files only with the "mask" argument
    - "method" specifies the stack operation (default is "numpy.median"). It must be a function, able to operate on a 3-dimensional numpy ndarray, collapsing it on the first of all three axes
    """
    data        = []
    frames      = []
    succ, counter = 0, 0
    for directory in directory_list:
        if "mask" in kwargs:
            frames += glob(directory + kwargs["mask"])
        else:
            if directory[-1] in ["/", "\\"]: directory += "*"
            frames += glob(directory)

    for frame in frames:
        try:
            data.append(fits.getdata(frame))
            succ += 1
        except:
            Warning("Cannot read the data in file:\n\t", frame, "\nMake sure that this is a valid fits file")
        counter += 1
    
    data = np.array(data, dtype=datatype)
    master_data = method(data, axis=0)

    if output != "":
        fits.writeto(filename=output, data=master_data, overwrite=True)
    return master_data



def get_strong_emission_lines(ion, threshold=50, spectral_range=(4000, 7000)):
    """
    A helper function to retrieve NIST spectra and automatically filter them for "undetectable" entries. Returns an Astropy.QTable object with the line strengths and wavelengths

    Returns:
        astropy.table.Table: The astropy table wit the data stored in it.
    """
    literatureSPECTRUM = Nist.query(4000*u.AA, 7000*u.AA, linename=ion, wavelength_type="vac+air") # load the spectral data from NIST with the help of a NIST query
    mask = []
    for strength in list(literatureSPECTRUM["Rel."]):
        try:
            if float(strength) >= threshold:    # check is the value is actually a floarting point number AND if it is larger than a threshold intensity
                mask.append(True)               # if both conditions are satified, consider the respective element as a detectable line
            else:
                mask.append(False)              # if the line's intensity is lower than the threshold, do not consider the line ...
        except:
            mask.append(False)                  # ... this also hold if the strength cannot be even converted into a numerical value and throws an exception
    
    literatureSPECTRUM.keep_columns(["Rel.", "Observed"]) # reduce the table to only the necessary columns
    
    literatureSPECTRUM = literatureSPECTRUM[mask] # apply filter mask on the data

    literatureSPECTRUM["Observed"] = np.array(literatureSPECTRUM["Observed"], dtype=np.float64) # set datatype to float
    literatureSPECTRUM["Rel."] = np.array(literatureSPECTRUM["Rel."], dtype=np.float64) # set datatype to float


    return literatureSPECTRUM # return the astropy.QTable instance as the result



# the cross-match routine outside the observation class
def cross_match(lit_lines, **kwargs):
    """Feed an list of lines and their relative strengths from observations and the literature.
    
    - obs_spectrum (numpy 1d array): The spectrum as a list of intensities
    - obs_lines (numpy list of peaks, in the style of the output of the scipy.find_peaks function. I.e. provide a 1d array of indices that correspond to the stencils of a respective 1d array at which those peaks are found). This is an alternative input to the "obs_spectrum" input
    - obs_length (int) This number determines the length of the spectrum array. This parameter is only required when you entered an input for the "obs_lines" argument of this function. Otherwise it will be ignored.
    - lit_lines: Astropy Table containing the line wavelengths and qualitative line strengths as well as the ion that they are referring to.
    
    The arguments should be in the form of two tuples, containing two lists each, e.g. ([wavelengths], [strengths]) ... blablabla
    
    This routine returns a transformation that relates the pixel coordinate the frame with an absolute wavelength.
    """

    # Input parameters ...
    peak_threshold      = 0.1 # multiple of the standard deviation of the spectrum (this is a parameter for the peak detection in the observation spectrum)
    wavelength_range    = (5000., 6800.)
    sampling            = 3.5 # Angstrom per pixel
    sampling_range      = (0.75, 1.25) # in multiples of the expected wavelength dispersion
    psf_width           = 2.
    resolution          = 1500.
    min_template_strength_level = 2

    if isinstance(lit_lines, str):
        print("[INFO] Entered a string as the literature input. Assume it directs to a file. Attempting to read:", lit_lines)
        
        lit_lines           = asc.read(lit_lines, delimiter="\t")
    
    elif isinstance(lit_lines, Table):
        print("[INFO] The literature data seems to be an Astropy Table. Proceed assuming the correct naming pattern ...")
    
    # Apparently the user specified a spectrum that they want to use as a comparison to the template. So check if the imput is valid
    if "obs_spectrum" in kwargs:
        try:
            obs_spectrum    = np.ravel(kwargs["obs_spectrum"])
        except:
            print("[ERROR] Your input for 'obs_spectrum' does not seem to be interpretable as an array. Check your input. Your input was of type:", type(kwargs["obs_spectrum"]))
        
        obs_spectrum        -= np.median(obs_spectrum)

        signal_std          = np.std(obs_spectrum)
        
        print("[INFO] Std =", signal_std)
        obs_lines, lines_dict   = sig.find_peaks(obs_spectrum, height=signal_std * peak_threshold, prominence=5)
        
        
        obs_spectrum        = np.zeros_like(obs_spectrum)
        # obs_spectrum[peaks] = peaks_dict["peak_heights"] / peaks_dict["peak_heights"] ######################## comment out the normalization if you wish to have peaks with different sizes (stronger peaks are correspondingly larger)
        obs_spectrum[obs_lines] = 1.
    
    # If the user only provides the location of the spectral lines, then check if the corresponding input meets the requirements.
    elif "obs_lines" in kwargs:
        try:
            obs_lines       = np.asarray(kwargs["obs_lines"], dtype=int)
            print("[INFO] You have entered an array with the the array indices of the lines that you have observed. Proceed to construct the normalized spectrum.")
            
        except:
            print("[ERROR] The array with wavelength stencils does not seem to be a valid array-like data structure. You entered a variable of type:", type(kwargs["obs_lines"]), ". This is not convertible into an array.")
        
        if "obs_length" in kwargs:
            try:
                spectrum_length = int(kwargs["obs_length"]) # read in the length of the digitized spectrum
            
            except:
                print("[ERROR] Cannot interpret the 'obs_length' keyword as an integer value. You entered a variable of type:", type(kwargs["obs_length"]))
        
            obs_spectrum        = np.zeros(spectrum_length)
            obs_spectrum[obs_lines] = np.ones_like(obs_lines) # at the respective indices, set the spectrum to one.
        
        else:
            raise Exception("If you want to use the 'obs_lines' routine, you need to specify the length of the array with the 'obs_length' keyword.")
    
    else:
        raise print("You have to provide an observed spectrum as a numpy array or a list of lines at certain array indices. You must provide exactly one of both. Otherwise I don't know how to proceed ...")
    
    fig = plt.figure()
    
    gs = fig.add_gridspec(ncols=3, nrows=4, width_ratios=[10, 10, 1], wspace=0.05)
    
    # start with the first axis that displays the peak positions of the observed calibration spectrum
    ax                  = fig.add_subplot(gs[0, :])
    
    # blur the spectrum such that the delta peaks become gaussian lines. This avoids that neighboring peaks are not identified as those
    obs_spectrum        = gaussian_filter(obs_spectrum, psf_width)
    ax.step(np.arange(len(obs_spectrum)), obs_spectrum)
    
    # the peaks that have been found with indicators
    ax.plot(obs_lines, obs_spectrum[obs_lines], marker="o", mew=1, mfc="none", markersize=5, lw=0, color="tab:red")
    
    ax.set_xlabel("Pixel coordinate")
    ax.set_ylabel("Observational Signal (re-modeled and de-noised)")
    
    # create the template array to correlate the observed spectrum with
    
    mid_wavelength      = np.mean(wavelength_range)
    template_lengths    = np.arange(
        int((wavelength_range[1] - wavelength_range[0]) / (sampling * np.max(sampling_range))),
        int((wavelength_range[1] - wavelength_range[0]) / (sampling * np.min(sampling_range))),
        1,
        dtype=int
    )
    ##############
    
    wavelength_increments = []
    templates_wavelengths = []
    templates             = []
    for length in template_lengths:
        template_wavelengths = np.linspace(*wavelength_range, length)
        templates_wavelengths.append(template_wavelengths)
        
        wavelength_increment = template_wavelengths[1] - template_wavelengths[0] # lambda per pixel
        wavelength_increments.append(wavelength_increment)
        
        strong_lines_mask = lit_lines["Strength"] >= min_template_strength_level # only consider really strong lines in the template
        
        lines_digitized   = np.digitize(lit_lines["Wavelength"][strong_lines_mask], template_wavelengths)
        
        strength_exponent = 0. # set this value to zero to normalize all the peaks to 1.
        
        template_data   = np.zeros(template_wavelengths.size + 1) # add one for the overflow bin
        template_data[lines_digitized] = lit_lines["Strength"][strong_lines_mask]**strength_exponent
        template_data   = gaussian_filter(template_data[:-1], mid_wavelength / (2. * resolution)) # cut off the last bin since this is the overflow bin
        templates.append(template_data)

    number_of_correlations = len(wavelength_increments)

    # plot the shape of the template that we will correlate the spectrum with
    ax = fig.add_subplot(gs[1, :])
    plt.step(np.arange(template_data.size), template_data)
    
    
    corr_filler         = np.zeros(np.max(template_lengths)) # append this to the end and the beginning of the array to be correlated. This avoid unwanted edge-effects ...
    corr_signals        = [
        np.correlate(
            np.concatenate((corr_filler, obs_spectrum, corr_filler)),
            template,
            mode="same"
        )[corr_filler.size:-corr_filler.size] for template in tqdm(templates) # chop off the left and right edge that are effected by boundary effects anyways.
    ]
    corr_signals_inv    = [
        np.correlate(
            np.concatenate((corr_filler, obs_spectrum, corr_filler)),
            np.flip(template),
            mode="same"
        )[corr_filler.size:-corr_filler.size] for template in tqdm(templates) # chop off the left and right edge that are effected by boundary effects anyways.
    ]
    
    ax                  = fig.add_subplot(gs[2, 0]) # the axis in which the correlation plot will be plotted at
    ax_inv              = fig.add_subplot(gs[2, 1]) # the axis in which the inverse correlation will be plotted at
    
    # generate the correlation array. It has the dimensions: [2 (0=no template inversion, 1= with template inversion), number_of_correlations, len(obs_spectrum)]
    correlation         = np.array([corr_signals, corr_signals_inv])
    
    # find teh maximum correlation value
    max_corr            = np.max(correlation)
    
    img = ax.pcolormesh(
        np.arange(correlation.shape[2]), wavelength_increments, correlation[0, :, :], vmax=max_corr
    )
    
    ax.set_xlabel("pixel offset")
    ax.set_ylabel(r"$\Delta\mathrm{\mathring{A}}/\Delta\mathrm{px}$")

    img_inv = ax_inv.pcolormesh(
        np.arange(correlation.shape[2]), wavelength_increments, correlation[1, :, :], vmax=max_corr
    )
    
    ax_inv.set_yticklabels([])
    ax.tick_params(axis="x", which="both", direction="in", bottom=True, top=True)
    ax.tick_params(axis="y", which="both", direction="in", left=True, right=True)
    ax_inv.tick_params(axis="x", which="both", direction="in", bottom=True, top=True)
    ax_inv.tick_params(axis="y", which="both", direction="in", left=True, right=True)
    
    cax = fig.add_subplot(gs[2, 2])
    plt.colorbar(img, cax=cax, pad=0)
        
    
    correlation_sorted = np.array(np.unravel_index(
        np.argsort(correlation, axis=None),
        correlation.shape
    ))
    
    # in the correlation array, find the respective index 
    correlation_match = correlation_sorted[:, -1]
    inverted          = correlation_match[0] == 1 # check if the template has to be inverted in order to get the observations to match in correlation matrix
    
    # this is the wavelength scale at which the correlation function maximizes
    wavelength_increment_match = wavelength_increments[correlation_match[1]] # this is a floating point value
    
    if inverted:
        ax_inv.axvline(correlation_match[2], ls="dashed", color="tab:red", lw=0.5) # draw a red indicator at the point where the correlation maximizes (this line inciates the pixel offset of the template with respect to the observed spectrum)
        ax_inv.axhline(wavelength_increment_match, ls="dashed", color="tab:red", lw=0.5) # draw a red indicator at the point where the correlation maximizes (this line characterizes the optimal wavelength sampling of the template with respect to the observed spectrum)
    
    else:
        ax.axvline(correlation_match[2], ls="dashed", color="tab:red", lw=0.5) # draw a red indicator at the point where the correlation maximizes (this line inciates the pixel offset of the template with respect to the observed spectrum)
        ax.axhline(wavelength_increments[correlation_match[1]], ls="dashed", color="tab:red", lw=0.5) # draw a red indicator at the point where the correlation maximizes (this line characterizes the optimal wavelength sampling of the template with respect to the observed spectrum)
        
    
    ax = fig.add_subplot(gs[3, :])
    
    # find the correct pixel offset
    reference = np.zeros((len(obs_spectrum) + template_lengths[correlation_match[1]], 2 * len(obs_spectrum) + template_lengths[correlation_match[1]])) # template stays fixed. Then shift the spectrum through
    
    print(len(obs_spectrum)+template_lengths[correlation_match[1]], reference.shape)
    for i in range(len(obs_spectrum)+template_lengths[correlation_match[1]]):
        if inverted:
            reference[i, i:i+len(obs_spectrum)] = np.flip(obs_spectrum)
        else:
            reference[i, i:i+len(obs_spectrum)] = obs_spectrum
    
    reference_sum = np.expand_dims(np.concatenate((np.zeros_like(obs_spectrum), templates[correlation_match[1]], np.zeros_like(obs_spectrum))), axis=0)
    
    reference_sum = reference_sum * reference
    reference_sum = np.sum(reference_sum, axis=1)
    
    offset_pix = int(np.argwhere(reference_sum == np.max(reference_sum)))
    print(offset_pix)
    
    wavelengths = np.arange(
        wavelength_range[0] - (len(obs_spectrum) * wavelength_increment_match),
        wavelength_range[1] + (len(obs_spectrum) * wavelength_increment_match) + wavelength_increment_match/1000., # add a tiny bit to allow the arange function to include the upper limit also ... othwise it is being cropped off ...
        wavelength_increment_match
        ) # the uncropped wavelength axis

    print(np.arange(-offset_pix, reference.shape[1] - offset_pix, 1))
    print(reference.shape, reference, len(wavelengths))
    ax.plot(wavelengths, reference[offset_pix])
    ax.plot(wavelengths, np.concatenate((np.zeros_like(obs_spectrum), templates[correlation_match[1]], np.zeros_like(obs_spectrum))))
    
    wavelengths    = wavelengths[offset_pix:len(obs_spectrum)+offset_pix]
    if inverted:
        spectrum_calib = np.flip(np.ravel(kwargs["obs_spectrum"]))
    else:
        spectrum_calib = np.ravel(kwargs["obs_spectrum"])
    
    fig.savefig("calibration_plotsheet.jpg", bbox_inches="tight", transparent=True, dpi=300)
    
    plt.show()
    
    fig = plt.figure(figsize=(100, 5))
    ax = fig.add_subplot()
    
    ax.plot(wavelengths, spectrum_calib)
    for wavelength, ion in zip(lit_lines["Wavelength"], lit_lines["Ion"]):
        ax.axvline(wavelength, color="tab:red", alpha=0.25, ls="dashed")
        ax.text(wavelength, ax.get_ylim()[1], ion + r" $\lambda$" + "%.1f" % wavelength, rotation="vertical", ha="right", va="top")
    
    ax.tick_params(which="both", axis="x", bottom=True, top=True)
    ax.tick_params(which="both", axis="y", left=True, right=True)
    ax.grid(True, ls="dashed", color="gray")
    ax.set_xlabel("Wavelength [$\mathrm{\mathring{A}}$]")
    ax.set_ylabel("Rel. flux [a. u.]")
    
    fig.savefig("calib_spectrum.pdf", transparent=True, bbox_inches="tight")
    plt.close()

    return wavelengths    

class observation:
    """The observation class. It provides a meaningful data structure to store the calibration data, master calbration files, as well and the calibrated data in
    """
    
    def __init__(self, spectrum_dir, flat_dir, calib_file, specs=DADOS_ST8_specs, **kwargs): # flat frame needs to be specified. Otherwise, cannot identify spectra location
        """Initialization of the observation class structure

        Args:
            spectrum_dir (str): The directory in which the science spectra are being stored
            flat_dir (str): The directory in which the flat frames are being stored
            calib_file (str): The file path to the wavelength calibration file
            specs (dict, optional): The dictionary that loads the hardware-specific parameters for the camera and spectrograph combination. Defaults to DADOS_ST8_specs.
        """

        self.spectrum_dir   = [spectrum_dir]    # inherits the directory containing the spectra from the functional __init__() argument
        self.flat_dir       = [flat_dir]        # inherits the directory containing the flat frames from the functional __init__() argument
        self.calib_file     = calib_file        # inherits the path to the calibration file from the functional __init__() argument
        self.darkflat_dir   = []                # defaults an empty list. If lists is not filled, no darkflat correction will be applied. One can proceed without such calibration, though, it is highly not recommended
        self.dark_dir       = []                # defaults an empty list. If lists is not filled, no dark correction will be applied. One can proceed without such calibration, though, it is highly not recommended

        self.spectrum       = None              # no spectrum is being generated to this point
        self.master_flat    = None              # no master flat is being generated to this point
        self.master_dark    = None              # no master dark is being generated to this point
        self.master_darkflat= None              # no master darkflat is being generated to this point
        self.calib          = fits.getdata(self.calib_file) # read in the wavelength calibration data from the file specified in the functional argument of __init__()
        self.specs          = specs             # load the specification file. The respective source of data is specified in the functional argument of __init__()

        self.orders         = []                # the vertical pixel coordinates of the order boundariers. Since no order identification ran yet, there is nothing, but an empty list.
        self.Norders_found  = None



    def add_darks(self, *directories):          # a simple function to add more dark directories after initialization
        for directory in directories: self.dark_dir.append(directory)



    def add_darkflats(self, *directories):      # a simple function to add more darkflats directories after initialization
        for directory in directories: self.darkflat_dir.append(directory)



    def add_flats(self, *directories):          # a simple function to add more flats directories after initialization
        for directory in directories: self.flat_dir.append(directory)



    def create_masterflat(self):                # create the master flat data from the data contained in the master_flat directories which were specified before
        self.master_flat        = create_masterfile(self.flat_dir)



    def create_masterdarkflat(self):            # create the master darkflat data from the data contained in the master_flat directories which were specified before
        self.master_darkflat    = create_masterfile(self.darkflat_dir)



    def create_masterdark(self):                # create the master dark data from the data contained in the master_flat directories which were specified before
        self.master_dark        = create_masterfile(self.dark_dir)



    def export_frame(self, data_array, filename): # conveience function to enable export of a specified data array (can be any )
        fits.writeto(data=data_array, filename=filename, overwrite=True)



    def calibrate(self, **kwargs):              # calibrate the spectral data with the given calibration data, which had to be created beforehand
        raw_spectrum            = create_masterfile(self.spectrum_dir)
        
        if self.master_flat is None:            # is there master flat data available? - if not, create an artificial lat frame consisting of only ones
            flat_correct        = np.ones_like(raw_spectrum)
        else:
            if self.master_darkflat is None:    # is there master darkflat data available? - if not, skip the darkflat correction
                flat_correct    = self.master_flat
            else:                               # is there master darkflat data available? - if so, apply the darkflat correction
                flat_correct    = self.master_flat - self.master_darkflat

        if self.master_dark is None:            # is there master dark data available? - if not, proceed without dark correction
            spectrum            = raw_spectrum / flat_correct          
        else:                                   # is there master dark data available? - if so, apply the dark correction
            spectrum            = (raw_spectrum - self.master_dark) / flat_correct

        self.spectrum           = spectrum                # set the "spectrum" attribute equal to the just genrated spectrum instance
    
    
    
    def findorder(self, output="", show_output=False, order_detection_threshold=300, order_poly_fit_degree=3):
        """Helper function to find the spectral orders of the spectrum image.

        Args:
            output (str, optional): The output file path in which the auxilliary files are being stored. Most of them are for debugging purposes and are not essential for the overall data reduction process. If Set to "" (default) no output will be generated. Defaults to "".
            show_output (bool, optional): Decides whether the axuilliary plots will be shown with the standard matplotlib qt interface. You may leave this parameter as false (default) if you do not intend to like to through all the windows by hand. Defaults to False.
            order_detection_threshold (int, optional): The threshold for the order detection. In the current verion, this is just the excess of signal above the background. Defaults to 300. I.e. The order's signal must must exceed the background level by at least this value. Defaults to 300.     
            order_poly_fit_degree (int, optional): The polynomial degree of the order location fit. Make sure to not overfit, i.e. choose a meaningfully low integer values here. Defaults to 3.
        """
        owidth          = int(self.specs["order_width"]) # read in the expected order width from the spectrograph's specifications
        if isinstance(self.master_flat, np.ndarray):
            processed_flat  = sig.medfilt2d(self.master_flat - self.master_darkflat) # apply median filter to denoise the flat image
        else:
            processed_flat  = sig.medfilt2d(self.master_flat) # apply median filter to denoise the flat image
        pixel_coord     = np.arange(self.master_flat.shape[0])[owidth:-owidth] # define the pixel coordinate axis. Because of the convolution, the flat image will be cropped on both ends by "owidth". Hence, create a list from owidth:length of uncropped frame - owidth
        
        flat_conv_cols  = [np.convolve(processed_flat[:,i], np.ones(owidth, dtype=float) / owidth, mode="same")[owidth:-owidth] for i in range(processed_flat.shape[1])] # convolve the flat image to identify the orders as the peaks of the convolved data. Image is sliced in columns after this step
        orders          = [sig.find_peaks(flat_conv_col, distance=owidth, height=order_detection_threshold)[0] for flat_conv_col in flat_conv_cols] # find the peaks in every successive pixel column of the image (we identify them as being the spectral orders of the spectrograph)
        ## ToDo: adjust the a height selection to filter false-positives

        peaks_found     = [] # create a new list with the coordinates found that seem to be associated with the 
        for i in range(len(orders)):
            peaks_found += [(i, pixel_coord[order]) for order in orders[i]] # transform peak position from the cropped coordinate frame back to the initial image coordinates
        peaks_found     = np.array(peaks_found).T

        peak_hist, bin_edges = np.histogram(peaks_found[1], bins=int(self.specs["Norders"] * 100.), range=(0, processed_flat.shape[0])) # vertical cut-through: create the histogram of the cross-signal, any signal peak here will correspont to an individual spectral order
        bin_centers     = (bin_edges[:-1] + bin_edges[1:])/2.
        bin_width       = float(bin_edges[1] - bin_edges[0])
        
        order_positions, _ = sig.find_peaks(peak_hist, height=order_detection_threshold, distance=float(owidth) / bin_width) # ToDo: redefine the "height" parameter as the threshold will is not indpendent on the histogram binning 
        self.Norders_found = len(order_positions)
        if self.Norders_found == self.specs["Norders"]: # check if all orders have been found
            print("[SUCCESS]\tExactly identified the number of orders provided by the spectrograph specifications!")
        elif self.Norders_found < self.specs["Norders"]:
            print("[WARNING] Unable to identify all orders. Less orders found than specified! Consider to decrease the 'order_detection_threshold'")
        else:
            print("[WARNING] More orders found than specified than by the spectrograph provided! Consider to increase the 'order_detection_threshold' as some of the orders might be just systematic noise peaks")
        
        order_center_coords = [] # this array will contain "len(order_positions)"-many subarrays, each containing coordinate tuples for the spectrum's image which are associated with each seperate order
        for peak in order_positions: # for each peak in the signal from the preceeding step, which is now considered to be a valid spectral order do ...
            order_lims      = (bin_centers[peak] - owidth / 2., bin_centers[peak] + self.specs["order_width"] / 2.) # define the area in which the signal is considered to belong to a certain order
            indices         = np.squeeze(np.argwhere((peaks_found[1] >= order_lims[0]) & (peaks_found[1] <= order_lims[1]))) # filter all points which belong to the current order
            order_center_coords.append(peaks_found[:, indices]) # append the list of coordinates to represent the whole spectral order

        order_center_fits   = []
        counter             = 1
        for order_coords in order_center_coords:
            coefficients    = np.polyfit(*order_coords, deg=order_poly_fit_degree) # coefficients for the polynomial fit function
            order_center_fits.append(coefficients)
            print(f"[INFORMATION] Spectrum {counter} seems to be tilted by an angle of {180/np.pi * np.arcsin(coefficients[-2]):.2f} degrees.") # take the linear term and calculate the tilt of the linear spectra against the image coordinate system.
            counter         += 1

        if output !="": # enable graphical output of the calibration flat data
            fig = plt.figure(dpi=200, figsize=(12, 9))
            ax = fig.add_subplot()
            ax.imshow(processed_flat)
            ax.scatter(peaks_found.T[0], peaks_found.T[1], s=1)\
            
            for i in range(len(order_center_coords)):
                ax.plot(order_center_coords[i][0], order_center_coords[i][1], alpha=0.25, c="k") # plot all data points that have been found
                ax.plot(order_center_coords[i][0], np.poly1d(order_center_fits[i])(order_center_coords[i][0])) # draw the central lines of each order
                
                ax.plot(order_center_coords[i][0], np.poly1d(order_center_fits[i])(order_center_coords[i][0]) + owidth/2., c="gray", linestyle="--") # draw the upper boundary of each order
                ax.plot(order_center_coords[i][0], np.poly1d(order_center_fits[i])(order_center_coords[i][0]) - owidth/2., c="gray", linestyle="--") # draw the lower boundary of each order
            
            fig.savefig(output, bbox_inches="tight")
            
            if show_output:
                plt.show()
            else:
                plt.close()
        
        self.orders = order_center_fits # save the coefficients of the polynomial fits to the self.orders instance an make them available for later steps
    


    def extract_flux(self, filename, **kwargs):
        """This is a helper routine that extracts the fluxes from the individual orders of the image. In oder to run this routine, you must first execute the "find_orders" routine. It uses aperture photometry.

        Args:
            filename (str): The filename of the the file to be read out.

        Raises:
            Exception: _description_
            Exception: _description_

        Returns:
            np.ndarray: A list of the one-dimensional signal arrays containing the fluxes of each order.
        """
        
        x = np.arange(self.master_flat.shape[1]) # x-axis definition, reference to the bottom image axis
        try:
            order_fits      = [np.poly1d(params) for params in self.orders] # read in the polynomial coefficients which should be available from the order identification process
        except:
            raise Exception("[ERROR] An error occurred while trying to read the order location parameters. Make sure you ran the order identification routine before you continues with the wavelenggth calibration")
        theta               = [[order_fit.deriv(m=1)(x_coord) for x_coord in x] for order_fit in order_fits] # the first derivative of the polynomial fits yield the rotation at image pixel (x,y)
        y                   = [[order_fit(x_coord) for x_coord in x] for order_fit in order_fits] # y is two-dim (e.g. for DADOS (3_orders x image_width))

        apertures           = []
        
        if "order" in kwargs:
            orders_to_extract = [kwargs["order"]]
        else:
            orders_to_extract = range(self.Norders_found)
        
        for n in orders_to_extract: 
            apertures.append(np.array([RectangularAperture((x[i], y[n][i]), 1, self.specs["order_width"], theta[n][i]) for i in range(len(x))])) # define ALL apterure to probe through the image
        if not isinstance(self.calib, np.ndarray): raise Exception("Wavelength calibration image does not exist!")

        signal              = []
        for i in range(len(orders_to_extract)): # for every order ...
            signal_data     = np.ravel([aperture_photometry(fits.getdata(filename), apertures[i][k])["aperture_sum"] for k in tqdm(range(len(x)))]) # read in the data from the image ... within the aperture that are already defined
            ########### plt.plot(signal_data, label="%i" % i)
            signal.append(signal_data)
        
        self.signals = signal.copy()
        self.extracted_orders = np.array(orders_to_extract)
        
        return signal



    """
    def wavelength_calib(self): # specify the order to use, use 2 for the center, highest resolution one
        x = np.arange(self.master_flat.shape[1]) # x-axis definition, reference to the bottom image axis
        try:
            order_fits      = [np.poly1d(params) for params in self.orders] # read in the polynomial coefficients which should be available from the order identification process
        except:
            raise Exception("[ERROR] An error occurred while trying to read the order location parameters. Make sure you ran the order identification routine before you continues with the wavelenggth calibration")
        theta               = [[order_fit.deriv(m=1)(x_coord) for x_coord in x] for order_fit in order_fits] # the first derivative of the polynomial fits yield the rotation at image pixel (x,y)
        y                   = [[order_fit(x_coord) for x_coord in x] for order_fit in order_fits] # y is two-dim (e.g. for DADOS (3_orders x image_width))

        apertures = []
        for n in range(self.Norders_found): 
            apertures.append(np.array([RectangularAperture((x[i], y[n][i]), 1, self.specs["order_width"], theta[n][i]) for i in range(len(x))])) # define ALL apterure to probe through the image
        if not isinstance(self.calib, np.ndarray): raise Exception("Wavelength calibration image does not exist!")

        signal              = []
        for i in range(self.Norders_found): # for every order ...
            signal_data     = np.array([aperture_photometry(self.calib, apertures[i][k])["aperture_sum"] for k in range(len(x))]) # read in the data from the image ... within the aperture that are already defined
            ########### plt.plot(signal_data, label="%i" % i)
            signal.append(signal_data)
        print(self.Norders_found)
        #[plt.plot(signal[i]) for i in range(self.Norders_found)]
        #plt.plot(signal[0], color="r")
        plt.plot(-3.506*(np.arange(len(signal[1])) - 688) + 6562.8, signal[1] - np.percentile(signal[1], 1), color="g")
        #plt.plot(signal[2], color="b")
        #plt.gca().invert_xaxis()

        signal_to_plot      = self.extract_flux("./testspec/spectra/Sirius-0010_5.fit")
        plt.plot(-3.5055114655*(np.arange(len(signal[1])) - 688) + 6562.8, signal_to_plot[1] - np.percentile(signal_to_plot[1], 1))
        plt.axvline(688)
        plt.axvline(1163.5)

        plt.legend()
        plt.show()
        for i in range(self.Norders_found): print(signal[i])
        """



    def cross_match(self, lit_lines, output="", show_output=False, **kwargs):
        """Feed an list of lines and their relative strengths from observations and the literature.
        
        - obs_spectrum (numpy 1d array): The spectrum as a list of intensities
        - obs_lines (numpy list of peaks, in the style of the output of the scipy.find_peaks function. I.e. provide a 1d array of indices that correspond to the stencils of a respective 1d array at which those peaks are found). This is an alternative input to the "obs_spectrum" input
        - obs_length (int) This number determines the length of the spectrum array. This parameter is only required when you entered an input for the "obs_lines" argument of this function. Otherwise it will be ignored.
        - lit_lines: Astropy Table containing the line wavelengths and qualitative line strengths as well as the ion that they are referring to.
        
        The arguments should be in the form of two tuples, containing two lists each, e.g. ([wavelengths], [strengths]) ... blablabla
        
        This routine returns a transformation that relates the pixel coordinate the frame with an absolute wavelength.
        """

        # Input parameters ...
        peak_threshold      = 0.1 # multiple of the standard deviation of the spectrum (this is a parameter for the peak detection in the observation spectrum)
        wavelength_range    = (5000., 6800.)
        sampling            = 3.6 # Angstrom per pixel
        sampling_range      = (0.9, 1.1) # in multiples of the expected wavelength dispersion
        psf_width           = 2.
        resolution          = 1500.
        min_template_strength_level = 2

        if isinstance(lit_lines, str):
            print("[INFO] Entered a string as the literature input. Assume it directs to a file. Attempting to read:", lit_lines)
            
            lit_lines           = asc.read(lit_lines, delimiter="\t")
        
        elif isinstance(lit_lines, Table):
            print("[INFO] The literature data seems to be an Astropy Table. Proceed assuming the correct naming pattern ...")
        
        # Apparently the user specified a spectrum that they want to use as a comparison to the template as we take the calibration file for that instance
        obs_spectra    = self.extract_flux(self.calib_file)
        
        self.calib_spectra = obs_spectra.copy()
        
        all_wavelengths = []
        for order_num, obs_spectrum in enumerate(obs_spectra):
            obs_spectrum        -= np.median(obs_spectrum)

            signal_std          = np.std(obs_spectrum)
            
            print("[INFO] Std =", signal_std)
            print(type(obs_spectrum), obs_spectrum)
            
            obs_lines, lines_dict   = sig.find_peaks(np.ravel(obs_spectrum), height=signal_std * peak_threshold, prominence=5)
            
            
            obs_spectrum        = np.zeros_like(obs_spectrum)
            # obs_spectrum[peaks] = peaks_dict["peak_heights"] / peaks_dict["peak_heights"] ######################## comment out the normalization if you wish to have peaks with different sizes (stronger peaks are correspondingly larger)
            obs_spectrum[obs_lines] = 1.
            
            if output != "":
                fig = plt.figure(figsize=(10, 14.14))
                
                gs = fig.add_gridspec(ncols=3, nrows=4, width_ratios=[10, 10, 1], wspace=0.05)
                
                # start with the first axis that displays the peak positions of the observed calibration spectrum
                ax                  = fig.add_subplot(gs[0, :])
            
            # blur the spectrum such that the delta peaks become gaussian lines. This avoids that neighboring peaks are not identified as those
            obs_spectrum        = gaussian_filter(obs_spectrum, psf_width)
            
            if output != "":
                ax.step(np.arange(len(obs_spectrum)), obs_spectrum)
            
                # the peaks that have been found with indicators
                ax.plot(obs_lines, obs_spectrum[obs_lines], marker="o", mew=1, mfc="none", markersize=5, lw=0, color="tab:red")
            
                ax.set_xlabel("Pixel coordinate")
                ax.set_ylabel("Observational Signal (re-modeled and de-noised)")
            
            # create the template array to correlate the observed spectrum with
            
            mid_wavelength      = np.mean(wavelength_range)
            template_lengths    = np.arange(
                int((wavelength_range[1] - wavelength_range[0]) / (sampling * np.max(sampling_range))),
                int((wavelength_range[1] - wavelength_range[0]) / (sampling * np.min(sampling_range))),
                1,
                dtype=int
            )
            ##############
            
            wavelength_increments = []
            templates_wavelengths = []
            templates             = []
            for length in template_lengths:
                template_wavelengths = np.linspace(*wavelength_range, length)
                templates_wavelengths.append(template_wavelengths)
                
                wavelength_increment = template_wavelengths[1] - template_wavelengths[0] # lambda per pixel
                wavelength_increments.append(wavelength_increment)
                
                strong_lines_mask = lit_lines["Strength"] >= min_template_strength_level # only consider really strong lines in the template
                
                lines_digitized   = np.digitize(lit_lines["Wavelength"][strong_lines_mask], template_wavelengths)
                
                strength_exponent = 0. # set this value to zero to normalize all the peaks to 1.
                
                template_data   = np.zeros(template_wavelengths.size + 1) # add one for the overflow bin
                template_data[lines_digitized] = lit_lines["Strength"][strong_lines_mask]**strength_exponent
                template_data   = gaussian_filter(template_data[:-1], mid_wavelength / (2. * resolution)) # cut off the last bin since this is the overflow bin
                templates.append(template_data)

            number_of_correlations = len(wavelength_increments)

            if output != "":
                # plot the shape of the template that we will correlate the spectrum with
                ax = fig.add_subplot(gs[1, :])
                plt.step(np.arange(template_data.size), template_data)
            
            
            corr_filler         = np.zeros(np.max(template_lengths)) # append this to the end and the beginning of the array to be correlated. This avoid unwanted edge-effects ...
            corr_signals        = [
                np.correlate(
                    np.concatenate((corr_filler, obs_spectrum, corr_filler)),
                    template,
                    mode="same"
                )[corr_filler.size:-corr_filler.size] for template in tqdm(templates) # chop off the left and right edge that are effected by boundary effects anyways.
            ]
            corr_signals_inv    = [
                np.correlate(
                    np.concatenate((corr_filler, obs_spectrum, corr_filler)),
                    np.flip(template),
                    mode="same"
                )[corr_filler.size:-corr_filler.size] for template in tqdm(templates) # chop off the left and right edge that are effected by boundary effects anyways.
            ]
            
            if output != "":
                ax                  = fig.add_subplot(gs[2, 0]) # the axis in which the correlation plot will be plotted at
                ax_inv              = fig.add_subplot(gs[2, 1]) # the axis in which the inverse correlation will be plotted at
            
            # generate the correlation array. It has the dimensions: [2 (0=no template inversion, 1= with template inversion), number_of_correlations, len(obs_spectrum)]
            correlation         = np.array([corr_signals, corr_signals_inv])
            
            # find teh maximum correlation value
            max_corr            = np.max(correlation)
            
            if output != "":
                img = ax.pcolormesh(
                    np.arange(correlation.shape[2]), wavelength_increments, correlation[0, :, :], vmax=max_corr, rasterized=True
                )
            
                ax.set_xlabel("pixel offset")
                ax.set_ylabel(r"$\Delta\mathrm{\mathring{A}}/\Delta\mathrm{px}$")

                img_inv = ax_inv.pcolormesh(
                    np.arange(correlation.shape[2]), wavelength_increments, correlation[1, :, :], vmax=max_corr, rasterized=True
                )
            
                ax_inv.set_yticklabels([])
                ax.tick_params(axis="x", which="both", direction="in", bottom=True, top=True)
                ax.tick_params(axis="y", which="both", direction="in", left=True, right=True)
                ax_inv.tick_params(axis="x", which="both", direction="in", bottom=True, top=True)
                ax_inv.tick_params(axis="y", which="both", direction="in", left=True, right=True)
            
                cax = fig.add_subplot(gs[2, 2])
                plt.colorbar(img, cax=cax, pad=0)
                
            
            correlation_sorted = np.array(np.unravel_index(
                np.argsort(correlation, axis=None),
                correlation.shape
            ))
            
            # in the correlation array, find the respective index 
            correlation_match = correlation_sorted[:, -1]
            inverted          = correlation_match[0] == 1 # check if the template has to be inverted in order to get the observations to match in correlation matrix
            
            # this is the wavelength scale at which the correlation function maximizes
            wavelength_increment_match = wavelength_increments[correlation_match[1]] # this is a floating point value
            
            if output != "":
                if inverted:
                    ax_inv.axvline(correlation_match[2], ls="dashed", color="tab:red", lw=0.5) # draw a red indicator at the point where the correlation maximizes (this line inciates the pixel offset of the template with respect to the observed spectrum)
                    ax_inv.axhline(wavelength_increment_match, ls="dashed", color="tab:red", lw=0.5) # draw a red indicator at the point where the correlation maximizes (this line characterizes the optimal wavelength sampling of the template with respect to the observed spectrum)
                    
                    ax_inv.text(correlation_match[2], ax_inv.get_ylim()[1], r"$\mathrm{offset}" + "=%i$" % correlation_match[2], color="white", ha="left", va="top")
                    ax_inv.text(ax_inv.get_xlim()[0], wavelength_increment_match, r"$\Delta\lambda/\Delta\mathrm{px}=" + "%.4f$" % wavelength_increment_match, color="white", ha="left", va="bottom")
                
                else:
                    ax.axvline(correlation_match[2], ls="dashed", color="tab:red", lw=0.5) # draw a red indicator at the point where the correlation maximizes (this line inciates the pixel offset of the template with respect to the observed spectrum)
                    ax.axhline(wavelength_increments[correlation_match[1]], ls="dashed", color="tab:red", lw=0.5) # draw a red indicator at the point where the correlation maximizes (this line characterizes the optimal wavelength sampling of the template with respect to the observed spectrum)
                    ax.text(correlation_match[2], ax.get_ylim()[1], r"$\mathrm{offset}" + "=%i$" % correlation_match[2], color="white", ha="left", va="top")
                    ax.text(ax.get_xlim()[0], wavelength_increment_match, r"$\Delta\lambda/\Delta\mathrm{px}=" + "%.4f$" % wavelength_increment_match, color="white", ha="left", va="bottom")
                    
                ax = fig.add_subplot(gs[3, :])
            
            # find the correct pixel offset
            reference = np.zeros((len(obs_spectrum) + template_lengths[correlation_match[1]], 2 * len(obs_spectrum) + template_lengths[correlation_match[1]])) # template stays fixed. Then shift the spectrum through
            
            print(len(obs_spectrum)+template_lengths[correlation_match[1]], reference.shape)
            for i in range(len(obs_spectrum)+template_lengths[correlation_match[1]]):
                if inverted:
                    reference[i, i:i+len(obs_spectrum)] = np.flip(obs_spectrum)
                else:
                    reference[i, i:i+len(obs_spectrum)] = obs_spectrum
            
            reference_sum = np.expand_dims(np.concatenate((np.zeros_like(obs_spectrum), templates[correlation_match[1]], np.zeros_like(obs_spectrum))), axis=0)
            
            reference_sum = reference_sum * reference
            reference_sum = np.sum(reference_sum, axis=1)
            
            offset_pix = int(np.argwhere(reference_sum == np.max(reference_sum)))
            
            wavelengths = np.arange(
                wavelength_range[0] - (len(obs_spectrum) * wavelength_increment_match),
                wavelength_range[1] + (len(obs_spectrum) * wavelength_increment_match) + wavelength_increment_match/1000., # add a tiny bit to allow the arange function to include the upper limit also ... othwise it is being cropped off ...
                wavelength_increment_match
                ) # the uncropped wavelength axis

            if output != "":
                ax.plot(wavelengths, reference[offset_pix])
                ax.plot(wavelengths, np.concatenate((np.zeros_like(obs_spectrum), templates[correlation_match[1]], np.zeros_like(obs_spectrum))))
            
            wavelengths    = wavelengths[offset_pix:len(obs_spectrum)+offset_pix]
            if inverted:
                wavelengths = np.flip(wavelengths)
            
            if output != "":
                fig.savefig(output + "_%s.pdf" % order_num, bbox_inches="tight", transparent=True, dpi=300)
            
            if show_output:
                plt.show()
            else:
                plt.close()
            
            # matplotlib plot style parameters
            rcParams["text.usetex"]     = False
            rcParams['font.family']     = 'serif'
            rcParams['font.serif']      = 'Palatino'
            
            if output != "":
                fig = plt.figure(figsize=(100, 5))
                ax = fig.add_subplot()
            
                ax.plot(wavelengths, obs_spectrum)
                for wavelength, ion in zip(lit_lines["Wavelength"], lit_lines["Ion"]):
                    ax.axvline(wavelength, color="tab:red", alpha=0.25, ls="dashed")
                    ax.text(wavelength, ax.get_ylim()[1], ion + r" $\lambda$" + "%.1f" % wavelength, rotation="vertical", ha="right", va="top")
            
                ax.tick_params(which="both", axis="x", bottom=True, top=True)
                ax.tick_params(which="both", axis="y", left=True, right=True)
                ax.grid(True, ls="dashed", color="gray")
                ax.set_xlabel("Wavelength [$\mathrm{\mathring{A}}$]")
                ax.set_ylabel("Rel. flux [a. u.]")
            
                fig.savefig("calibration_spectrum_%s.pdf" % order_num, transparent=True, bbox_inches="tight")
            
            if show_output:
                plt.show()
            else:
                plt.close()
            
            all_wavelengths.append(wavelengths)

        self.wavelengths = all_wavelengths

        return all_wavelengths



    def func_pix2wav(self, degree=1, **kwargs):
        """This function returns a function that does the conversion from pixel coordinates to wavelengths.
        

        kwargs:
            pixels (np.ndarray): An array in which pixel coordinates are stored
            wavelengths (np.ndarray): An array with the corresponding wavelengths
        """
        
        orders_to_iterate = range(len(self.wavelengths))
        
        if ("wavelengths" in kwargs) and ("pixels" in kwargs):
            pixels      = kwargs["pixels"].copy()
            wavelengths = kwargs["wavelengths"]
        else:
            pixels      = [np.arange(self.calib_spectra[i].size) for i in orders_to_iterate]
            wavelengths = [np.asarray(self.wavelengths[i], dtype=float) for i in orders_to_iterate]


        poly_parameters = [np.polyfit(pixels[i], wavelengths[i], deg=degree) for i in orders_to_iterate]
        polyfunc        = [np.poly1d(poly_parameters[i]) for i in orders_to_iterate]
        
        return polyfunc
    
    
    
    def func_wav2pix(self, degree=1, **kwargs):
        """_summary_

        Args:
            degree (int, optional): _description_. Defaults to 1.
        """
        
        orders_to_iterate = range(len(self.wavelengths))
        if ("wavelengths" in kwargs) and ("pixels" in kwargs):
            pixels      = kwargs["pixels"].copy()
            wavelengths = kwargs["wavelengths"]
        else:
            pixels = [np.arange(self.calib_spectra[i].size) for i in orders_to_iterate]
            wavelengths = [np.asarray(self.wavelengths[i], dtype=float) for i in orders_to_iterate]

        poly_parameters = [np.polyfit(wavelengths[i], pixels[i], deg=degree) for i in orders_to_iterate]
        polyfunc = [np.poly1d(poly_parameters[i]) for i in orders_to_iterate]
        
        return polyfunc



    def check_and_refine_wavelength_solution(self, lit_lines, **kwargs):
        
        min_peak_dist = 5
        min_line_strength = 1
        poly_fit_degree = 3
        
        if isinstance(lit_lines, str):
            print("[INFO] Entered a string as the literature input. Assume it directs to a file. Attempting to read:", lit_lines)
                
            lit_lines           = asc.read(lit_lines, delimiter="\t")
            
        elif isinstance(lit_lines, Table):  # Table with columns: ['Wavelength', 'Ion', 'Strength']
            print("[INFO] The literature data seems to be an Astropy Table. Proceed assuming the correct naming pattern ...")
        
        all_line_peak_pairs = []
        
        for k, spectrum in enumerate(self.calib_spectra):
            peaks, _ = sig.find_peaks(spectrum, distance=min_peak_dist)
            peaks = np.expand_dims(peaks, axis=1)
            print(peaks, peaks.shape)
            
            tree = cKDTree(peaks)
            lit_lines_selected = lit_lines[lit_lines["Strength"] >= min_line_strength]
            
            matches = tree.query_ball_point(self.func_wav2pix()[k](np.expand_dims(lit_lines_selected["Wavelength"], axis=1)), r=min_peak_dist)
            
            line_peak_pairs = []
            for i, match in enumerate(matches):
                if len(match) == 1:
                    line_peak_pairs.append((lit_lines_selected["Wavelength"][i], float(peaks[match])))
        
            all_line_peak_pairs.append(np.array(line_peak_pairs))

        pix2wav_funcs = [np.poly1d(np.polyfit(pairs[:, 1], pairs[:, 0], deg=poly_fit_degree)) for pairs in all_line_peak_pairs]
        self.pix2wav_funcs = pix2wav_funcs

        fig, ax = plt.subplots()
        ax.set_title("Wavelength calibration refinement")
        
        for i, pairs in enumerate(all_line_peak_pairs):
            plot_range = np.linspace(np.min(pairs[:, 1]), np.max(pairs[:, 1]), 100)
            #          Pixel        Wavelength
            ax.scatter(pairs[:, 1], pairs[:, 0], label=str(i), marker="x", color="C%i" % i)
            ax.plot(plot_range, self.pix2wav_funcs[i](plot_range), color="C%i" % i)
            
            
            
        ax.legend()
        plt.show()


obs = observation(spectrum_dir="testspec/spectra/", flat_dir="testspec/flats/", calib_file="testspec/wavelength_calib.fit")
obs.add_darks("testspec/darks/")
obs.add_darkflats("testspec/darks/")
obs.create_masterdark()
obs.create_masterflat()
obs.create_masterdarkflat()
obs.findorder(output="output/orderfind.jpg", order_poly_fit_degree=1, order_detection_threshold=300)

wavelengths = obs.cross_match("linelist/manufacturer_lines.dat", output="output/calibration_plot.pdf")

obs.check_and_refine_wavelength_solution("linelist/manufacturer_lines.dat")

#signal = np.flip(np.ravel(obs.extract_flux("testspec/spectra/Sirius-0001_5.fit")))
#flat_signal = np.flip(np.ravel(obs.extract_flux("testspec/flats/Sirius-0002_5s.fit")))
#obs.wavelength_calib()
#obs.calibrate()
#obs.export_frame(obs.spectrum, "calib_spectrum.fit")

fig, ax = plt.subplots(figsize=(20, 5))

ax.set_title("Spectrum of Sirius (2022-02-12 18:50 UTC)")

for i in range(3):
    #calib_signal = signal / flat_signal

    #ax.step(wavelengths[1], calib_signal / np.max(calib_signal), label=str(i))
    ax.step(obs.pix2wav_funcs[i](np.arange(len(obs.calib_spectra[i]))), obs.calib_spectra[i] / np.max(obs.calib_spectra[i]), label=str(i))
    ax.tick_params(which="both", axis="x", bottom=True, top=True)
    ax.tick_params(which="both", axis="y", left=True, right=True)
    ax.grid(True, ls="dashed", color="gray")
    ax.set_xlabel("Wavelength [$\mathrm{\mathring{A}}$]")
    ax.set_ylabel("Rel. flux [a. u.]")
    ax.set_ylim(bottom=0)
    ax.legend()
    
    np.savetxt("output/spectrum_%i.dat" % i, np.array([wavelengths[i], obs.calib_spectra[i] / np.max(obs.calib_spectra[i])]).T)

fig.savefig("output/sirius_spectrum.pdf", bbox_inches="tight", transparent=True)

plt.show()